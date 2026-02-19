"""RAG service using LangGraph for agent orchestration."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from ..config import settings
from .embedding_service import embedding_service
from .qdrant_service import qdrant_service
from .knowledge_graph_service import knowledge_graph_service
from ..models import Agent, DataDomain

logger = structlog.get_logger()


class RAGState(TypedDict):
    """State for the RAG graph.

    ``messages`` tracks the LangGraph message history for the ReAct
    tool-calling loop (used when the agent has tools configured).
    The ``add_messages`` reducer appends new messages automatically.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    agent_config: dict[str, Any]
    retrieved_docs: list[dict[str, Any]]
    response: str
    sources: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int


class RAGService:
    """Service for RAG operations using LangGraph."""

    def __init__(self):
        self._llm_cache: dict[str, Any] = {}

    def get_llm(self, model: str, temperature: float = 0.7, max_tokens: int = 2048):
        """Get or create an LLM instance using the configured OpenAI-compatible endpoint."""
        cache_key = f"{model}_{temperature}_{max_tokens}"

        if cache_key not in self._llm_cache:
            self._llm_cache[cache_key] = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_API_BASE,
            )
            logger.info(
                "Created LLM instance",
                model=model,
                temperature=temperature,
                base_url=settings.LLM_API_BASE,
            )

        return self._llm_cache[cache_key]

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from LLM response, with fallbacks when content is empty or structured."""
        if response is None:
            logger.warning("LLM returned None response")
            return ""
        content = getattr(response, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if content is not None and isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
            if parts:
                return "\n".join(parts).strip()
        if hasattr(response, "text") and isinstance(getattr(response, "text"), str):
            text = getattr(response, "text", "").strip()
            if text:
                logger.info("Used response.text fallback for LLM response")
                return text
        repr_str = str(response).strip()
        if repr_str and repr_str not in ("None", ""):
            logger.warning(
                "LLM response.content empty; used str(response) fallback",
                response_type=type(response).__name__,
                repr_preview=repr_str[:300],
            )
            return repr_str
        logger.warning(
            "LLM response had no extractable text",
            response_type=type(response).__name__,
            has_content=hasattr(response, "content"),
            content_type=type(getattr(response, "content", None)).__name__,
        )
        return ""

    def retrieve_documents(
        self, query: str, data_domain: DataDomain, k: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents from the vector store."""
        try:
            # Generate query embedding
            query_embedding = embedding_service.embed_query(
                query, data_domain.embedding_model
            )

            # Search in Qdrant
            results = qdrant_service.search(
                collection_name=data_domain.qdrant_collection,
                query_vector=query_embedding,
                limit=k,
                score_threshold=0.3,  # Minimum relevance score
            )

            # Log each retrieved chunk (payload keys: content or text, source_file or filename)
            for i, doc in enumerate(results):
                payload = doc.get("payload", {})
                text = payload.get("content", "") or payload.get("text", "")
                filename = payload.get("source_file", "") or payload.get("filename", "Unknown")
                score = doc.get("score")
                logger.info(
                    "Retrieved chunk",
                    index=i + 1,
                    total=len(results),
                    filename=filename,
                    score=score,
                    content_preview=(text[:200] + "…") if len(text) > 200 else text,
                    content_length=len(text),
                )

            logger.info(
                "Retrieved documents",
                query_length=len(query),
                results_count=len(results),
                collection=data_domain.qdrant_collection,
            )

            return results
        except Exception as e:
            logger.error("Document retrieval failed", error=str(e))
            return []

    def retrieve_documents_multi(
        self, query: str, data_domains: list[DataDomain], k: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve from multiple domains, merge by score, return top k."""
        if not data_domains:
            return []
        all_results: list[tuple[float, dict]] = []
        k_per_domain = max(1, (k + len(data_domains) - 1) // len(data_domains))
        for domain in data_domains:
            docs = self.retrieve_documents(query, domain, k=k_per_domain)
            for doc in docs:
                score = doc.get("score") or 0.0
                all_results.append((score, doc))
        all_results.sort(key=lambda x: -x[0])
        return [doc for _, doc in all_results[:k]]

    def find_reasoning_path(
        self, query: str, retrieved_docs: list[dict[str, Any]]
    ) -> list[tuple[str, str]] | None:
        """
        Find shortest reasoning path using knowledge graph.

        Args:
            query: User query
            retrieved_docs: Retrieved documents from Qdrant

        Returns:
            Reasoning path or None
        """
        try:
            # Extract chunk contents from retrieved documents
            chunk_contents = []
            for doc in retrieved_docs:
                payload = doc.get("payload", {})
                content = payload.get("content", "") or payload.get("text", "")
                if content:
                    chunk_contents.append(content)

            if not chunk_contents:
                return None

            # Find reasoning path using knowledge graph
            path = knowledge_graph_service.find_reasoning_path(query, chunk_contents)

            return path
        except Exception as e:
            logger.error("Reasoning path finding failed", error=str(e))
            return None

    def format_context(self, documents: list[dict[str, Any]]) -> str:
        """Format retrieved documents as context for the LLM."""
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            payload = doc.get("payload", {})
            # Worker stores content/source_file; support text/filename for other ingest paths
            text = payload.get("content", "") or payload.get("text", "")
            filename = payload.get("source_file", "") or payload.get("filename", "Unknown")
            page = payload.get("page", "N/A")

            context_parts.append(f"[Source {i}: {filename}, Page {page}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)
        logger.info(
            "Formatted context for LLM",
            num_sources=len(documents),
            total_chars=len(context),
            context_preview=context[:500] + "…" if len(context) > 500 else context,
        )
        return context

    def create_rag_chain(
        self, agent: Agent, data_domains: list[DataDomain] | DataDomain
    ):
        """Create a RAG chain using LangGraph.

        When the agent has ``tools`` configured, builds a ReAct-style tool-calling
        graph.  Otherwise falls back to the classic retrieve → generate pipeline
        so existing behaviour is preserved.

        ``data_domains`` can be a list or single domain.
        """
        domains = (
            list(data_domains)
            if isinstance(data_domains, (list, tuple))
            else [data_domains]
        )

        # Resolve agent tools (if any)
        agent_tools = getattr(agent, "tools", None) or []
        if agent_tools:
            return self._create_tool_calling_chain(agent, domains, agent_tools)

        return self._create_classic_chain(agent, domains)

    # ------------------------------------------------------------------
    # Classic retrieve → generate pipeline (backward compatible)
    # ------------------------------------------------------------------

    def _create_classic_chain(
        self, agent: Agent, domains: list[DataDomain]
    ):
        """Build the original retrieve → generate LangGraph chain."""

        def retrieve_node(state: RAGState) -> RAGState:
            """Retrieve relevant documents from all domains."""
            docs = (
                self.retrieve_documents_multi(
                    state["query"], domains, k=agent.retrieval_k
                )
                if len(domains) > 1
                else self.retrieve_documents(
                    state["query"], domains[0], k=agent.retrieval_k
                )
            )
            state["retrieved_docs"] = docs

            # Add retrieved chunks to knowledge graph (incremental building)
            for doc in docs:
                payload = doc.get("payload", {})
                content = payload.get("content", "") or payload.get("text", "")
                if content:
                    knowledge_graph_service.add_chunk_to_graph(
                        content,
                        chunk_id=doc.get("id"),
                        extract_relationships=True
                    )

            # Find reasoning path using knowledge graph
            reasoning_path = self.find_reasoning_path(state["query"], docs)
            if reasoning_path:
                state["agent_config"]["reasoning_path"] = reasoning_path
                logger.info(
                    "Found reasoning path",
                    path_length=len(reasoning_path),
                    query=state["query"][:100],
                )

            return state

        def generate_node(state: RAGState) -> RAGState:
            """Generate response using LLM."""
            llm = self.get_llm(
                agent.llm_model,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
            )

            context = self.format_context(state["retrieved_docs"])

            reasoning_info = ""
            if "reasoning_path" in state["agent_config"]:
                path = state["agent_config"]["reasoning_path"]
                if path:
                    entities = [f"{e[0]} ({e[1]})" for e in path]
                    reasoning_info = f"\n\nReasoning Path (entity connections): {' -> '.join(entities)}"

            system_message = f"""{agent.system_prompt}

Use the following context to answer the user's question. If the context doesn't contain relevant information, say so.
Always cite your sources by mentioning the document name and page number when providing information.
{reasoning_info}

Context:
{context}"""

            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=state["query"]),
            ]

            response = llm.invoke(messages)

            input_tokens = 0
            output_tokens = 0
            if hasattr(response, "usage_metadata"):
                input_tokens = response.usage_metadata.get("input_tokens", 0)
                output_tokens = response.usage_metadata.get("output_tokens", 0)

            sources = []
            seen_sources = set()
            for doc in state["retrieved_docs"]:
                payload = doc.get("payload", {})
                document_id = payload.get("document_id") or 0
                page = payload.get("page")
                filename = payload.get("filename") or payload.get("source_file", "Unknown")
                text = payload.get("text") or payload.get("content", "")

                source_key = (document_id, page)
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    sources.append(
                        {
                            "document_id": document_id,
                            "filename": filename,
                            "page": page,
                            "chunk_text": (text[:200] + "...") if len(text) > 200 else text,
                            "score": doc.get("score", 0.0),
                        }
                    )

            state["response"] = self._extract_response_text(response)
            state["sources"] = sources
            state["input_tokens"] = input_tokens
            state["output_tokens"] = output_tokens

            return state

        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("generate", generate_node)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # ReAct / tool-calling pipeline
    # ------------------------------------------------------------------

    def _create_tool_calling_chain(
        self,
        agent: Agent,
        domains: list[DataDomain],
        tool_names: list[str],
    ):
        """Build a ReAct-style LangGraph chain that binds the agent's tools to
        the LLM and iterates between the model and tool execution nodes."""
        from langgraph.prebuilt import ToolNode
        from ..tools import get_tools_by_names

        tools = get_tools_by_names(tool_names)
        if not tools:
            logger.warning(
                "No valid tools resolved for agent, falling back to classic chain",
                agent_id=agent.id,
                requested_tools=tool_names,
            )
            return self._create_classic_chain(agent, domains)

        llm = self.get_llm(
            agent.llm_model,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
        )
        llm_with_tools = llm.bind_tools(tools)

        def agent_node(state: RAGState) -> dict:
            """Invoke the LLM (with tool bindings) on the current messages."""
            system_prompt = agent.system_prompt or "You are a helpful assistant."
            msgs = [SystemMessage(content=system_prompt)] + list(state["messages"])
            response = llm_with_tools.invoke(msgs)
            return {"messages": [response]}

        def should_continue(state: RAGState) -> str:
            """Route to 'tools' if the last message has tool calls, else END."""
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        tool_node = ToolNode(tools)

        workflow = StateGraph(RAGState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def query(
        self,
        query: str,
        agent: Agent,
        data_domains: list[DataDomain] | DataDomain,
    ) -> dict[str, Any]:
        """Execute a RAG query. data_domains can be a list or single domain."""
        try:
            domains = (
                list(data_domains)
                if isinstance(data_domains, (list, tuple))
                else [data_domains]
            )
            if not domains:
                raise ValueError("At least one data domain is required")
            chain = self.create_rag_chain(agent, domains)

            # Initial state
            initial_state: RAGState = {
                "messages": [],
                "query": query,
                "agent_config": {
                    "model": agent.llm_model,
                    "temperature": agent.temperature,
                    "system_prompt": agent.system_prompt,
                },
                "retrieved_docs": [],
                "response": "",
                "sources": [],
                "input_tokens": 0,
                "output_tokens": 0,
            }

            # Execute the chain
            result = chain.invoke(initial_state)

            logger.info(
                "RAG query completed",
                agent_id=agent.id,
                query_length=len(query),
                response_length=len(result["response"]),
                sources_count=len(result["sources"]),
            )

            return {
                "response": result["response"],
                "sources": result["sources"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
            }
        except Exception as e:
            logger.error("RAG query failed", agent_id=agent.id, error=str(e))
            raise


# Singleton instance
rag_service = RAGService()
