"""Tests for agent tools infrastructure and tool-calling agent changes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============== Tools registry tests ==============


def test_builtin_tools_registry():
    """BUILTIN_TOOLS contains the three expected tools."""
    from app.tools import BUILTIN_TOOLS

    assert "search_knowledge_base" in BUILTIN_TOOLS
    assert "translate_text" in BUILTIN_TOOLS
    assert "ingest_url" in BUILTIN_TOOLS
    assert len(BUILTIN_TOOLS) == 3


def test_get_tools_by_names_resolves_known():
    """get_tools_by_names returns tool objects for known names."""
    from app.tools import get_tools_by_names

    tools = get_tools_by_names(["search_knowledge_base", "translate_text"])
    assert len(tools) == 2
    assert tools[0].name == "search_knowledge_base"
    assert tools[1].name == "translate_text"


def test_get_tools_by_names_skips_unknown():
    """Unknown tool names are silently skipped."""
    from app.tools import get_tools_by_names

    tools = get_tools_by_names(["search_knowledge_base", "nonexistent_tool"])
    assert len(tools) == 1
    assert tools[0].name == "search_knowledge_base"


def test_get_tools_by_names_empty_list():
    """Empty input returns empty list."""
    from app.tools import get_tools_by_names

    assert get_tools_by_names([]) == []


# ============== search_knowledge_base tool tests ==============


def test_search_knowledge_base_requires_params():
    """Tool returns error when collection_name or embedding_model is empty."""
    from app.tools.search_knowledge_base import search_knowledge_base

    result = search_knowledge_base.invoke({"query": "hello"})
    assert "Error" in result
    assert "collection_name" in result


@patch("app.services.qdrant_service.qdrant_service")
@patch("app.services.embedding_service.embedding_service")
def test_search_knowledge_base_returns_context(mock_embed, mock_qdrant):
    """Tool formats Qdrant results into context string."""
    from app.tools.search_knowledge_base import search_knowledge_base

    mock_embed.embed_query.return_value = [0.1] * 1536
    mock_qdrant.search.return_value = [
        {
            "id": "1",
            "score": 0.95,
            "payload": {"content": "Test content", "source_file": "doc.pdf", "page": 1},
        }
    ]

    result = search_knowledge_base.invoke(
        {
            "query": "test query",
            "collection_name": "test_col",
            "embedding_model": "text-embedding-3-small",
            "k": 3,
        }
    )
    assert "Test content" in result
    assert "doc.pdf" in result
    mock_embed.embed_query.assert_called_once()
    mock_qdrant.search.assert_called_once()


@patch("app.services.qdrant_service.qdrant_service")
@patch("app.services.embedding_service.embedding_service")
def test_search_knowledge_base_no_results(mock_embed, mock_qdrant):
    """Tool returns 'No relevant documents' when Qdrant returns empty."""
    from app.tools.search_knowledge_base import search_knowledge_base

    mock_embed.embed_query.return_value = [0.1] * 1536
    mock_qdrant.search.return_value = []

    result = search_knowledge_base.invoke(
        {
            "query": "test query",
            "collection_name": "test_col",
            "embedding_model": "text-embedding-3-small",
        }
    )
    assert "No relevant documents found" in result


# ============== translate_text tool tests ==============


def test_translate_text_requires_target_language():
    """Tool returns error when target_language is empty."""
    from app.tools.translate_text import translate_text

    result = translate_text.invoke({"text": "Hello", "target_language": ""})
    assert "Error" in result


def test_translate_text_rejects_unsupported_language():
    """Tool returns error for unsupported language code."""
    from app.tools.translate_text import translate_text

    result = translate_text.invoke({"text": "Hello", "target_language": "xx_invalid"})
    assert "unsupported" in result.lower()


# ============== ingest_url tool tests ==============


def test_ingest_url_requires_params():
    """Tool returns error when url or collection_name is empty."""
    from app.tools.ingest_url import ingest_url

    result = ingest_url.invoke({"url": "", "collection_name": "test"})
    assert "Error" in result

    result = ingest_url.invoke({"url": "http://example.com", "collection_name": ""})
    assert "Error" in result


# ============== MCP client tests ==============


def test_mcp_client_wrapper_init():
    """MCPClientWrapper stores the server URL."""
    from app.tools.mcp_client import MCPClientWrapper

    wrapper = MCPClientWrapper(server_url="http://localhost:8080/sse")
    assert wrapper.server_url == "http://localhost:8080/sse"


# ============== Schema tests ==============


def test_rag_agent_create_has_tools_field():
    """RAGAgentCreate schema includes tools field with default []."""
    from app.schemas import RAGAgentCreate

    schema = RAGAgentCreate(
        name="Test",
        llm_model="gpt-4",
        system_prompt="You are helpful.",
        data_domain_ids=[1],
    )
    assert schema.tools == []


def test_rag_agent_create_with_tools():
    """RAGAgentCreate schema accepts tools list."""
    from app.schemas import RAGAgentCreate

    schema = RAGAgentCreate(
        name="Test",
        llm_model="gpt-4",
        system_prompt="You are helpful.",
        data_domain_ids=[1],
        tools=["search_knowledge_base", "translate_text"],
    )
    assert schema.tools == ["search_knowledge_base", "translate_text"]


def test_agent_update_has_tools_field():
    """AgentUpdate schema includes tools field (None by default)."""
    from app.schemas import AgentUpdate

    schema = AgentUpdate()
    assert schema.tools is None

    schema = AgentUpdate(tools=["ingest_url"])
    assert schema.tools == ["ingest_url"]


def test_agent_response_has_tools_field():
    """AgentResponse schema includes tools field."""
    from app.schemas import AgentResponse
    from datetime import datetime

    resp = AgentResponse(
        id=1,
        name="Test",
        description=None,
        agent_type="rag",
        llm_model="gpt-4",
        system_prompt="test",
        temperature=0.7,
        top_p=1.0,
        top_k=50,
        max_tokens=2048,
        retrieval_k=5,
        source_language=None,
        target_language=None,
        supported_languages=None,
        owner_id=1,
        is_public=False,
        test_questions=None,
        tools=["search_knowledge_base"],
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    assert resp.tools == ["search_knowledge_base"]


# ============== Model tests ==============


def test_agent_model_has_tools_column(db, test_user):
    """Agent ORM model has a tools column that defaults to empty list."""
    from app.models import Agent, AgentType

    agent = Agent(
        name="Test Tools Agent",
        agent_type=AgentType.RAG.value,
        llm_model="gpt-4",
        system_prompt="test",
        owner_id=test_user.id,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    # Tools should default to empty list (or None in SQLite, which is fine)
    assert agent.tools is None or agent.tools == []


def test_agent_model_stores_tools(db, test_user):
    """Agent ORM model can persist and retrieve a tools list."""
    from app.models import Agent, AgentType

    agent = Agent(
        name="Test Tools Agent 2",
        agent_type=AgentType.RAG.value,
        llm_model="gpt-4",
        system_prompt="test",
        tools=["search_knowledge_base", "translate_text"],
        owner_id=test_user.id,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    assert agent.tools == ["search_knowledge_base", "translate_text"]


# ============== RAGState tests ==============


def test_rag_state_has_messages_field():
    """RAGState TypedDict includes the messages field."""
    from app.services.rag_service import RAGState

    assert "messages" in RAGState.__annotations__


# ============== Config tests ==============


def test_config_has_mcp_server_url():
    """Settings includes MCP_SERVER_URL."""
    from app.config import settings

    # Default is None
    assert hasattr(settings, "MCP_SERVER_URL")


# ============== Agent API endpoint tests ==============


class FakeModel:
    def __init__(self, id):
        self.id = id


def _mock_model_service():
    mock = MagicMock()
    mock.get_llm_models = AsyncMock(return_value=[FakeModel("gpt-4")])
    return mock


@patch("app.services.model_service.model_service", new_callable=_mock_model_service)
def test_create_rag_agent_with_tools(mock_svc, client, test_data_domain):
    """Creating a RAG agent with tools should persist and return them."""
    response = client.post(
        "/api/agents/",
        json={
            "name": "Agent With Tools",
            "description": "Agent with tools configured",
            "llm_model": "gpt-4",
            "system_prompt": "You are helpful.",
            "data_domain_ids": [test_data_domain.id],
            "tools": ["search_knowledge_base", "translate_text"],
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["tools"] == ["search_knowledge_base", "translate_text"]


@patch("app.services.model_service.model_service", new_callable=_mock_model_service)
def test_create_rag_agent_without_tools(mock_svc, client, test_data_domain):
    """Creating a RAG agent without tools should default to empty list."""
    response = client.post(
        "/api/agents/",
        json={
            "name": "Agent No Tools",
            "llm_model": "gpt-4",
            "system_prompt": "You are helpful.",
            "data_domain_ids": [test_data_domain.id],
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["tools"] == []


@patch("app.services.model_service.model_service", new_callable=_mock_model_service)
def test_get_agent_returns_tools(mock_svc, client, test_data_domain):
    """GET agent detail includes tools field."""
    # Create agent first
    create_resp = client.post(
        "/api/agents/",
        json={
            "name": "Detail Tools Agent",
            "llm_model": "gpt-4",
            "system_prompt": "You are helpful.",
            "data_domain_ids": [test_data_domain.id],
            "tools": ["ingest_url"],
        },
    )
    assert create_resp.status_code == 201
    agent_id = create_resp.json()["id"]

    # GET the agent
    response = client.get(f"/api/agents/{agent_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["tools"] == ["ingest_url"]


@patch("app.services.model_service.model_service", new_callable=_mock_model_service)
def test_update_agent_tools(mock_svc, client, test_agent):
    """Updating an agent's tools should persist and return them."""
    response = client.put(
        f"/api/agents/{test_agent.id}",
        json={"tools": ["translate_text"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tools"] == ["translate_text"]
