"""Knowledge graph service for entity extraction and co-occurrence graph reasoning."""

import re
import json
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
from difflib import SequenceMatcher
import networkx as nx
import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import settings

logger = structlog.get_logger()


class EntityExtractor:
    """Entity extractor with regex and optional LLM-based extraction."""

    def __init__(self, use_llm: bool = False, llm_model: str = "gpt-4o-mini"):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self._llm = None

        # Common entity patterns - can be extended
        self.patterns = {
            "person": [
                r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # First Last
                r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",  # Titles
            ],
            "organization": [
                r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Co)\b",
                r"\b(?:The\s+)?[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:University|Institute|College|Foundation)\b",
            ],
            "location": [
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+[A-Z]{2}\b",  # City, State
                r"\b(?:New|Old|North|South|East|West)\s+[A-Z][a-z]+\b",  # Compound locations
            ],
            "date": [
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # Dates
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            ],
            "number": [
                r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",  # Numbers with commas/decimals
                r"\$\d+(?:,\d{3})*(?:\.\d{2})?\b",  # Currency
            ],
            "acronym": [
                r"\b[A-Z]{2,}\b",  # All caps words (likely acronyms)
            ],
        }

    @property
    def llm(self):
        """Lazy-load LLM for entity extraction."""
        if self.use_llm and self._llm is None:
            self._llm = ChatOpenAI(
                model=self.llm_model,
                temperature=0.0,  # Low temperature for consistent extraction
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_API_BASE,
            )
        return self._llm

    def extract_regex(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Extract entities from text using regex patterns.

        Returns:
            List of (entity, entity_type, confidence) tuples
        """
        entities = []
        seen = set()

        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = match.group().strip()
                    # Normalize entity (lowercase for deduplication)
                    entity_key = (entity.lower(), entity_type)
                    if entity_key not in seen:
                        seen.add(entity_key)
                        entities.append(
                            (entity, entity_type, 0.7)
                        )  # Medium confidence for regex

        return entities

    def extract_llm(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Extract entities using LLM.

        Returns:
            List of (entity, entity_type, confidence) tuples
        """
        if not self.llm:
            return []

        try:
            prompt = f"""Extract all entities from the following text. Return a JSON array of objects with "entity", "type", and "confidence" fields.

Entity types should be: person, organization, location, date, number, acronym, or other relevant types.

Text:
{text}

Return only valid JSON array, no additional text."""

            messages = [
                SystemMessage(
                    content="You are an expert entity extraction system. Always return valid JSON."
                ),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Try to parse JSON (might be wrapped in markdown code blocks)
            if content.startswith("```"):
                # Extract JSON from code block
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            entities_data = json.loads(content)
            entities = [
                (item["entity"], item["type"], float(item.get("confidence", 0.8)))
                for item in entities_data
                if "entity" in item and "type" in item
            ]

            return entities

        except Exception as e:
            logger.warning("LLM entity extraction failed", error=str(e))
            return []

    def extract(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Extract entities from text using regex and optionally LLM.

        Returns:
            List of (entity, entity_type, confidence) tuples
        """
        # Always use regex
        regex_entities = self.extract_regex(text)

        if not self.use_llm:
            return regex_entities

        # Also use LLM if enabled
        llm_entities = self.extract_llm(text)

        # Merge entities, preferring LLM results for duplicates
        entity_map = {}

        # Add regex entities first
        for entity, entity_type, confidence in regex_entities:
            key = (entity.lower(), entity_type)
            entity_map[key] = (entity, entity_type, confidence)

        # Add/update with LLM entities (higher confidence)
        for entity, entity_type, confidence in llm_entities:
            key = (entity.lower(), entity_type)
            if key in entity_map:
                # Keep the one with higher confidence
                existing_conf = entity_map[key][2]
                if confidence > existing_conf:
                    entity_map[key] = (entity, entity_type, confidence)
            else:
                entity_map[key] = (entity, entity_type, confidence)

        return list(entity_map.values())


class EntityDeduplicator:
    """Entity deduplication and normalization."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.entity_canonical_map: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self.entity_clusters: Dict[Tuple[str, str], List[Tuple[str, str]]] = (
            defaultdict(list)
        )

    def similarity(self, entity1: str, entity2: str) -> float:
        """Calculate similarity between two entity strings."""
        return SequenceMatcher(None, entity1.lower(), entity2.lower()).ratio()

    def normalize_entity(self, entity: str, entity_type: str) -> Tuple[str, str]:
        """
        Normalize entity name and return canonical form.

        Returns:
            (canonical_entity, entity_type) tuple
        """
        normalized_key = (entity.lower().strip(), entity_type)

        # Check if we already have a canonical form
        if normalized_key in self.entity_canonical_map:
            return self.entity_canonical_map[normalized_key]

        # Check for similar entities
        best_match = None
        best_similarity = 0.0

        for existing_key in self.entity_canonical_map.keys():
            if existing_key[1] == entity_type:  # Same type
                sim = self.similarity(entity, existing_key[0])
                if sim > best_similarity and sim >= self.similarity_threshold:
                    best_similarity = sim
                    best_match = existing_key

        if best_match:
            # Use existing canonical form
            canonical = self.entity_canonical_map[best_match]
            self.entity_canonical_map[normalized_key] = canonical
            self.entity_clusters[canonical].append(normalized_key)
            return canonical

        # New entity, use as canonical
        canonical = (entity, entity_type)
        self.entity_canonical_map[normalized_key] = canonical
        self.entity_clusters[canonical].append(normalized_key)
        return canonical

    def get_cluster(self, entity: str, entity_type: str) -> List[Tuple[str, str]]:
        """Get all entities in the same cluster (duplicates)."""
        normalized_key = (entity.lower().strip(), entity_type)
        canonical = self.entity_canonical_map.get(normalized_key)
        if canonical:
            return self.entity_clusters.get(canonical, [])
        return []


class RelationshipExtractor:
    """Extract semantic relationships from text."""

    def __init__(self, use_llm: bool = False, llm_model: str = "gpt-4o-mini"):
        self.use_llm = use_llm
        self.llm_model = llm_model
        self._llm = None

    @property
    def llm(self):
        """Lazy-load LLM for relationship extraction."""
        if self.use_llm and self._llm is None:
            self._llm = ChatOpenAI(
                model=self.llm_model,
                temperature=0.0,
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_API_BASE,
            )
        return self._llm

    def extract_relationships(
        self, text: str, entities: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, str, float]]:
        """
        Extract relationships between entities.

        Args:
            text: Source text
            entities: List of (entity, entity_type) tuples found in text

        Returns:
            List of (subject, predicate, object, confidence) tuples
        """
        if not self.use_llm or not entities:
            return []

        try:
            # Build entity list for prompt
            entity_list = [
                f"- {e[0]} ({e[1]})" for e in entities[:20]
            ]  # Limit to 20 entities

            prompt = f"""Extract relationships between entities in the following text. Return a JSON array of objects with "subject", "predicate", "object", and "confidence" fields.

Entities found:
{chr(10).join(entity_list)}

Common relationship types: works_for, located_in, part_of, related_to, owns, created_by, etc.

Text:
{text[:2000]}  # Limit text length

Return only valid JSON array, no additional text."""

            messages = [
                SystemMessage(
                    content="You are an expert relationship extraction system. Always return valid JSON."
                ),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Parse JSON
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            relationships_data = json.loads(content)
            relationships = [
                (
                    item["subject"],
                    item["predicate"],
                    item["object"],
                    float(item.get("confidence", 0.7)),
                )
                for item in relationships_data
                if all(k in item for k in ["subject", "predicate", "object"])
            ]

            return relationships

        except Exception as e:
            logger.warning("LLM relationship extraction failed", error=str(e))
            return []


class CoOccurrenceGraph:
    """Co-occurrence graph for tracking entity relationships."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Directed graph for relationships
        self.entity_counts = defaultdict(int)
        self.co_occurrence_counts = defaultdict(int)
        self.relationship_counts = defaultdict(int)

    def add_entities(
        self,
        entities: List[Tuple[str, str]],
        chunk_id: Optional[str] = None,
        relationships: Optional[List[Tuple[str, str, str, float]]] = None,
    ):
        """
        Add entities and relationships from a chunk to the graph.

        Args:
            entities: List of (entity, entity_type) tuples
            chunk_id: Optional chunk identifier for tracking
            relationships: Optional list of (subject, predicate, object, confidence) tuples
        """
        # Normalize entities
        normalized_entities = [(e.lower(), et) for e, et in entities]

        # Update entity counts
        for entity, entity_type in normalized_entities:
            self.entity_counts[(entity, entity_type)] += 1
            self.graph.add_node((entity, entity_type), entity=entity, type=entity_type)

        # Add semantic relationships if provided
        if relationships:
            for subject, predicate, obj, confidence in relationships:
                subj_normalized = (
                    subject.lower(),
                    "unknown",
                )  # Type will be updated if entity exists
                obj_normalized = (obj.lower(), "unknown")

                # Find matching entities
                for e, et in normalized_entities:
                    if e == subject.lower():
                        subj_normalized = (e, et)
                    if e == obj.lower():
                        obj_normalized = (e, et)

                # Add relationship edge
                if (
                    subj_normalized[0] != obj_normalized[0]
                ):  # Don't connect entity to itself
                    self.graph.add_edge(
                        subj_normalized,
                        obj_normalized,
                        relationship=predicate,
                        weight=confidence,
                        type="semantic",
                    )
                    self.relationship_counts[
                        (subj_normalized, predicate, obj_normalized)
                    ] += 1

        # Add co-occurrence edges (undirected, for entities that appear together)
        for i, (e1, et1) in enumerate(normalized_entities):
            for e2, et2 in normalized_entities[i + 1 :]:
                if e1 != e2:  # Don't connect entity to itself
                    edge_key = tuple(sorted([(e1, et1), (e2, et2)]))
                    self.co_occurrence_counts[edge_key] += 1

                    # Add bidirectional co-occurrence edge
                    # Check if edge exists (check both directions for undirected co-occurrence)
                    has_edge_forward = self.graph.has_edge((e1, et1), (e2, et2))
                    has_edge_backward = self.graph.has_edge((e2, et2), (e1, et1))

                    if not has_edge_forward and not has_edge_backward:
                        self.graph.add_edge(
                            (e1, et1),
                            (e2, et2),
                            relationship="co_occurs_with",
                            weight=1,
                            type="co_occurrence",
                        )
                    else:
                        # Update weight - find existing edge
                        edge_found = False
                        if has_edge_forward:
                            # MultiDiGraph returns dict of edge keys
                            edge_dict = self.graph[(e1, et1)][(e2, et2)]
                            for edge_key, edge_data in edge_dict.items():
                                if edge_data.get("type") == "co_occurrence":
                                    edge_data["weight"] = edge_data.get("weight", 1) + 1
                                    edge_found = True
                                    break

                        if not edge_found and has_edge_backward:
                            edge_dict = self.graph[(e2, et2)][(e1, et1)]
                            for edge_key, edge_data in edge_dict.items():
                                if edge_data.get("type") == "co_occurrence":
                                    edge_data["weight"] = edge_data.get("weight", 1) + 1
                                    break

    def find_shortest_path(
        self,
        source_entities: List[Tuple[str, str]],
        target_entities: List[Tuple[str, str]],
    ) -> Optional[List[Tuple[str, str]]]:
        """
        Find shortest reasoning path between source and target entities.

        Args:
            source_entities: List of (entity, entity_type) tuples from query
            target_entities: List of (entity, entity_type) tuples from retrieved docs

        Returns:
            Shortest path as list of entity tuples, or None if no path exists
        """
        # Normalize entities
        source_normalized = [(e.lower(), et) for e, et in source_entities]
        target_normalized = [(e.lower(), et) for e, et in target_entities]

        # Find all possible paths
        shortest_path = None
        shortest_length = float("inf")

        for source in source_normalized:
            if source not in self.graph:
                continue

            for target in target_normalized:
                if target not in self.graph:
                    continue

                if source == target:
                    # Direct match
                    return [source]

                try:
                    path = nx.shortest_path(self.graph, source, target)
                    if len(path) < shortest_length:
                        shortest_length = len(path)
                        shortest_path = path
                except nx.NetworkXNoPath:
                    continue

        return shortest_path

    def get_related_entities(
        self, entity: Tuple[str, str], max_relations: int = 10
    ) -> List[Tuple[str, str]]:
        """Get entities related to the given entity."""
        normalized = (entity[0].lower(), entity[1])
        if normalized not in self.graph:
            return []

        neighbors = list(self.graph.neighbors(normalized))
        # Sort by edge weight (co-occurrence count)
        neighbors_with_weights = [
            (n, self.graph[normalized][n].get("weight", 1)) for n in neighbors
        ]
        neighbors_with_weights.sort(key=lambda x: x[1], reverse=True)

        return [n[0] for n in neighbors_with_weights[:max_relations]]

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        # Convert to undirected for some metrics
        undirected = self.graph.to_undirected()

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "unique_entities": len(self.entity_counts),
            "relationships": len(self.relationship_counts),
            "co_occurrences": len(self.co_occurrence_counts),
            "density": nx.density(undirected)
            if undirected.number_of_nodes() > 1
            else 0.0,
        }

    def compute_centrality(self, entity: Optional[Tuple[str, str]] = None) -> Dict:
        """
        Compute centrality metrics for the graph or a specific entity.

        Args:
            entity: Optional (entity, entity_type) tuple. If None, returns all entities.

        Returns:
            Dict with centrality metrics
        """
        # Convert to undirected for centrality
        undirected = self.graph.to_undirected()

        if undirected.number_of_nodes() == 0:
            return {}

        if entity:
            normalized = (entity[0].lower(), entity[1])
            if normalized not in undirected:
                return {}

            degree_centrality = nx.degree_centrality(undirected)
            betweenness_centrality = nx.betweenness_centrality(undirected)
            closeness_centrality = nx.closeness_centrality(undirected)

            return {
                "entity": entity,
                "degree_centrality": degree_centrality.get(normalized, 0.0),
                "betweenness_centrality": betweenness_centrality.get(normalized, 0.0),
                "closeness_centrality": closeness_centrality.get(normalized, 0.0),
            }
        else:
            # Return top entities by centrality
            degree_centrality = nx.degree_centrality(undirected)
            betweenness_centrality = nx.betweenness_centrality(undirected)

            top_degree = sorted(
                degree_centrality.items(), key=lambda x: x[1], reverse=True
            )[:10]
            top_betweenness = sorted(
                betweenness_centrality.items(), key=lambda x: x[1], reverse=True
            )[:10]

            return {
                "top_by_degree": [{"entity": e, "score": s} for e, s in top_degree],
                "top_by_betweenness": [
                    {"entity": e, "score": s} for e, s in top_betweenness
                ],
            }

    def find_communities(self) -> List[List[Tuple[str, str]]]:
        """
        Find communities (clusters) of related entities.

        Returns:
            List of communities, each containing entity tuples
        """
        undirected = self.graph.to_undirected()

        if undirected.number_of_nodes() == 0:
            return []

        try:
            # Use Louvain community detection
            communities = nx.community.louvain_communities(undirected, seed=42)
            return [list(community) for community in communities]
        except Exception as e:
            logger.warning("Community detection failed", error=str(e))
            # Fallback to connected components
            return [
                list(component) for component in nx.connected_components(undirected)
            ]


class KnowledgeGraphService:
    """Service for knowledge graph operations."""

    def __init__(
        self,
        use_llm_entities: bool = False,
        use_llm_relationships: bool = False,
        llm_model: str = "gpt-4o-mini",
        similarity_threshold: float = 0.85,
    ):
        self.entity_extractor = EntityExtractor(
            use_llm=use_llm_entities, llm_model=llm_model
        )
        self.relationship_extractor = RelationshipExtractor(
            use_llm=use_llm_relationships, llm_model=llm_model
        )
        self.deduplicator = EntityDeduplicator(
            similarity_threshold=similarity_threshold
        )
        self.co_occurrence_graph = CoOccurrenceGraph()

    def extract_entities(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Extract entities from text (with confidence scores).

        Returns:
            List of (entity, entity_type, confidence) tuples
        """
        return self.entity_extractor.extract(text)

    def add_chunk_to_graph(
        self,
        chunk_content: str,
        chunk_id: Optional[str] = None,
        extract_relationships: bool = True,
    ):
        """
        Add entities and relationships from a chunk to the graph.

        Args:
            chunk_content: Text content of the chunk
            chunk_id: Optional chunk identifier
            extract_relationships: Whether to extract semantic relationships

        Returns:
            Dict with entities and relationships
        """
        # Extract entities
        entities_with_conf = self.extract_entities(chunk_content)

        # Deduplicate and normalize entities
        normalized_entities = []
        for entity, entity_type, confidence in entities_with_conf:
            canonical = self.deduplicator.normalize_entity(entity, entity_type)
            normalized_entities.append(canonical)

        # Extract relationships if enabled
        relationships = []
        if extract_relationships and normalized_entities:
            # Convert to format expected by relationship extractor
            entity_list = [(e[0], e[1]) for e in normalized_entities]
            relationships = self.relationship_extractor.extract_relationships(
                chunk_content, entity_list
            )

        # Add to graph
        if normalized_entities:
            self.co_occurrence_graph.add_entities(
                normalized_entities, chunk_id, relationships
            )

        return {
            "entities": normalized_entities,
            "relationships": relationships,
            "entity_count": len(normalized_entities),
            "relationship_count": len(relationships),
        }

    def find_reasoning_path(
        self, query: str, retrieved_chunks: List[str]
    ) -> Optional[List[Tuple[str, str]]]:
        """
        Find shortest reasoning path between query entities and retrieved chunk entities.

        Args:
            query: User query text
            retrieved_chunks: List of retrieved chunk contents

        Returns:
            Shortest reasoning path or None
        """
        # Extract entities from query (with confidence)
        query_entities_with_conf = self.extract_entities(query)
        # Normalize query entities
        query_entities = [
            self.deduplicator.normalize_entity(e, et)
            for e, et, _ in query_entities_with_conf
        ]

        if not query_entities:
            return None

        # Extract entities from all retrieved chunks
        all_chunk_entities = []
        for chunk in retrieved_chunks:
            chunk_entities_with_conf = self.extract_entities(chunk)
            # Normalize chunk entities
            chunk_entities = [
                self.deduplicator.normalize_entity(e, et)
                for e, et, _ in chunk_entities_with_conf
            ]
            all_chunk_entities.extend(chunk_entities)

        if not all_chunk_entities:
            return None

        # Find shortest path
        path = self.co_occurrence_graph.find_shortest_path(
            query_entities, all_chunk_entities
        )

        if path:
            logger.info(
                "Found reasoning path",
                path_length=len(path),
                query_entities=len(query_entities),
                chunk_entities=len(set(all_chunk_entities)),
            )

        return path

    def get_entity_centrality(self, entity: str, entity_type: str) -> Dict:
        """Get centrality metrics for a specific entity."""
        return self.co_occurrence_graph.compute_centrality((entity, entity_type))

    def get_top_entities(self) -> Dict:
        """Get top entities by centrality."""
        return self.co_occurrence_graph.compute_centrality()

    def get_communities(self) -> List[List[Tuple[str, str]]]:
        """Get communities of related entities."""
        return self.co_occurrence_graph.find_communities()

    def get_entity_cluster(
        self, entity: str, entity_type: str
    ) -> List[Tuple[str, str]]:
        """Get all entities in the same cluster (duplicates/normalizations)."""
        return self.deduplicator.get_cluster(entity, entity_type)

    def get_graph_stats(self) -> Dict:
        """Get knowledge graph statistics."""
        return self.co_occurrence_graph.get_stats()

    def build_graph_from_collection(
        self, collection_name: str, qdrant_service, limit: Optional[int] = None
    ):
        """
        Build knowledge graph from chunks stored in Qdrant collection.

        Args:
            collection_name: Qdrant collection name
            qdrant_service: QdrantService instance
            limit: Optional limit on number of chunks to process
        """
        try:
            # Get collection info
            collection_info = qdrant_service.get_collection_info(collection_name)
            if not collection_info:
                logger.warning("Collection not found", collection=collection_name)
                return

            total_points = collection_info.get("points_count", 0)
            logger.info(
                "Building knowledge graph from collection",
                collection=collection_name,
                total_points=total_points,
            )

            # Note: Qdrant doesn't have a direct "get all points" API
            # In practice, you'd iterate through chunks or use scroll API
            # For now, this is a placeholder - the graph will be built incrementally
            # as chunks are retrieved during queries

            logger.info("Knowledge graph will be built incrementally during queries")

        except Exception as e:
            logger.error(
                "Failed to build graph from collection",
                error=str(e),
                collection=collection_name,
            )


# Singleton instance
# Can be configured via environment variables
import os

knowledge_graph_service = KnowledgeGraphService(
    use_llm_entities=os.getenv("KG_USE_LLM_ENTITIES", "false").lower() == "true",
    use_llm_relationships=os.getenv("KG_USE_LLM_RELATIONSHIPS", "false").lower()
    == "true",
    llm_model=os.getenv("KG_LLM_MODEL", "gpt-4o-mini"),
    similarity_threshold=float(os.getenv("KG_SIMILARITY_THRESHOLD", "0.85")),
)
