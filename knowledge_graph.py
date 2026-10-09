"""
Phase 7b: Knowledge Graph (Neo4j)
--------------------------------------
Builds a graph representing the document's concept structure:

  (:Document)-[:HAS_SECTION]->(:Section)-[:MENTIONS]->(:Concept)
  (:Concept)-[:RELATED_TO {weight}]->(:Concept)   -- co-occurrence based
  (:Concept)-[:PRECEDES]->(:Concept)               -- order-of-first-appearance proxy

This lets the RAG assistant answer questions like "what should I understand
before this concept?" or "what's related to X?" using graph traversal
instead of just vector similarity -- useful for scaffolding explanations
(if a reader struggles with a concept, look up its prerequisites and
explain those first).

IMPORTANT -- prerequisite relationships here are a PROXY, not a validated
one: PRECEDES is inferred purely from the order concepts first appear in
the document, which correlates with (but isn't the same as) true
conceptual prerequisite structure. This is a reasonable, defensible
starting heuristic -- document authors usually do introduce foundational
concepts before ones that depend on them -- but say so explicitly in your
report rather than presenting it as a validated dependency graph.

Setup required before this will connect to anything:
  1. Install Neo4j Community Edition (Desktop app is easiest for Windows:
     https://neo4j.com/download/) OR run it via Docker:
       docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/yourpassword neo4j:5-community
  2. Open http://localhost:7474 in a browser to confirm it's running
  3. Set your credentials as environment variables (see .env.example)
"""

import os
from neo4j import GraphDatabase

try:
    from dotenv import load_dotenv
    load_dotenv()  # ensures .env loads even when this module is imported directly
except ImportError:
    pass


class KnowledgeGraph:
    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = None):
        uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.environ.get("NEO4J_USER", "neo4j")
        password = password or os.environ.get("NEO4J_PASSWORD")
        self.database = database or os.environ.get("NEO4J_DATABASE", "neo4j")
        if not password:
            raise ValueError(
                "No Neo4j password found. Set NEO4J_PASSWORD in your .env file "
                "(the password you chose when setting up Neo4j)."
            )
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def _session(self):
        return self.driver.session(database=self.database)

    def close(self):
        self.driver.close()

    def clear_document(self, doc_title: str):
        """Wipe any existing graph data for this document (useful when re-indexing)."""
        with self._session() as session:
            session.run(
                "MATCH (d:Document {title: $title})-[*0..2]-(n) DETACH DELETE d, n",
                title=doc_title,
            )

    def build_graph(self, doc_title: str, sections: list[dict], concepts_by_section: dict,
                     co_occurrence_threshold: int = 1):
        """
        sections: Phase 1's list of {"heading", "text", "page_start"}
        concepts_by_section: Phase 7a's output, {section_index: [concept, ...]}
        """
        with self._session() as session:
            session.run("MERGE (d:Document {title: $title})", title=doc_title)

            # Sections + their concepts
            for i, sec in enumerate(sections):
                session.run(
                    """
                    MATCH (d:Document {title: $title})
                    MERGE (s:Section {doc_title: $title, index: $idx})
                    SET s.heading = $heading, s.page = $page
                    MERGE (d)-[:HAS_SECTION]->(s)
                    """,
                    title=doc_title, idx=i, heading=sec["heading"], page=sec["page_start"],
                )
                for concept in concepts_by_section.get(i, []):
                    session.run(
                        """
                        MATCH (s:Section {doc_title: $title, index: $idx})
                        MERGE (c:Concept {doc_title: $title, name: $concept})
                        MERGE (s)-[:MENTIONS]->(c)
                        """,
                        title=doc_title, idx=i, concept=concept,
                    )

            # RELATED_TO: concepts that co-occur in the same section(s)
            all_concept_lists = list(concepts_by_section.values())
            co_occurrence = {}
            for concepts in all_concept_lists:
                for i, c1 in enumerate(concepts):
                    for c2 in concepts[i + 1:]:
                        key = tuple(sorted([c1, c2]))
                        co_occurrence[key] = co_occurrence.get(key, 0) + 1

            for (c1, c2), weight in co_occurrence.items():
                if weight >= co_occurrence_threshold:
                    session.run(
                        """
                        MATCH (a:Concept {doc_title: $title, name: $c1})
                        MATCH (b:Concept {doc_title: $title, name: $c2})
                        MERGE (a)-[r:RELATED_TO]-(b)
                        SET r.weight = $weight
                        """,
                        title=doc_title, c1=c1, c2=c2, weight=weight,
                    )

            # PRECEDES: order-of-first-appearance proxy for prerequisite structure
            first_seen_order = []
            seen = set()
            for concepts in all_concept_lists:
                for c in concepts:
                    if c not in seen:
                        seen.add(c)
                        first_seen_order.append(c)

            for earlier, later in zip(first_seen_order, first_seen_order[1:]):
                session.run(
                    """
                    MATCH (a:Concept {doc_title: $title, name: $earlier})
                    MATCH (b:Concept {doc_title: $title, name: $later})
                    MERGE (a)-[:PRECEDES]->(b)
                    """,
                    title=doc_title, earlier=earlier, later=later,
                )

    def get_related_concepts(self, doc_title: str, concept: str, limit: int = 5):
        with self._session() as session:
            result = session.run(
                """
                MATCH (c:Concept {doc_title: $title, name: $concept})-[r:RELATED_TO]-(other)
                RETURN other.name AS name, r.weight AS weight
                ORDER BY r.weight DESC
                LIMIT $limit
                """,
                title=doc_title, concept=concept, limit=limit,
            )
            return [{"name": r["name"], "weight": r["weight"]} for r in result]

    def get_prerequisites(self, doc_title: str, concept: str, limit: int = 5):
        """Concepts that (heuristically) should be understood before this one."""
        with self._session() as session:
            result = session.run(
                """
                MATCH (pre:Concept {doc_title: $title})-[:PRECEDES]->(c:Concept {doc_title: $title, name: $concept})
                RETURN pre.name AS name
                LIMIT $limit
                """,
                title=doc_title, concept=concept, limit=limit,
            )
            return [r["name"] for r in result]

    def get_sections_for_concept(self, doc_title: str, concept: str):
        with self._session() as session:
            result = session.run(
                """
                MATCH (s:Section {doc_title: $title})-[:MENTIONS]->(c:Concept {doc_title: $title, name: $concept})
                RETURN s.heading AS heading, s.page AS page
                ORDER BY s.index
                """,
                title=doc_title, concept=concept,
            )
            return [{"heading": r["heading"], "page": r["page"]} for r in result]