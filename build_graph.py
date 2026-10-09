"""
Phase 7: Build and query the concept knowledge graph.

Usage:
    python build_graph.py build output.json
        -- extracts concepts per section and builds the Neo4j graph

    python build_graph.py related "photosynthesis" output.json
        -- shows concepts related to "photosynthesis"

    python build_graph.py prereqs "calvin cycle" output.json
        -- shows concepts that (heuristically) precede "calvin cycle"

Prerequisites:
    - Neo4j must be running (see knowledge_graph.py docstring for setup)
    - NEO4J_PASSWORD must be set in your .env file
    - pip install neo4j keybert python-dotenv
"""

import sys
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from concept_extractor import extract_concepts_for_sections
from knowledge_graph import KnowledgeGraph


def build(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    title = data["title"]
    sections = data["sections"]
    print(f"Extracting concepts from {len(sections)} sections (this can take a minute)...")
    concepts_by_section = extract_concepts_for_sections(sections)

    total_concepts = sum(len(c) for c in concepts_by_section.values())
    print(f"Extracted {total_concepts} concept mentions across all sections")
    for i, sec in enumerate(sections[:5]):
        print(f"  [{sec['heading'][:35]:<35}] {concepts_by_section.get(i, [])}")
    if len(sections) > 5:
        print(f"  ... and {len(sections) - 5} more sections")

    print(f"\nConnecting to Neo4j and building graph for '{title}'...")
    kg = KnowledgeGraph()
    kg.clear_document(title)  # avoid duplicate nodes if re-running
    kg.build_graph(title, sections, concepts_by_section)
    kg.close()
    print("Done. Open http://localhost:7474 to explore the graph visually, e.g.:")
    print(f'  MATCH (d:Document {{title: "{title}"}})-[*1..3]-(n) RETURN d, n LIMIT 100')


def show_related(concept: str, json_path: str):
    title = json.load(open(json_path, encoding="utf-8"))["title"]
    kg = KnowledgeGraph()
    related = kg.get_related_concepts(title, concept)
    kg.close()
    if not related:
        print(f"No related concepts found for '{concept}'. Check spelling, or that the graph was built.")
        return
    print(f"Concepts related to '{concept}':")
    for r in related:
        print(f"  {r['name']} (co-occurrence weight: {r['weight']})")


def show_prereqs(concept: str, json_path: str):
    title = json.load(open(json_path, encoding="utf-8"))["title"]
    kg = KnowledgeGraph()
    prereqs = kg.get_prerequisites(title, concept)
    kg.close()
    if not prereqs:
        print(f"No prerequisite concepts found for '{concept}' (it may be one of the first concepts introduced).")
        return
    print(f"Concepts that (heuristically) precede '{concept}':")
    for p in prereqs:
        print(f"  {p}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python build_graph.py build <output.json>")
        print('  python build_graph.py related "<concept>" <output.json>')
        print('  python build_graph.py prereqs "<concept>" <output.json>')
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "build":
        build(sys.argv[2])
    elif mode == "related":
        show_related(sys.argv[2], sys.argv[3])
    elif mode == "prereqs":
        show_prereqs(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)