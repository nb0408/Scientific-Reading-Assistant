"""
Phase 3: Full RAG pipeline -- ask a question, get a grounded answer.

This is the "ask the assistant" experience your project is building
toward: it combines Phase 2's semantic retrieval with Gemini's generation
to produce an actual explanation, with citations back to the source
sections -- instead of raw text snippets.

Usage:
    python ask.py "how does the calvin cycle work"

Prerequisites:
    - You must have already run (Phase 1 + Phase 2):
        python run_pipeline.py paper.pdf output.json
        python run_embedding_pipeline.py index output.json
    - GEMINI_API_KEY must be set, either via a .env file in this folder
      or a real environment variable (see rag_generator.py for setup)
"""

import sys
import os

from embedder import Embedder
from vector_store import VectorStore
from rag_generator import RAGGenerator


def ask(question: str, top_k: int = 5, doc_title: str = None):
    print("Retrieving relevant sections...")
    embedder = Embedder()
    store = VectorStore()

    if store.count() == 0:
        print("Vector store is empty. Index a document first:")
        print("  python run_embedding_pipeline.py index output.json")
        return

    query_vec = embedder.embed_query(question)
    chunks = store.query(query_vec, top_k=top_k, doc_title=doc_title)

    print(f"Found {len(chunks)} relevant chunks. Generating answer...\n")
    generator = RAGGenerator()
    answer = generator.generate_answer(question, chunks)

    print("=" * 70)
    print(f"Q: {question}\n")
    print(answer)
    print("=" * 70)
    print("\nSources used:")
    for i, c in enumerate(chunks, 1):
        print(f"  [Source {i}] {c['section']} (page {c['page']}) -- similarity {1 - c['distance']:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question here"')
        sys.exit(1)

    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY is not set. See rag_generator.py for setup steps.\n")

    question = sys.argv[1]
    ask(question)