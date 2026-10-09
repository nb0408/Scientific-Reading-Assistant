"""
Phase 2: Embed chunks and store them for semantic search.

Two modes:

1. Index a document (run once per paper, after Phase 1's run_pipeline.py):
     python run_embedding_pipeline.py index output.json

2. Query the store (run any time, across all indexed documents):
     python run_embedding_pipeline.py query "how does the calvin cycle work"

Everything is persisted to ./chroma_db, so indexed documents survive
between runs -- you build up a searchable library over time.
"""

import sys
import json

from embedder import Embedder
from vector_store import VectorStore


def index_document(json_path: str):
    print(f"Loading chunks from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    title = data["title"]
    chunks = data["chunks"]
    print(f"Document: '{title}' -- {len(chunks)} chunks to embed")

    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    print(f"Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")

    store = VectorStore()
    store.add_chunks(doc_title=title, chunks=chunks, embeddings=embeddings)
    print(f"Stored in ChromaDB. Collection now has {store.count()} total chunks.")


def query_store(query_text: str, top_k: int = 5, doc_title: str = None):
    embedder = Embedder()
    store = VectorStore()

    if store.count() == 0:
        print("Vector store is empty. Index a document first:")
        print("  python run_embedding_pipeline.py index output.json")
        return

    query_vec = embedder.embed_query(query_text)
    hits = store.query(query_vec, top_k=top_k, doc_title=doc_title)

    print(f"\nTop {len(hits)} results for: \"{query_text}\"\n" + "=" * 60)
    for i, hit in enumerate(hits, 1):
        similarity = 1 - hit["distance"]  # cosine distance -> similarity
        print(f"\n[{i}] similarity={similarity:.3f} | section: {hit['section']} | page {hit['page']} | doc: {hit['doc_title']}")
        print(f"    {hit['text'][:200]}...")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print('  python run_embedding_pipeline.py index <chunks.json>')
        print('  python run_embedding_pipeline.py query "<your question>"')
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "index":
        index_document(sys.argv[2])
    elif mode == "query":
        query_store(sys.argv[2])
    else:
        print(f"Unknown mode: {mode}. Use 'index' or 'query'.")
        sys.exit(1)
