"""
Phase 2b: Vector Store Module
--------------------------------
Wraps ChromaDB to store chunk embeddings and perform semantic search.

Why ChromaDB (over Pinecone, as the report suggests):
Pinecone is a hosted service that needs an API key and network calls for
every operation. ChromaDB runs embedded, in-process, writing to a local
folder -- zero accounts, zero cost, zero network dependency. It's the
right choice for development and for a college project demo. If you later
need a hosted/production vector DB, swapping to Pinecone only requires
changing this one file; nothing else in the pipeline (embedder, chunker,
RAG layer) needs to change.
"""

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, doc_title: str, chunks: list[dict], embeddings: list[list[float]]):
        """
        chunks: list of {"chunk_id": int, "section": str, "page": int, "text": str}
        embeddings: parallel list of vectors from Embedder.embed_texts()
        """
        ids = [f"{doc_title}::{c['chunk_id']}" for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {"doc_title": doc_title, "section": c["section"], "page": c["page"]}
            for c in chunks
        ]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int = 5, doc_title: str = None):
        """
        Semantic search. Returns list of dicts:
            {"text": str, "section": str, "page": int, "doc_title": str, "distance": float}
        Pass doc_title to restrict search to one document (useful once you
        store multiple papers in the same collection).
        """
        where_filter = {"doc_title": doc_title} if doc_title else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "section": results["metadatas"][0][i]["section"],
                "page": results["metadatas"][0][i]["page"],
                "doc_title": results["metadatas"][0][i]["doc_title"],
                "distance": results["distances"][0][i],
            })
        return hits

    def count(self):
        return self.collection.count()
