"""
Phase 2a: Embedding Module
----------------------------
Converts text chunks (from Phase 1) into dense semantic vectors.

Model choice: all-MiniLM-L6-v2 (Sentence-Transformers)
  - Small (~80MB), fast, runs on CPU with no GPU needed
  - Good general-purpose semantic similarity out of the box
  - This is the right starting point for a working pipeline

Upgrade path -> SciBERT / SPECTER2:
  Your report specifies SciBERT for scientific-domain embeddings, which is
  a legitimate upgrade once the pipeline works end-to-end. SciBERT is a
  masked-language-model checkpoint (not trained with a sentence-similarity
  objective), so using it directly for chunk-level cosine similarity gives
  noticeably worse retrieval than a model like MiniLM or SPECTER2, which
  ARE trained for this. The practical path: get the RAG pipeline working
  with MiniLM first, then swap the encoder here for SPECTER2
  ("allenai/specter2") which is trained specifically for scientific paper
  embeddings and is a drop-in replacement in this same class.
"""

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name} (first run downloads it)...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_texts(self, texts: list[str]):
        """Embed a list of strings -> list of vectors (as plain lists of floats)."""
        vectors = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 20,
            normalize_embeddings=True,  # so cosine similarity == dot product
        )
        return vectors.tolist()

    def embed_query(self, query: str):
        """Embed a single query string -> one vector."""
        return self.embed_texts([query])[0]


if __name__ == "__main__":
    # quick smoke test
    embedder = Embedder()
    vecs = embedder.embed_texts(["photosynthesis occurs in chloroplasts", "the mitochondria is the powerhouse of the cell"])
    print(f"Embedded {len(vecs)} texts, each of dimension {len(vecs[0])}")
