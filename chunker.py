"""
Phase 1b: Chunking Module
---------------------------
Splits extracted sections into embedding-ready chunks.

Why chunk at all: embedding models have a token limit, and retrieval quality
is much better on paragraph-sized pieces than on whole sections. This keeps
chunks anchored to their section heading, which is exactly what your RAG
pipeline (Phase 3) will need to cite "this came from Methodology" etc.
"""

import re


def chunk_text(text: str, max_words: int = 180, overlap_words: int = 30):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def chunk_document(structured_doc: dict, max_words: int = 180, overlap_words: int = 30):
    """
    Takes the output of pdf_extractor.extract_structured_text and produces
    a flat list of chunks, each tagged with its source section.

    Output: list of dicts:
        {"chunk_id": int, "section": str, "page": int, "text": str}
    """
    chunks = []
    chunk_id = 0
    for section in structured_doc["sections"]:
        text = section["text"]
        if not text.strip():
            continue
        for piece in chunk_text(text, max_words, overlap_words):
            chunks.append({
                "chunk_id": chunk_id,
                "section": section["heading"],
                "page": section["page_start"],
                "text": piece,
            })
            chunk_id += 1
    return chunks


if __name__ == "__main__":
    import sys
    import json
    from pdf_extractor import extract_structured_text

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <path_to_pdf>")
        sys.exit(1)

    doc = extract_structured_text(sys.argv[1])
    chunks = chunk_document(doc)
    print(f"Produced {len(chunks)} chunks from {len(doc['sections'])} sections.\n")
    for c in chunks[:3]:
        print(json.dumps(c, indent=2)[:500], "\n---")
