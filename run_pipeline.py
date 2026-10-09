"""
Phase 1: Run the full ingestion pipeline on a PDF.

Usage:
    python run_pipeline.py path/to/paper.pdf [output.json]

Produces a JSON file with structured sections AND retrieval-ready chunks,
which is exactly what Phase 2 (embeddings + vector store) will consume.
"""

import sys
import json
from pathlib import Path

from pdf_extractor import extract_structured_text
from chunker import chunk_document


def run(pdf_path: str, output_path: str = None):
    print(f"[1/3] Extracting structured text from: {pdf_path}")
    doc = extract_structured_text(pdf_path)
    print(f"      -> {doc['num_pages']} pages, {len(doc['sections'])} sections detected")

    print("[2/3] Chunking sections for retrieval...")
    chunks = chunk_document(doc)
    print(f"      -> {len(chunks)} chunks produced")

    output = {
        "title": doc["title"],
        "num_pages": doc["num_pages"],
        "sections": doc["sections"],
        "chunks": chunks,
    }

    if output_path is None:
        output_path = Path(pdf_path).stem + "_structured.json"

    print(f"[3/3] Saving to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nDone. Section breakdown:")
    for s in doc["sections"]:
        preview = s["text"][:60].replace("\n", " ")
        print(f"  [p{s['page_start']:>2}] {s['heading'][:40]:<40} | {len(s['text'].split())} words | {preview}...")

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <path_to_pdf> [output.json]")
        sys.exit(1)

    pdf_arg = sys.argv[1]
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    run(pdf_arg, out_arg)
