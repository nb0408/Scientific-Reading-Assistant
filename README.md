# Phase 1 — PDF Ingestion Pipeline

Turns a scientific PDF into structured, retrieval-ready JSON: sections with
headings + page numbers, plus paragraph-sized chunks tagged by source section.

## Setup
```bash
pip install pymupdf
```

## Run
```bash
python run_pipeline.py path/to/paper.pdf output.json
```

## Output shape
```json
{
  "title": "...",
  "num_pages": 28,
  "sections": [
    {"heading": "Abstract", "level": 1, "text": "...", "page_start": 2}
  ],
  "chunks": [
    {"chunk_id": 0, "section": "Abstract", "page": 2, "text": "..."}
  ]
}
```

## How it works
- `pdf_extractor.py` — reads text line-by-line via PyMuPDF, uses font size +
  boldness + known section-name patterns ("Abstract", "Introduction", etc.)
  to detect headings, then merges any section under 40 words into its
  neighbor (cleans up noise from densely formatted PDFs).
- `chunker.py` — splits each section into ~180-word overlapping chunks,
  sized for embedding models. Overlap (30 words) keeps context from being
  cut awkwardly at chunk boundaries.
- `run_pipeline.py` — CLI that ties both together and saves JSON.

## Known limitation
Font-based heading detection is a heuristic, not a real parser — it works
well on most single-column scientific papers but can still over/under-segment
on unusually formatted documents (e.g. heavy inline bold text, multi-column
layouts). If you need production-grade section/reference parsing later,
swap this module for GROBID (same output shape, just point Phase 2 at its
XML output instead) — everything downstream (chunker, Phase 2 embeddings)
stays unchanged.

## Tested on
This repo's own sample HCI report PDF: 28 pages → 41 sections → 59 chunks,
5,052 words captured with zero data loss.

## Next: Phase 2
Feed `chunks` into a Sentence-Transformers / SciBERT embedding step and
store vectors in ChromaDB for semantic search.
