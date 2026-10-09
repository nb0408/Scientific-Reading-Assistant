"""
Phase 1: PDF Extraction Module
--------------------------------
Extracts structured text from scientific PDFs using PyMuPDF.

Why this approach (no GROBID):
GROBID gives the cleanest section/reference parsing, but it requires running
a Java server (heavy to set up for a first pass). Instead, this module infers
document structure directly from font sizes and boldness, which works well
for most scientific PDFs (including ones exported from Word, like this one)
and needs zero extra infrastructure. You can swap in GROBID later without
changing the rest of the pipeline (the output JSON shape stays the same).

Output shape:
{
  "title": str,
  "num_pages": int,
  "sections": [
      {"heading": str, "level": int, "text": str, "page_start": int}
  ],
  "full_text": str
}
"""

import pymupdf  # fitz
import re
import statistics
from pathlib import Path


def _extract_lines(doc):
    """
    Pull every LINE (not span) of text, with the line's dominant font size
    and whether the whole line is bold. Operating at line level (rather than
    span level) avoids false positives from justified-text spans that happen
    to get a slightly different size/weight mid-sentence -- a common artifact
    in PDFs exported from Word, like this one.
    """
    lines = []
    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                sizes = [s["size"] for s in spans]
                bold_flags = [bool(s["flags"] & 2**4) for s in spans]
                lines.append({
                    "text": text,
                    "size": round(max(sizes), 1) if sizes else 10.0,
                    "bold": all(bold_flags) if bold_flags else False,
                    "page": page_num,
                })
    return lines


def _body_font_size(lines):
    """Assume the most common font size is body text."""
    sizes = [l["size"] for l in lines]
    if not sizes:
        return 10.0
    return statistics.mode(sizes)


def _is_heading(line, body_size):
    """
    Heuristic: a heading is a SHORT, standalone line that is either
    noticeably larger than body text, or fully bold -- and doesn't end
    mid-sentence (no trailing comma, and either ends cleanly or has no
    terminal punctuation at all, since headings usually don't end in a period).
    """
    text = line["text"]
    word_count = len(text.split())
    if word_count == 0 or word_count > 12:
        return False
    if text.endswith(",") or text.endswith(";"):
        return False
    is_larger = line["size"] > body_size + 1.0
    is_bold_heading = line["bold"] and line["size"] >= body_size - 0.5
    return is_larger or is_bold_heading


def _merge_short_sections(sections, min_words=40):
    """
    Font-based heading detection over-fires on densely formatted PDFs
    (e.g. justified text with inline bold/emphasis, like citation lists).
    Rather than chasing a perfect heuristic, fold any section shorter than
    `min_words` into the previous one -- this keeps real headings (which
    are followed by substantial body text) while cleaning up noise.
    """
    if not sections:
        return sections
    merged = [sections[0]]
    for sec in sections[1:]:
        if len(merged[-1]["text"].split()) < min_words and merged:
            merged[-1]["text"] += " " + sec["heading"] + " " + sec["text"]
        else:
            merged.append(sec)
    return merged


def _clean_text(text):
    """Remove common PDF extraction artifacts."""
    text = re.sub(r'-\n', '', text)          # de-hyphenate line breaks
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'\x0c', '', text)          # form feed chars
    return text.strip()


def extract_structured_text(pdf_path: str) -> dict:
    """Main entry point: PDF path -> structured dict."""
    doc = pymupdf.open(pdf_path)
    lines = _extract_lines(doc)
    body_size = _body_font_size(lines)

    sections = []
    current = {"heading": "Untitled", "level": 0, "text": "", "page_start": 0}
    # Named-section pattern catches conventional scientific headings even
    # when font styling alone is ambiguous (e.g. "Abstract", "1. Introduction")
    named_section_pattern = re.compile(
        r'^\s*(\d+(\.\d+)*\.?\s+)?(abstract|introduction|problem statement|'
        r'novelty|components?|methodology|method|results?|discussion|'
        r'conclusion|references?|literature reviews?|summary|comparison'
        r'|acknowledg(e)?ments?)\b',
        re.IGNORECASE,
    )

    for line in lines:
        text = line["text"]
        is_named = bool(named_section_pattern.match(text)) and len(text.split()) <= 10
        looks_like_heading = is_named or _is_heading(line, body_size)

        if looks_like_heading:
            if current["text"].strip():
                current["text"] = _clean_text(current["text"])
                sections.append(current)
            current = {
                "heading": text,
                "level": 1 if line["size"] > body_size + 3 or is_named else 2,
                "text": "",
                "page_start": line["page"],
            }
        else:
            current["text"] += " " + text

    if current["text"].strip():
        current["text"] = _clean_text(current["text"])
        sections.append(current)

    sections = _merge_short_sections(sections)

    full_text = _clean_text(" ".join(s["heading"] + " " + s["text"] for s in sections))
    title = Path(pdf_path).stem
    if sections and len(sections[0]["heading"]) > 15:
        title = sections[0]["heading"]

    result = {
        "title": title,
        "num_pages": doc.page_count,
        "sections": sections,
        "full_text": full_text,
    }
    doc.close()
    return result


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        sys.exit(1)

    data = extract_structured_text(sys.argv[1])
    print(json.dumps(data, indent=2)[:3000])
    print(f"\n... extracted {len(data['sections'])} sections, "
          f"{len(data['full_text'])} chars total.")
