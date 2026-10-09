"""
Phase 7a: Concept Extraction (KeyBERT)
-------------------------------------------
Extracts key concepts/keyphrases per section, which become the nodes of
the knowledge graph. This is the piece your report specifies KeyBERT for.

Note: KeyBERT internally uses a Sentence-Transformers model (defaults to
all-MiniLM-L6-v2, the same one Phase 2 already uses) to score candidate
phrases by how well they represent the document -- so no separate model
download beyond what you already have from Phase 2.
"""

from keybert import KeyBERT

_kw_model = None


def _get_model():
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    return _kw_model


def extract_concepts(text: str, top_n: int = 5, ngram_range=(1, 2)) -> list[str]:
    """
    Returns up to top_n key concepts/phrases from the given text,
    e.g. "calvin cycle", "chlorophyll", "light reaction".
    """
    if not text or len(text.split()) < 5:
        return []

    model = _get_model()
    keywords = model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words="english",
        top_n=top_n,
        use_mmr=True,       # Maximal Marginal Relevance: reduces redundant/near-duplicate phrases
        diversity=0.5,
    )
    return [kw for kw, score in keywords]


def extract_concepts_for_sections(sections: list[dict], top_n: int = 5) -> dict:
    """
    sections: Phase 1's list of {"heading": ..., "text": ..., "page_start": ...}
    Returns: {section_index: [concept1, concept2, ...]}
    """
    result = {}
    for i, sec in enumerate(sections):
        result[i] = extract_concepts(sec["text"], top_n=top_n)
    return result


if __name__ == "__main__":
    sample = (
        "The Calvin cycle uses ATP and NADPH produced during the light "
        "reactions to fix carbon dioxide into sugar. RuBisCO catalyzes the "
        "first step, attaching CO2 to RuBP to form 3-PGA."
    )
    print(extract_concepts(sample))