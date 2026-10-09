"""
Phase 4: Behaviour-Aware Reading Assistant -- Streamlit App
--------------------------------------------------------------
Ties Phases 1-3 into a real interface, and adds the piece that makes this
project "behaviour-aware" rather than just a RAG chatbot: it tracks how
long the reader spends on each section and how often they come back to
it, and proactively offers a simplified explanation when those signals
suggest struggle -- BEFORE the reader has to ask for help.

Behavioural signals captured here (deliberately scoped down from your
report's full list -- see README_phase4.md for what's deferred and why):
  - Reading time per section (real: measured via session timestamps)
  - Revisit count per section (real: counted every time a section is reopened)
  - Scroll/cursor/gaze/highlighting -- NOT implemented here; these need
    either a custom JS component or webcam pipeline (MediaPipe/DeepFace),
    which is a separate, heavier subsystem. See README_phase4.md.

Knowledge-gap detection here is a RULE-BASED heuristic (dwell time +
revisits crossing a threshold), not the PyTorch deep-learning model your
report describes. This is intentional: ship the behaviour-aware loop
end-to-end first, then replace this function with a trained model without
touching anything else (retrieval, generation, or the UI).

Run with:
    streamlit run app.py
"""

import time
import tempfile
import os
import streamlit as st

from pdf_extractor import extract_structured_text
from chunker import chunk_document
from embedder import Embedder
from vector_store import VectorStore
from rag_generator import RAGGenerator
from text_selector import text_selector
import gap_predictor
import gaze_tracker
import numpy as np
import cv2
import time as _time_module
import threading
from streamlit_webrtc import webrtc_streamer
from streamlit_autorefresh import st_autorefresh

ATTENTION_CAPTURE_INTERVAL_SEC = 5


class GazeVideoProcessor:
    """
    Runs on streamlit-webrtc's background video thread. Every incoming
    frame is checked against a timer -- only every ATTENTION_CAPTURE_INTERVAL_SEC
    seconds is a frame actually run through gaze_tracker (running MediaPipe
    on every single frame, ~30/sec, would be wasteful and slow).

    Thread safety note: recv() runs on a different thread than the main
    Streamlit script. Results are handed off via a lock-protected variable
    rather than touching st.session_state directly from this thread (which
    Streamlit doesn't support safely).
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_result = None
        self._last_capture_time = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        now = _time_module.time()
        if now - self._last_capture_time >= ATTENTION_CAPTURE_INTERVAL_SEC:
            self._last_capture_time = now
            try:
                result = gaze_tracker.analyze_frame(img)
                with self._lock:
                    self._latest_result = result
            except FileNotFoundError:
                pass  # model file missing -- silently skip, UI surfaces this elsewhere
        return frame

    def pop_latest_result(self):
        """Returns the newest result and clears it, or None if nothing new since last check."""
        with self._lock:
            result, self._latest_result = self._latest_result, None
            return result


# ---------------------------------------------------------------------------
# Cached resources (loaded once per app session, not on every rerun)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_embedder():
    return Embedder()


@st.cache_resource
def get_vector_store():
    return VectorStore(persist_dir="./chroma_db_app")


@st.cache_resource
def get_rag_generator():
    return RAGGenerator()


# ---------------------------------------------------------------------------
# Knowledge-gap heuristic (rule-based stand-in for the Phase 6 ML model)
# ---------------------------------------------------------------------------

DWELL_TIME_THRESHOLD_SEC = 45   # "spending a while" on one section
REVISIT_THRESHOLD = 2           # "coming back to it repeatedly"


def compute_gap_score(dwell_seconds: float, revisit_count: int, section_text: str = "",
                       avg_attention_score: float = None) -> float:
    """
    Returns a 0-1 struggle score. Uses the trained ML model
    (gap_predictor.py) if gap_model.joblib exists; otherwise falls back
    to this simple weighted heuristic. This fallback design is what lets
    the app run before any model has been trained, and upgrade silently
    once one is, with zero other code changes.

    avg_attention_score: optional webcam-derived signal (see gaze_tracker.py).
    Pass None if no webcam data exists for this section yet -- both the
    ML model and the heuristic below handle that case explicitly rather
    than silently misusing a missing value.
    """
    ml_score = gap_predictor.predict_gap_score(dwell_seconds, revisit_count, section_text, avg_attention_score)
    if ml_score is not None:
        return ml_score

    dwell_component = min(dwell_seconds / (DWELL_TIME_THRESHOLD_SEC * 2), 1.0)
    revisit_component = min(revisit_count / (REVISIT_THRESHOLD * 2), 1.0)
    score = 0.6 * dwell_component + 0.4 * revisit_component
    if avg_attention_score is not None:
        # Blend in attention: low attention nudges the heuristic score up.
        # Weighted lightly (15%) since this heuristic wasn't tuned against
        # real data -- the trained model path above is the "real" one.
        attention_penalty = 1.0 - avg_attention_score
        score = 0.85 * score + 0.15 * attention_penalty
    return round(min(score, 1.0), 2)


def is_struggling(dwell_seconds: float, revisit_count: int, section_text: str = "",
                   avg_attention_score: float = None) -> bool:
    score = compute_gap_score(dwell_seconds, revisit_count, section_text, avg_attention_score)
    return score >= 0.5


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "doc_data": None,           # output of extract_structured_text + chunks
        "doc_title": None,
        "indexed": False,
        "current_section_idx": 0,
        "section_enter_time": time.time(),
        "section_stats": {},        # {section_idx: {"dwell": float, "revisits": int, "visited": bool}}
        "chat_history": [],         # list of {"role": ..., "content": ...}
        "dismissed_nudges": set(),  # section indices where user dismissed the nudge already
        "last_selection_processed": {},   # {section_idx: last selected string already simplified}
        "selection_explanations": {},     # {section_idx: {"selection": str, "explanation": str}}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def record_time_on_current_section():
    """Call this BEFORE switching sections to bank the dwell time."""
    idx = st.session_state.current_section_idx
    elapsed = time.time() - st.session_state.section_enter_time
    stats = st.session_state.section_stats.setdefault(idx, {"dwell": 0.0, "revisits": 0, "visited": False, "attention_scores": []})
    stats["dwell"] += elapsed


def switch_to_section(new_idx: int):
    record_time_on_current_section()
    stats = st.session_state.section_stats.setdefault(new_idx, {"dwell": 0.0, "revisits": 0, "visited": False, "attention_scores": []})
    if stats["visited"]:
        stats["revisits"] += 1
    stats["visited"] = True
    st.session_state.current_section_idx = new_idx
    st.session_state.section_enter_time = time.time()


def avg_attention_for(stats: dict):
    """Returns the average attention score for a section, or None if no webcam data exists yet."""
    scores = stats.get("attention_scores", [])
    return (sum(scores) / len(scores)) if scores else None


# ---------------------------------------------------------------------------
# Pipeline glue: ingest + index a newly uploaded PDF
# ---------------------------------------------------------------------------

def process_uploaded_pdf(uploaded_file):
    # Use the system temp directory (cross-platform: works on Windows, Mac, Linux)
    # instead of a hardcoded /tmp path, which only exists on Linux/Mac.
    tmp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Extracting text and structure..."):
        doc = extract_structured_text(tmp_path)
        chunks = chunk_document(doc)

    with st.spinner("Generating embeddings and indexing (first run downloads the model)..."):
        embedder = get_embedder()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_texts(texts)
        store = get_vector_store()
        store.add_chunks(doc_title=doc["title"], chunks=chunks, embeddings=embeddings)

    st.session_state.doc_data = doc
    st.session_state.doc_title = doc["title"]
    st.session_state.indexed = True
    st.session_state.current_section_idx = 0
    st.session_state.section_enter_time = time.time()
    st.session_state.section_stats = {0: {"dwell": 0.0, "revisits": 0, "visited": True, "attention_scores": []}}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Behaviour-Aware Reading Assistant", layout="wide", page_icon="📖")
init_state()

st.title("📖 Behaviour-Aware Scientific Reading Assistant")
st.caption("Upload a paper, read section by section, and get proactive help when you seem to be struggling.")

with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None and (
        st.session_state.doc_title is None or uploaded_file.name != st.session_state.get("_last_upload_name")
    ):
        process_uploaded_pdf(uploaded_file)
        st.session_state["_last_upload_name"] = uploaded_file.name
        st.rerun()

    if st.session_state.doc_data:
        st.divider()
        st.header("Sections")
        sections = st.session_state.doc_data["sections"]
        for i, sec in enumerate(sections):
            label = sec["heading"][:40] or f"Section {i+1}"
            is_current = i == st.session_state.current_section_idx
            if st.button(("👉 " if is_current else "") + label, key=f"nav_{i}", use_container_width=True):
                switch_to_section(i)
                st.rerun()

# ---------------------------------------------------------------------------
# Main area: reading pane + chat pane
# ---------------------------------------------------------------------------

if not st.session_state.doc_data:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    sections = st.session_state.doc_data["sections"]
    idx = st.session_state.current_section_idx
    current_section = sections[idx]
    stats = st.session_state.section_stats.get(idx, {"dwell": 0.0, "revisits": 0})
    live_dwell = stats["dwell"] + (time.time() - st.session_state.section_enter_time)

    read_col, chat_col = st.columns([3, 2])

    with read_col:
        st.subheader(current_section["heading"] or f"Section {idx+1}")
        st.caption(f"Page {current_section['page_start']} | Section {idx+1} of {len(sections)}")

        # --- Select-to-simplify: highlighting any part of the passage below
        # automatically triggers a simplified explanation of just that part,
        # with no button click needed. ---
        selected_text = text_selector(current_section["text"], key=f"text_selector_{idx}")
        already_processed = st.session_state.last_selection_processed.get(idx)
        if selected_text and selected_text.strip() and selected_text != already_processed:
            st.session_state.last_selection_processed[idx] = selected_text
            generator = get_rag_generator()
            fake_chunk = [{
                "section": current_section["heading"],
                "page": current_section["page_start"],
                "text": selected_text,
            }]
            with st.spinner("Simplifying selected passage..."):
                simple_explanation = generator.generate_answer(
                    "Explain the following selected passage in simple, plain terms, "
                    "as if to someone finding it difficult. Focus only on this "
                    "passage, not the rest of the document.",
                    fake_chunk,
                )
            st.session_state.selection_explanations[idx] = {
                "selection": selected_text,
                "explanation": simple_explanation,
            }
            preview = selected_text[:80] + ("..." if len(selected_text) > 80 else "")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"**Simplified selection: \"{preview}\"**\n\n{simple_explanation}",
            })
            st.rerun()

        current_explanation = st.session_state.selection_explanations.get(idx)
        if current_explanation:
            with st.container(border=True):
                st.markdown(f"🔎 **Simplified:** _{current_explanation['selection'][:120]}{'...' if len(current_explanation['selection']) > 120 else ''}_")
                st.write(current_explanation["explanation"])

        # --- Behaviour-aware nudge ---
        section_attention = avg_attention_for(stats)
        struggling = is_struggling(live_dwell, stats["revisits"], current_section["text"], section_attention)
        already_dismissed = idx in st.session_state.dismissed_nudges
        if struggling and not already_dismissed:
            gap_score = compute_gap_score(live_dwell, stats["revisits"], current_section["text"], section_attention)
            method_tag = "ML model" if gap_predictor.model_available() else "heuristic"
            attention_note = f", avg attention {section_attention:.2f}" if section_attention is not None else ""
            st.warning(
                f"You've spent {int(live_dwell)}s here"
                + (f" and revisited it {stats['revisits']}x" if stats["revisits"] else "")
                + attention_note
                + f" (struggle score: {gap_score}, via {method_tag}). Want a simplified explanation of this section?"
            )
            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("Yes, simplify this section", type="primary"):
                    generator = get_rag_generator()
                    fake_chunk = [{
                        "section": current_section["heading"],
                        "page": current_section["page_start"],
                        "text": current_section["text"],
                    }]
                    with st.spinner("Generating a simpler explanation..."):
                        explanation = generator.generate_answer(
                            "Explain this section in simple terms, as if to someone finding it difficult.",
                            fake_chunk,
                        )
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"**Simplified explanation of '{current_section['heading']}':**\n\n{explanation}",
                    })
                    st.rerun()
            with btn_col2:
                if st.button("No thanks"):
                    st.session_state.dismissed_nudges.add(idx)
                    st.rerun()

        st.divider()

        # --- Automatic webcam attention tracking (Phase 7b) ---
        with st.expander("📷 Automatic attention tracking (optional, uses your webcam)"):
            st.caption(
                "When enabled, this analyzes a snapshot from your live camera every "
                f"{ATTENTION_CAPTURE_INTERVAL_SEC} seconds -- processed in memory only, "
                "never saved to disk or sent anywhere. See README_phase7_gaze.md for "
                "exactly what is and isn't tracked."
            )
            consent = st.checkbox(
                "I consent to periodic webcam snapshots being analyzed while this is open",
                key=f"consent_{idx}",
            )
            if consent:
                webrtc_ctx = webrtc_streamer(
                    key=f"gaze_stream_{idx}",
                    video_processor_factory=GazeVideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                )
                if webrtc_ctx.state.playing:
                    st_autorefresh(interval=2000, key=f"attn_refresh_{idx}")
                    if webrtc_ctx.video_processor:
                        latest = webrtc_ctx.video_processor.pop_latest_result()
                        if latest is not None:
                            stats["attention_scores"].append(latest["attention_score"])
                            st.session_state[f"_last_attn_{idx}"] = latest
                        shown = st.session_state.get(f"_last_attn_{idx}")
                        if shown:
                            st.write(
                                f"Latest attention score: **{shown['attention_score']}** "
                                f"(looking away: {shown['looking_away']}, blinking: {shown['is_blinking']}, "
                                f"face detected: {shown['face_detected']})"
                            )

        st.divider()
        nav_col1, nav_col2 = st.columns([1, 1])
        with nav_col1:
            if idx > 0 and st.button("⬅ Previous section"):
                switch_to_section(idx - 1)
                st.rerun()
        with nav_col2:
            if idx < len(sections) - 1 and st.button("Next section ➡"):
                switch_to_section(idx + 1)
                st.rerun()

    with chat_col:
        st.subheader("Ask the assistant")
        st.caption("Answers are grounded in the document you uploaded, with citations.")

        chat_box = st.container(height=400)
        with chat_box:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        question = st.chat_input("Ask a question about this document...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            embedder = get_embedder()
            store = get_vector_store()
            query_vec = embedder.embed_query(question)
            chunks = store.query(query_vec, top_k=5, doc_title=st.session_state.doc_title)
            generator = get_rag_generator()
            with st.spinner("Thinking..."):
                answer = generator.generate_answer(question, chunks)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

    # --- Debug/insight panel: shows the raw behavioural data being captured ---
    with st.expander("📊 Reading behaviour log (for evaluation / demo purposes)"):
        st.caption(f"Gap scoring method: **{'Trained ML model (gap_model.joblib)' if gap_predictor.model_available() else 'Rule-based heuristic (no trained model found)'}**")
        for i, sec in enumerate(sections):
            s = st.session_state.section_stats.get(i)
            if s:
                dwell = s["dwell"] + (time.time() - st.session_state.section_enter_time if i == idx else 0)
                sec_attention = avg_attention_for(s)
                gap = compute_gap_score(dwell, s["revisits"], sec["text"], sec_attention)
                attn_scores = s.get("attention_scores", [])
                attn_str = f" | avg attention: {sec_attention:.2f} (n={len(attn_scores)})" if sec_attention is not None else ""
                st.write(f"**{sec['heading'][:50]}** -- dwell: {int(dwell)}s | revisits: {s['revisits']} | gap score: {gap}{attn_str}")