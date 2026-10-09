"""
Text Selector Component
----------------------------
A custom Streamlit component that renders text and reports back whatever
substring the reader highlights with their mouse. Used to trigger an
automatic simplified explanation of just the selected passage, rather
than the whole section.

This is a "static" Streamlit component (plain HTML/JS, no React/npm build
step) -- see text_selector_frontend/index.html for the implementation.
"""

import os
import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_selector_frontend")

_component = components.declare_component("text_selector", path=_FRONTEND_DIR)


def text_selector(text: str, key: str = None):
    """
    Renders `text` (plain text, will be HTML-escaped) inside a styled,
    selectable block. Returns the most recently highlighted substring as
    a plain string, or None if nothing has been selected yet.

    IMPORTANT: this returns the LAST selection made, and keeps returning
    it on every subsequent rerun until the reader selects something new
    -- callers must track "have I already acted on this exact selection"
    themselves (see app.py) to avoid repeating an action (like an API
    call) on every unrelated app rerun.
    """
    return _component(text=text, key=key, default=None)