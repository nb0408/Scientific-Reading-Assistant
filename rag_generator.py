"""
Phase 3: RAG Generation Module (Gemini)
------------------------------------------
Takes retrieved chunks (from Phase 2's vector_store.query) and a user
question, builds a grounded prompt, and calls Gemini to generate an
explanation -- instead of just dumping raw text snippets at the user.

This is the actual "Retrieval-Augmented Generation" step in your project
title: retrieval (Phase 2) + generation (this file) = RAG.

Setup:
  1. Get a free API key from https://aistudio.google.com/apikey
  2. Store it permanently using ONE of these:
       a) A .env file in this same folder containing:
            GEMINI_API_KEY=your-key-here
          (requires: pip install python-dotenv -- this file auto-loads it)
       b) A permanent environment variable:
            Windows (PowerShell, run once):
              [System.Environment]::SetEnvironmentVariable("GEMINI_API_KEY", "your-key-here", "User")
            Mac/Linux (add to ~/.bashrc or ~/.zshrc):
              export GEMINI_API_KEY="your-key-here"
  3. pip install google-genai python-dotenv

IMPORTANT: if you use a .env file, add ".env" to a .gitignore file so the
key never gets committed if you push this project to GitHub.
"""

import os
from google import genai

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in the current folder, if present -- no-op if missing
except ImportError:
    pass  # dotenv not installed; that's fine if you're using a real env variable instead


class RAGGenerator:
    def __init__(self, model_name: str = "gemini-3.6-flash", api_key: str = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment "
                "variable, or pass api_key= directly."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        """
        Builds a grounded prompt: retrieved context + instructions.
        Each chunk is labeled with its section/page so the model can
        (and is told to) cite where information came from.
        """
        context_blocks = []
        for i, c in enumerate(chunks, 1):
            context_blocks.append(
                f"[Source {i} | Section: {c['section']} | Page: {c['page']}]\n{c['text']}"
            )
        context_text = "\n\n".join(context_blocks)

        prompt = f"""You are a scientific reading assistant. Answer the reader's question using ONLY the context provided below, which was retrieved from the document they are reading.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so clearly instead of guessing.
- Cite which Source number(s) you used for each claim, like this: (Source 1).
- Keep the explanation clear and appropriately simplified for a reader who may be struggling with this concept -- that is the whole point of this tool.

Context:
{context_text}

Reader's question: {question}

Answer:"""
        return prompt

    def generate_answer(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            return "I couldn't find any relevant content in the document to answer that."

        prompt = self._build_prompt(question, chunks)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text


if __name__ == "__main__":
    # quick smoke test with fake chunks (no vector store needed)
    fake_chunks = [
        {"section": "Photosynthesis", "page": 5, "text": "Photosynthesis occurs in the chloroplasts of plant cells, using sunlight, water, and carbon dioxide to produce glucose and oxygen."},
    ]
    gen = RAGGenerator()
    answer = gen.generate_answer("Where does photosynthesis happen?", fake_chunks)
    print(answer)