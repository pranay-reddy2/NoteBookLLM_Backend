"""
rag/generator.py
────────────────
Grounded answer generation using Google Gemini 2.5 Flash.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
import google.generativeai as genai

from rag.prompts import SYSTEM_PROMPT, build_user_prompt

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────

_gemini_model: Any = None


def _get_gemini_model():
    """
    Lazy initialize Gemini model singleton.
    """

    global _gemini_model

    if _gemini_model is None:

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set."
            )

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Gemini 2.5 Flash
        _gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",

            system_instruction=SYSTEM_PROMPT,

            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                top_p=0.95,
                max_output_tokens=1024,
            ),
        )

        logger.info("✅ Gemini 2.5 Flash initialized.")

    return _gemini_model


# ─────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Generate grounded answer from retrieved chunks.
    """

    if not retrieved_chunks:
        return (
            "I could not find relevant information "
            "in the uploaded document."
        )

    try:

        model = _get_gemini_model()

        # Build prompt using retrieved context
        user_prompt = build_user_prompt(
            query=query,
            context_chunks=retrieved_chunks,
        )

        logger.info("Generating Gemini response...")

        response = model.generate_content(user_prompt)

        logger.info("Gemini response received.")

        # Safe extraction
        answer = getattr(response, "text", None)

        if not answer:
            return (
                "The model could not generate a valid answer "
                "from the retrieved context."
            )

        return answer.strip()

    except Exception as e:

        logger.exception("❌ Gemini generation failed")

        raise RuntimeError(
            f"LLM generation failed: {str(e)}"
        )