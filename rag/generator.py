"""
rag/generator.py
────────────────
Grounded answer generation over a chain of free LLM providers.

Provider order is `LLM_PROVIDER` first, then `LLM_FALLBACK_PROVIDERS`
(comma-separated), skipping any provider whose API key is not set.

  groq        Groq free tier, OpenAI-compatible.   GROQ_API_KEY, GROQ_MODEL
  gemini      Gemini free tier via AI Studio key.  GEMINI_API_KEY, GEMINI_MODEL
  openrouter  OpenRouter ":free" models.           OPENROUTER_API_KEY, OPENROUTER_MODEL

All calls are plain HTTPS via httpx: no provider SDK is imported, so this
module costs nothing at boot.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List

import httpx
from dotenv import load_dotenv

from rag.prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv()
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
LLM_FALLBACK_PROVIDERS = [
    p.strip().lower()
    for p in os.getenv("LLM_FALLBACK_PROVIDERS", "gemini,openrouter").split(",")
    if p.strip()
]

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.7:free")

MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.0
TIMEOUT_S = 60.0


# ── Providers ──────────────────────────────────────────────────────────────────

def _post(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    with httpx.Client(timeout=TIMEOUT_S) as client:
        resp = client.post(url, json=body, headers={"Content-Type": "application/json", **headers})
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def _openai_compatible(url: str, api_key: str, model: str, user_prompt: str, extra_headers=None) -> str:
    data = _post(
        url,
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
        },
        {"Authorization": f"Bearer {api_key}", **(extra_headers or {})},
    )
    return (data["choices"][0]["message"]["content"] or "").strip()


def _call_groq(user_prompt: str) -> str:
    return _openai_compatible(
        "https://api.groq.com/openai/v1/chat/completions",
        os.environ["GROQ_API_KEY"], GROQ_MODEL, user_prompt,
    )


def _call_openrouter(user_prompt: str) -> str:
    return _openai_compatible(
        "https://openrouter.ai/api/v1/chat/completions",
        os.environ["OPENROUTER_API_KEY"], OPENROUTER_MODEL, user_prompt,
        {"HTTP-Referer": "https://github.com/pranay-reddy2/NoteBookLLM_Backend", "X-Title": "DocMind RAG"},
    )


def _call_gemini(user_prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    data = _post(
        url,
        {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "topP": 0.95,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
            },
        },
        {"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    )
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {str(data)[:200]}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


PROVIDERS: Dict[str, tuple[str, Callable[[str], str], str]] = {
    # name: (env var holding the key, call, model label)
    "groq": ("GROQ_API_KEY", _call_groq, GROQ_MODEL),
    "gemini": ("GEMINI_API_KEY", _call_gemini, GEMINI_MODEL),
    "openrouter": ("OPENROUTER_API_KEY", _call_openrouter, OPENROUTER_MODEL),
}


def provider_chain() -> List[str]:
    """Ordered providers that are both known and have a key configured."""
    order: List[str] = []
    for name in [LLM_PROVIDER, *LLM_FALLBACK_PROVIDERS]:
        if name in PROVIDERS and name not in order:
            order.append(name)
    return [n for n in order if os.getenv(PROVIDERS[n][0])]


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Generate a grounded answer from retrieved chunks, falling through providers."""
    if not retrieved_chunks:
        return "I could not find relevant information in the uploaded document."

    chain = provider_chain()
    if not chain:
        raise EnvironmentError(
            "No LLM API key configured. Set GROQ_API_KEY (free: https://console.groq.com) "
            "or GEMINI_API_KEY (free: https://aistudio.google.com/apikey)."
        )

    user_prompt = build_user_prompt(query=query, context_chunks=retrieved_chunks)

    errors: List[str] = []
    for name in chain:
        _, call, model = PROVIDERS[name]
        try:
            logger.info(f"LLM: {name} ({model})")
            answer = call(user_prompt)
            if answer:
                return answer
            errors.append(f"{name}: empty response")
        except Exception as e:  # noqa: BLE001 - any provider failure means try the next one
            logger.warning(f"LLM provider {name} failed: {e}")
            errors.append(f"{name}: {e}")

    raise RuntimeError("All LLM providers failed - " + " | ".join(errors))
