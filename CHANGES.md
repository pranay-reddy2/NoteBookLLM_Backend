# Changes: demo reliability (branch `fix/demo-reliability`)

## Why
1. Paid API credits were exhausted, so every demo request failed.
2. Cold start on Render free tier took ~2 minutes because `import app` pulled in
   sentence-transformers → torch → sklearn (32 s of import on a fast laptop, far
   worse on a 512 MB Render instance), then loaded the MiniLM model and opened
   ChromaDB before the first request could be served.

## What changed
| Area | Before | After |
|---|---|---|
| Embeddings | Local MiniLM via sentence-transformers, loaded at boot | Gemini `gemini-embedding-001` (768-dim) over HTTPS, free tier. `EMBEDDING_PROVIDER=local` keeps the old path, imported lazily on first use. |
| LLM | Gemini SDK, single provider | Provider chain over plain HTTPS: `groq` → `gemini` → `openrouter` (":free" models). No SDKs imported at boot. |
| ChromaDB | Opened at startup | Opened lazily on first real request. Collection name now carries the embedding signature (`docmind_gemini-embedding-001-768`). |
| Legacy index | MiniLM 384-dim collection `docmind_chunks` | Auto-migrated (re-embedded) once, on first real request, when the new collection is empty. `scripts/reindex.py` does the same by hand. |
| `GET /health` | Opened Chroma, returned stats | Returns `{"ok": true}` and touches nothing. Old stats moved to `GET /stats`. |
| Daily cap | none | `DAILY_REQUEST_CAP` (default 200) on `/ask` and `/upload`, in-memory, resets at UTC midnight. Over cap, `/ask` returns 200 with a friendly "email pranayreddy672@gmail.com" answer and `limit_reached: true`. |
| Answer cache | none | In-memory LRU keyed on (document ids, normalised question, top_k). Cache hits skip the cap. Cleared on upload/delete. |
| requirements | torch, transformers, sentence-transformers, langchain, langchain-community, google-generativeai | Removed. Added `langchain-text-splitters` (small). Local embeddings live in `requirements-local-embeddings.txt`. |
| render.yaml | | `healthCheckPath: /health`, new env vars, Python 3.11.9. |

## Measured (same laptop, same venv)
| Metric | Before | After |
|---|---|---|
| `import app` | 37.2 s | 2.8 s |
| Process start → first `/health` response | >600 s (timed out at 10 min; import alone is 37 s, then MiniLM model load) | 2.8 s |
| Warm `/health` latency | n/a (opened Chroma) | ~2 ms |

On Render free tier expect roughly 5–10 s after the container is up (pip install
is also much faster now that torch is gone).

## Env vars you now need
| Var | Required | Where |
|---|---|---|
| `GROQ_API_KEY` | yes (primary LLM) | https://console.groq.com/keys |
| `GEMINI_API_KEY` | yes (embeddings + LLM fallback) | https://aistudio.google.com/apikey — must start with `AIza` |
| `OPENROUTER_API_KEY` | optional | https://openrouter.ai/keys |
| `EMBEDDING_PROVIDER` | default `gemini` | `gemini` or `local` |
| `LLM_PROVIDER` / `LLM_FALLBACK_PROVIDERS` | default `groq` / `gemini,openrouter` | |
| `DAILY_REQUEST_CAP` | default `200` | |
| `CONTACT_EMAIL` | default `pranayreddy672@gmail.com` | |

See `.env.example` for the full list. Never commit `.env`.

## Note on the first commit on this branch
`ef63644 WIP: simplified RAG pipeline as staged locally` commits the changes that
were already staged in the working copy (the reversion from the "Advanced RAG 2"
pipeline to the simpler one). The reliability fixes sit on top of it.
