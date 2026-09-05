"""
One-off: re-embed the legacy MiniLM ChromaDB collection into the collection
for the current EMBEDDING_PROVIDER, persisting to disk.

    python scripts/reindex.py

The server also does this automatically on the first real request when the
new collection is empty (AUTO_MIGRATE_LEGACY=1), so this script is only for
running the migration ahead of time or re-running it by hand.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from rag import retriever  # noqa: E402

t0 = time.perf_counter()
os.environ.setdefault("AUTO_MIGRATE_LEGACY", "0")   # we call it explicitly below
retriever.initialize_vector_store()
n = retriever.migrate_legacy_collection()
stats = retriever.get_collection_stats()
print(f"migrated {n} chunks in {time.perf_counter() - t0:.1f}s -> {stats}")
