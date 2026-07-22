#!/usr/bin/env python3
"""
ingest.py  —  MAIN INGESTION SCRIPT  (Phase 1 + Phase 2, fully automated)
─────────────────────────────────────────────────────────────────────────────
Run this ONCE (or whenever you get new forensic data) to:

  Phase 1  →  Parse raw artifacts (.evtx / .pcap / .csv / .log)
  Phase 2A →  Build one chunk per event
  Phase 2B →  Embed every chunk with nomic-embed-text (via Ollama)
  Phase 2C →  Store vectors + metadata in Qdrant (persistent)
  Phase 2D →  Build BM25 keyword index and save to disk

After this completes, run  query.py  to start asking questions.

Usage
─────
  cd  RV_internship/code
  python ingest.py

Prerequisites
─────────────
  • Qdrant running:  docker run -p 6333:6333 qdrant/qdrant
  • Ollama running with models pulled:
      ollama pull nomic-embed-text
      ollama pull llama3:8b

Changes from original
─────────────────────
• All paths / constants imported from config.py.
• Qdrant collection name read from config.py (no inline string).
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
from collections import Counter

# Make sure Python can find our sibling modules (parsers, store, etc.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config  import INPUT_DIR, OUTPUT_DIR, BM25_CACHE, COLLECTION_NAME, EMBED_CHECKPOINT
from parsers import parse_all_files
from store   import (
    make_chunks, embed_batch,
    get_qdrant_client, ensure_collection, upsert_to_qdrant,
    build_and_save_bm25,
)

# Set this to True only if you want to WIPE the Qdrant collection and re-ingest
FORCE_RECREATE = False


# ─── PRETTY PRINTING HELPERS ─────────────────────────────────────────────────

def banner():
    """Prints the startup banner so users know what's about to happen."""
    w = 66
    print()
    print("╔" + "═" * w + "╗")
    print("║" + "  [SEARCH]  FORENSIC RAG INGESTION PIPELINE".center(w) + "║")
    print("║" + "  Phase 1 (Parse) + Phase 2 (Embed + Index)".center(w) + "║")
    print("╚" + "═" * w + "╝")
    print()


def section(title):
    """Prints a clear section header between pipeline stages."""
    print(f"\n{'─' * 66}")
    print(f"  {title}")
    print(f"{'─' * 66}")


def done_line(label, elapsed, extra=""):
    """Prints a [OK] summary line at the end of each stage."""
    print(f"\n  [OK]  {label}  completed in  {elapsed:.1f}s  {extra}")


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────

def main():
    banner()
    wall_start = time.time()

    # ── PHASE 1: PARSE RAW ARTIFACTS ─────────────────────────────────────────
    section("PHASE 1  —  Parsing Raw Forensic Artifacts")
    print(f"  [DIR]  Input  directory : {INPUT_DIR}")
    print(f"  [DIR]  Output directory : {OUTPUT_DIR}")
    print()

    t = time.time()
    events = parse_all_files(INPUT_DIR, OUTPUT_DIR)
    t1 = time.time() - t

    if not events:
        print("\n  [FAIL]  No events were parsed.")
        print("      Make sure sample_data/ contains .evtx / .pcap / .csv files.\n")
        sys.exit(1)

    fmt_counts = Counter(e["format"] for e in events)
    done_line("Phase 1 — Parsing", t1, f"({len(events):,} events total)")
    print()
    for fmt, cnt in sorted(fmt_counts.items()):
        print(f"       {fmt:<8} : {cnt:>7,} events")
    print(f"\n  Combined JSON → {os.path.join(OUTPUT_DIR, 'all_artifacts.json')}")

    # ── PHASE 2A: CHUNKING ────────────────────────────────────────────────────
    section("PHASE 2A  —  Building Chunks  (1 chunk = 1 event)")

    t = time.time()
    chunks = make_chunks(events)
    t2a = time.time() - t

    done_line("Phase 2A — Chunking", t2a, f"({len(chunks):,} chunks built)")

    # ── PHASE 2B: EMBEDDING ───────────────────────────────────────────────────
    section("PHASE 2B  —  Generating Embeddings  (nomic-embed-text via Ollama)")
    print(f"  Model     : nomic-embed-text")
    print(f"  Dimension : 768")
    print(f"  Chunks    : {len(chunks):,}")
    print(f"  ⏳  This is the slowest step — grab a coffee \n")

    t = time.time()
    texts      = [c["chunk_text"] for c in chunks]
    embeddings = embed_batch(texts)   # resumes from checkpoint if one exists
    t2b = time.time() - t

    good_vectors = sum(1 for e in embeddings if e is not None)
    done_line("Phase 2B — Embedding", t2b,
              f"({good_vectors:,} vectors, {len(embeddings)-good_vectors} failed)")

    # ── PHASE 2C: QDRANT STORAGE ──────────────────────────────────────────────
    section("PHASE 2C  —  Storing Vectors in Qdrant")
    print("  Connecting to Qdrant at localhost:6333 ...")

    try:
        client = get_qdrant_client()
        client.get_collections()       # will throw if Qdrant isn't running
        print("  [OK]  Connected to Qdrant\n")
    except Exception as exc:
        print(f"\n  [FAIL]  Cannot reach Qdrant: {exc}")
        print("      Start Qdrant with:")
        print("      docker run -p 6333:6333 qdrant/qdrant\n")
        sys.exit(1)

    t = time.time()
    ensure_collection(client, force_recreate=FORCE_RECREATE)
    upsert_to_qdrant(client, chunks, embeddings)
    t2c = time.time() - t

    # Clear the embedding checkpoint — ingest succeeded, next run starts fresh
    if os.path.exists(EMBED_CHECKPOINT):
        os.remove(EMBED_CHECKPOINT)
        print("  ️  Embedding checkpoint cleared (ingest complete).")

    stored_count = client.count(collection_name=COLLECTION_NAME).count
    done_line("Phase 2C — Qdrant", t2c, f"({stored_count:,} vectors in collection)")

    # ── PHASE 2D: BM25 INDEX ─────────────────────────────────────────────────
    section("PHASE 2D  —  Building BM25 Keyword Index")
    print(f"  Cache file : {BM25_CACHE}\n")

    t = time.time()
    build_and_save_bm25(chunks, BM25_CACHE)
    t2d = time.time() - t

    done_line("Phase 2D — BM25", t2d)

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    total = time.time() - wall_start
    w = 66
    print()
    print("╔" + "═" * w + "╗")
    print("║" + "    ALL PHASES COMPLETE!".center(w) + "║")
    print("╠" + "═" * w + "╣")
    print(f"║  Phase 1  — Parsing   : {t1:>7.1f}s   {len(events):,} events".ljust(w + 1) + "║")
    print(f"║  Phase 2A — Chunking  : {t2a:>7.2f}s   {len(chunks):,} chunks".ljust(w + 1) + "║")
    print(f"║  Phase 2B — Embedding : {t2b:>7.1f}s   nomic-embed-text".ljust(w + 1) + "║")
    print(f"║  Phase 2C — Qdrant    : {t2c:>7.1f}s   {stored_count:,} vectors stored".ljust(w + 1) + "║")
    print(f"║  Phase 2D — BM25      : {t2d:>7.2f}s   keyword index saved".ljust(w + 1) + "║")
    print(f"║  ─────────────────────────────────────────────────────────────  ║")
    print(f"║  Total Time           : {total:>7.1f}s".ljust(w + 1) + "║")
    print("╠" + "═" * w + "╣")
    print("║" + "    Next step:  python query.py".center(w) + "║")
    print("╚" + "═" * w + "╝")
    print()


if __name__ == "__main__":
    main()
