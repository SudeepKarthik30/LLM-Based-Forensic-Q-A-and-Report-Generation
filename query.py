#!/usr/bin/env python3
"""
query.py  —  INTERACTIVE FORENSIC Q&A TERMINAL  (Phase 2 query pipeline)
─────────────────────────────────────────────────────────────────────────────
Run this AFTER  ingest.py  to start asking questions about your forensic data.

The full pipeline for each question:
  Step 1  →  Forensic relevance gate  (reject off-topic questions instantly)
  Step 2  →  Dense retrieval          (Qdrant cosine similarity, top-20)
  Step 3  →  Sparse retrieval         (BM25 keyword match, top-20)
  Step 4  →  RRF fusion               (merge both lists, keep top-10)
  Step 5  →  Context assembly         (build numbered EVIDENCE block)
  Step 6  →  LLM generation           (llama3:8b, temperature=0)
  Step 7  →  Citation validation      (flag/reject hallucinated [Source N])
  Step 8  →  Print result + audit log (EVIDENCE + QUESTION + ANSWER)

Special commands at the prompt
───────────────────────────────
  report        →  Generate Phase 4 Markdown forensic report from this session
  exit / quit   →  End the session

Usage
─────
  cd  RV_internship/code
  python query.py

Changes from original
─────────────────────
• All constants/paths imported from config.py.
• Audit logging: every question + answer is appended to forensic_audit.log
  (timestamped, UTC) so the investigation has a permanent chain-of-custody
  record of who asked what and got what answer.
• Ollama health check at startup: a test embedding call is made before the
  Q&A loop begins.  If Ollama is down, the user gets a clean error message
  and the script exits gracefully instead of crashing mid-demo.
• generate_answer replaced with generate_answer_validated (includes citation
  validation + retry).
• chat_history accumulates {question, answer, sources} dicts per turn for
  the Phase 4 report generator.
• 'report' command triggers report_generator.generate_report().

Known limitations
─────────────────
• All event payloads are loaded into memory at startup via Qdrant scroll.
  This works well at ~10k–50k event scale but will not scale to millions of
  events.  For production use, BM25 result lookups should use Qdrant's
  retrieve() API per-query instead of a full in-memory map.
  Documented here intentionally — not hidden.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import logging
import logging.handlers
from datetime import datetime, timezone

# Make sure Python can find sibling modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config    import (
    COLLECTION_NAME, BM25_CACHE, AUDIT_LOG,
    DENSE_TOP_K, SPARSE_TOP_K, RRF_K, RRF_TOP_N,
    EMBED_MODEL, LLM_MODEL,
)
from store     import get_qdrant_client, load_bm25
from retriever import (
    is_forensic_query_with_fallback,
    dense_search, sparse_search, rrf_fuse,
    build_context_and_sources, compute_retrieval_confidence,
)
from llm       import generate_answer_validated, format_output
from report_generator import generate_report


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOGGER  (chain-of-custody log of every query + answer)
# ─────────────────────────────────────────────────────────────────────────────

def _setup_audit_logger():
    """
    Configures the forensic audit logger that writes every Q&A pair to
    forensic_audit.log (path from config.py).

    Format per entry:
        2024-01-15T10:23:45Z | QUERY    | <question>
        2024-01-15T10:23:46Z | ANSWER   | <answer (first 500 chars)>
        2024-01-15T10:23:46Z | SOURCES  | src1.evtx, src2.pcap, ...

    The log rotates at 10 MB and keeps 5 backups.
    This is the audit trail a forensic tool needs for chain-of-custody.
    """
    logger = logging.getLogger("forensic.audit")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    fh = logging.handlers.RotatingFileHandler(
        AUDIT_LOG, maxBytes=10 * 1024 * 1024, backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s UTC | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    ))

    logger.addHandler(fh)
    return logger


_audit = _setup_audit_logger()


def _log_query(question, answer, sources, gate_reason="keyword", conf_signals=None):
    """
    Appends one Q&A turn to the audit log.

    Inputs
    ------
    question     : user question string
    answer       : validated LLM answer string
    sources      : list of source metadata dicts
    gate_reason  : how the query passed the relevance gate
    conf_signals : dict from compute_retrieval_confidence
    """
    source_summary = ", ".join(
        f"{s.get('source_file','?')}[EID:{s.get('event_id','?')}]"
        for s in sources
    ) or "none"

    _audit.info("QUERY   | %s", question)
    _audit.info("ANSWER  | %s", answer[:500].replace('\n', ' '))
    _audit.info("SOURCES | %s", source_summary)
    _audit.info("GATE    | %s", gate_reason)
    if conf_signals:
        _audit.info("CONF    | top_dense=%.4f sources=%d",
                    conf_signals.get('top_dense_score', 0),
                    conf_signals.get('source_count', 0))
    _audit.info("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def step(num, label):
    """Prints a numbered step indicator so users can watch each stage in real time."""
    print(f"  Step {num}  →  {label} ...", end=" ", flush=True)


def ok(note=""):
    """Prints [OK] after a step completes, with an optional note."""
    print(f"[OK]  {note}")


def load_all_payloads_from_qdrant(client):
    """
    Fetches ALL event payloads from Qdrant into a local Python dict.

    We need this because BM25 returns chunk IDs but not payloads — Qdrant only
    returns payloads for the points it finds in a vector search.  By pre-loading
    all payloads once at startup we can look up any chunk_id in O(1).

    We use Qdrant's scroll API (paginated) so it works even for very large
    collections without loading everything at once per page.

    Known limitation: this holds ALL event payloads in RAM.  Works well at
    10k–50k event scale.  For production-grade scale (millions of events),
    replace with per-query Qdrant retrieve() calls instead.

    Input  : QdrantClient
    Output : dict  {chunk_id (int) → payload dict}
    """
    payloads = {}
    offset   = None

    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=offset,
            limit=1000,           # fetch 1000 points per page
            with_payload=True,
            with_vectors=False,   # we don't need the raw vectors here
        )
        for point in results:
            payloads[point.id] = point.payload

        if next_offset is None:
            break                 # last page reached
        offset = next_offset

    return payloads


def banner():
    """Welcome banner shown once at startup."""
    w = 66
    print()
    print("╔" + "═" * w + "╗")
    print("║" + "  [INFO]   FORENSIC RAG QUERY TERMINAL".center(w) + "║")
    print("║" + f"  {EMBED_MODEL}  +  BM25  +  RRF  +  {LLM_MODEL}".center(w) + "║")
    print("╚" + "═" * w + "╝")
    print()
    print("  Type your forensic question and press Enter.")
    print("  Type  report  to generate a Phase 4 forensic report.")
    print("  Type  exit  or  quit  to stop.\n")


# ─────────────────────────────────────────────────────────────────────────────
# QUERY PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_query(question, client, bm25, corpus_tokens, chunk_ids, payloads_map, chat_history, filter_params=None):
    """
    Runs the complete RAG pipeline for one user question and prints the result.

    Pipeline stages:
      1.  Forensic relevance gate   — keyword check + BM25 fallback
      2.  Dense retrieval           — Qdrant cosine similarity
      3.  Sparse retrieval          — BM25 keyword matching
      4.  RRF fusion                — merge dense + sparse ranked lists
      5.  Context assembly          — build numbered EVIDENCE block (injection-guarded)
      5.5 Confidence computation    — grounded HIGH/MEDIUM/LOW from retrieval signals
      6.  LLM generation            — send context + prior turns + confidence hint
      7.  Citation validation       — flag/reject hallucinated [Source N]
      7.5 Confidence override       — prevent LLM from self-reporting higher than signal
      8.  Audit log + accumulate    — write to forensic_audit.log and chat_history

    Inputs
    ------
    question     : user question string
    client       : QdrantClient (shared, opened once at startup)
    bm25         : BM25Okapi index
    corpus_tokens: tokenized corpus for BM25
    chunk_ids    : list of integer chunk IDs corresponding to corpus positions
    payloads_map : dict chunk_id → Qdrant payload (pre-loaded at startup)
    chat_history : list accumulated across the session — modified in place
    """
    print(f"\n{'─' * 66}")
    t_start = time.time()

    # ── Step 1: Forensic gate (two-stage: keyword + BM25 fallback) ─────────────
    t_retrieval_start = time.time()
    step(1, "Forensic relevance check")
    allowed, gate_reason = is_forensic_query_with_fallback(
        question, bm25, corpus_tokens, chunk_ids
    )

    if not allowed:
        ok("[WARNING]  NOT forensic")
        print()
        print("  " + "═" * 66)
        print("  [WARNING]   IRRELEVANT QUERY")
        print("  " + "═" * 66)
        print("  This system is designed for forensic investigation only.")
        print("  Please ask questions related to:")
        print("    • Login attempts / authentication events")
        print("    • Process creation / suspicious executables")
        print("    • Network connections / IP addresses / ports")
        print("    • Malware activity / persistence mechanisms")
        print("    • Incident response / log analysis")
        print("  " + "═" * 66)
        _audit.info("QUERY   | %s", question)
        _audit.info("ANSWER  | REJECTED — failed forensic relevance gate")
        _audit.info("─" * 60)
        return

    if gate_reason == "bm25_fallback":
        ok("forensic   (BM25 fallback)")
        print(f"  [WARNING]  Gate: keyword check failed, BM25 signal strong enough — proceeding")
    else:
        ok("forensic ")

    # ── Step 2: Dense retrieval ───────────────────────────────────────────────
    step(2, f"Dense retrieval  (Qdrant cosine similarity, top-{DENSE_TOP_K})")
    dense_results = dense_search(client, question, top_k=DENSE_TOP_K, filter_params=filter_params)
    ok(f"{len(dense_results)} results")

    # ── Step 3: Sparse retrieval ───────────────────────────────────────────────
    step(3, f"Sparse retrieval (BM25 keyword match, top-{SPARSE_TOP_K})")
    sparse_results = sparse_search(bm25, corpus_tokens, chunk_ids, question, top_k=SPARSE_TOP_K)
    ok(f"{len(sparse_results)} results")

    # ── Step 4: RRF fusion ───────────────────────────────────────────────────
    step(4, f"RRF fusion        (k={RRF_K}, merging dense + sparse)")
    fused_ids = rrf_fuse(dense_results, sparse_results, k=RRF_K, top_n=RRF_TOP_N)
    ok(f"{len(fused_ids)} fused results")

    # ── Step 5: Context assembly (with injection guard) ───────────────────────
    step(5, "Assembling EVIDENCE context with citations")
    context, sources = build_context_and_sources(fused_ids, payloads_map)
    ok(f"{len(sources)} sources assembled")

    if not sources:
        print()
        print("  " + "═" * 66)
        print("  INSUFFICIENT EVIDENCE")
        print("  " + "═" * 66)
        print("  No matching events were found in the forensic database for this query.")
        print("  Try rephrasing, or make sure ingest.py has been run on relevant files.")
        print("  " + "═" * 66)
        _audit.info("QUERY   | %s", question)
        _audit.info("ANSWER  | INSUFFICIENT EVIDENCE — no matching events found")
        _audit.info("─" * 60)
        return

    # ── Step 5.5: Compute grounded retrieval confidence ────────────────────────
    retrieval_confidence, conf_signals = compute_retrieval_confidence(dense_results, sources)

    t_retrieval_elapsed = time.time() - t_retrieval_start
    print(f"  [TIME]  Retrieval (gate+dense+sparse+RRF+context): {t_retrieval_elapsed:.2f}s")

    # ── Step 6: LLM generation + Step 7: Citation validation ─────────────────
    t_llm_start = time.time()
    step(6, f"LLM generation + citation validation  ({LLM_MODEL}, temp=0)")
    try:
        answer = generate_answer_validated(
            question, context, sources, max_retries=1,
            chat_history=chat_history,
            retrieval_confidence=retrieval_confidence,
        )
    except RuntimeError as exc:
        ok("FAILED")
        print(f"\n  [FAIL]  {exc}")
        _audit.info("QUERY   | %s", question)
        _audit.info("ANSWER  | LLM ERROR — %s", str(exc)[:200])
        _audit.info("─" * 60)
        return
    ok("done")

    t_llm_elapsed = time.time() - t_llm_start
    print(f"  [TIME]  LLM generation (ollama.chat): {t_llm_elapsed:.2f}s")

    elapsed = time.time() - t_start
    print(f"\n  [TIME]  Total query time: {elapsed:.2f}s")

    # ── Step 8: Audit log + accumulate chat history ───────────────────────────
    _log_query(question, answer, sources, gate_reason=gate_reason, conf_signals=conf_signals)
    chat_history.append({
        "question": question,
        "answer":   answer,
        "sources":  sources,
    })

    # ── Print formatted result ────────────────────────────────────────────────
    print(format_output(question, context, sources, answer))


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP & MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def _check_ollama():
    """
    Startup health check: tries a lightweight Ollama embeddings call to confirm
    that Ollama is running and the embedding model is available.

    Raises RuntimeError with a clean message if Ollama is not reachable,
    so main() can exit gracefully before the Q&A loop starts.
    """
    import ollama as _ollama
    from config import EMBED_MODEL
    try:
        # Minimal test call — just checks reachability and model availability
        _ollama.embeddings(model=EMBED_MODEL, prompt="health-check")
    except Exception as exc:
        raise RuntimeError(
            f"Ollama is not reachable or '{EMBED_MODEL}' is not pulled.\n"
            f"  Start Ollama, then run:\n"
            f"      ollama pull {EMBED_MODEL}\n"
            f"      ollama pull {LLM_MODEL}\n"
            f"  Original error: {exc}"
        ) from exc


def main():
    banner()

    # ── Load all components once at startup ───────────────────────────────────
    print("    Loading retrieval components ...\n")

    # 1. Ollama health check
    print("  Checking Ollama connectivity ...", end=" ", flush=True)
    try:
        _check_ollama()
        print(f"[OK]  Ollama up  ({EMBED_MODEL} ready)")
    except RuntimeError as exc:
        print("[FAIL]  FAILED")
        print(f"\n  [FAIL]  {exc}\n")
        sys.exit(1)

    # 2. Connect to Qdrant
    print("  Connecting to Qdrant ...", end=" ", flush=True)
    try:
        client    = get_qdrant_client()
        vec_count = client.count(collection_name=COLLECTION_NAME).count
        print(f"[OK]  {vec_count:,} vectors in '{COLLECTION_NAME}'")
    except Exception as exc:
        print(f"[FAIL]  FAILED: {exc}")
        print("\n  [FAIL]  Qdrant is not reachable.")
        print("      Run  ingest.py  first, and make sure Qdrant is running:\n")
        print("      docker run -p 6333:6333 qdrant/qdrant\n")
        sys.exit(1)

    # 3. Load BM25 index from disk
    print("  Loading BM25 index  ...", end=" ", flush=True)
    try:
        bm25, corpus_tokens, chunk_ids = load_bm25(BM25_CACHE)
        print(f"[OK]  {len(chunk_ids):,} documents indexed")
    except FileNotFoundError:
        print("[FAIL]  NOT FOUND")
        print(f"\n  [FAIL]  BM25 cache not found at:\n  {BM25_CACHE}")
        print("      Run  ingest.py  first to build the index.\n")
        sys.exit(1)

    # 4. Pre-load all payloads from Qdrant (needed for BM25 result lookups)
    print("  Loading event payloads ...", end=" ", flush=True)
    payloads_map = load_all_payloads_from_qdrant(client)
    print(f"[OK]  {len(payloads_map):,} events in memory")

    print(f"\n  [OK]  Ready!  {vec_count:,} events indexed and searchable.")
    print(f"    Audit log → {AUDIT_LOG}")
    print("  " + "═" * 66)

    # ── Interactive Q&A loop ──────────────────────────────────────────────────
    chat_history = []   # accumulates {question, answer, sources} per turn

    while True:
        try:
            question = input("\n[INFO]  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  [INFO]  Goodbye!\n")
            break

        if not question:
            print("  (Please type a question and press Enter)")
            continue

        if question.lower() in ('exit', 'quit', 'q', 'bye', 'exit()', 'quit()'):
            print("\n  [INFO]  Goodbye!\n")
            break

        # ── Special command: generate Phase 4 report ──────────────────────────
        if question.lower() == 'report':
            if not chat_history:
                print("  [WARNING]  No questions asked yet — run some queries first.")
                continue
            print(f"\n  [FILE]  Generating forensic report from {len(chat_history)} Q&A turn(s)...\n")
            result = generate_report(chat_history)
            if result:
                print(f"\n  [OK]  MD report  → {result['md']}")
                if result['pdf']:
                    print(f"  [OK]  PDF report → {result['pdf']}")
            continue

        run_query(question, client, bm25, corpus_tokens, chunk_ids, payloads_map, chat_history)


if __name__ == "__main__":
    main()
