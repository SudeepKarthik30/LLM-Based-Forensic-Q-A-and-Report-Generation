"""
config.py  —  Central Configuration for the Forensic RAG Pipeline
─────────────────────────────────────────────────────────────────────────────
All constants that were previously scattered across parsers.py, store.py,
llm.py, and query.py are now defined here.

To change any setting (host, model, limits, paths), edit ONLY this file.

Colab override note
-------------------
In Google Colab, set the environment variable QDRANT_PATH to a Drive path
(e.g. /content/drive/MyDrive/RV_internship/qdrant_data) before importing.
store.get_qdrant_client() will switch from TCP mode to local-file mode
automatically when this variable is present.
─────────────────────────────────────────────────────────────────────────────
"""

import os

# ── Qdrant ────────────────────────────────────────────────────────────────────
QDRANT_HOST      = "localhost"
QDRANT_PORT      = 6333
COLLECTION_NAME  = "forensic_events"

# ── Ollama models ─────────────────────────────────────────────────────────────
EMBED_MODEL      = "nomic-embed-text"   # Embedding model  (768 dims)
EMBED_DIM        = 768                  # Must match nomic-embed-text output
LLM_MODEL        = "llama3:8b-instruct-q4_K_M"  # q4_K_M: mixed-precision quant, better quality than q4_0

# ── Parsing limits ────────────────────────────────────────────────────────────
# Raised from 2000 to 50_000.  A loud TRUNCATION WARNING is logged if hit.
# Set to 0 to disable the cap entirely (stream-processes the whole file).
MAX_RECORDS      = 50_000

# Maximum characters in a single chunk_text before truncation.
# Prevents Ollama HTTP 500 "input length exceeds context length" on huge
# EventData payloads (registry blobs, long CommandLine values etc.).
# nomic-embed-text context is 8192 tokens ≈ ~32k chars — 6000 chars is safe.
MAX_CHUNK_CHARS  = 6000

# ── Retrieval settings ────────────────────────────────────────────────────────
DENSE_TOP_K      = 20     # Qdrant cosine results per query
SPARSE_TOP_K     = 20     # BM25 keyword results per query
RRF_K            = 60     # RRF constant (standard default)
RRF_TOP_N        = 3      # Final fused results sent to LLM
SNIPPET_CHARS    = 400    # Max chars of chunk_text sent to LLM as evidence.
                          # 5 sources × 800 chars ≈ ~1,000 tokens of evidence —
                          # prompt eval is often the bigger CPU cost than generation,
                          # so halving snippet size directly cuts prompt processing time.

# ── Confidence grounding thresholds ──────────────────────────────────────────
# These values are PLACEHOLDERS.  After the first successful ingest, run
# 10-15 sample queries covering known events (password spray, mimikatz,
# lateral movement), print raw dense_results[0][1] scores, and adjust
# these numbers to match the actual score distribution.
# nomic-embed-text cosine scores for short forensic snippets often sit
# in the 0.40-0.70 range, not the 0.75+ range these defaults assume.
# —— CALIBRATE AFTER INGEST ——
CONFIDENCE_HIGH_THRESHOLD = 0.62   # top dense score ≥ this + ≥3 sources = HIGH
CONFIDENCE_MED_THRESHOLD  = 0.55   # top dense score ≥ this OR ≥2 sources = MEDIUM
                                    # below both thresholds = LOW

# Minimum BM25 score for gate fallback (is_forensic_query_with_fallback).
# If keyword gate fails but BM25 top score exceeds this, query is allowed.
# —— CALIBRATE AFTER INGEST ——
BM25_FALLBACK_THRESHOLD    = 2.0    # typical BM25 scores on 30k corpus: 1-10

# ── LLM generation settings ───────────────────────────────────────────────────
LLM_TEMPERATURE  = 0.0    # Deterministic — no randomness
LLM_NUM_CTX      = 4096   # Context window for Q&A calls (tokens)
                          # Reduced 8192→4096: actual prompt (RRF_TOP_N=5 sources ×
                          # SNIPPET_CHARS=800 + prior context) stays well under 4096.
                          # Smaller ctx = less KV-cache memory alloc = faster first token.
REPORT_NUM_CTX   = 16384  # Larger window for Phase 4 report generation call
REPORT_MAX_TURNS = 20     # Cap chat history at this many turns for report
                          # generation to avoid context overflow. A warning is
                          # printed if the session had more turns than this.

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve relative to this file's location so scripts work from any CWD.
_CODE_DIR        = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR        = os.path.dirname(_CODE_DIR)

INPUT_DIR        = os.path.join(_BASE_DIR, "sample_data")   # raw artifacts
OUTPUT_DIR       = os.path.join(_BASE_DIR, "output")        # parsed JSON + reports
BM25_CACHE       = os.path.join(_CODE_DIR, "bm25_cache.pkl")
EMBED_CHECKPOINT = os.path.join(_CODE_DIR, "embed_checkpoint.pkl")
# Cleared automatically by ingest.py after a successful Qdrant upsert.
# If the ingest run crashes, restart ingest.py — it will resume from here.

# ── Audit / logging ───────────────────────────────────────────────────────────
AUDIT_LOG   = os.path.join(_CODE_DIR, "forensic_audit.log")
PARSER_LOG  = os.path.join(_CODE_DIR, "parser_errors.log")
