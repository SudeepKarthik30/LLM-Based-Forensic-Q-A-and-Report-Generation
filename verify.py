"""
verify.py — Functional verification of all code review fixes.
Run with:  python -X utf8 verify.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0

def check(label, condition):
    global PASS, FAIL
    if condition:
        print(f"  [PASS]  {label}")
        PASS += 1
    else:
        print(f"  [FAIL]  {label}")
        FAIL += 1


# ─── TEST 1: Timestamp normalisation ─────────────────────────────────────────
print("\n=== TEST 1: Timestamp normalisation ===")
from parsers import _to_iso_utc

cases = [
    ("2019-04-30 18:08:29.123456 UTC", "2019-04-30T18:08:29Z"),
    ("2019-04-30T18:08:29.123456",     "2019-04-30T18:08:29Z"),
    ("2019-04-30T18:08:29Z",           "2019-04-30T18:08:29Z"),
    ("Unknown",                         "Unknown"),
    ("",                                "Unknown"),
]
for raw, expected in cases:
    result = _to_iso_utc(raw, "test")
    check(f"_to_iso_utc({raw!r}) == {expected!r}  (got {result!r})",
          result == expected)


# ─── TEST 2: Forensic gate ────────────────────────────────────────────────────
print("\n=== TEST 2: Forensic relevance gate (tightened) ===")
from retriever import is_forensic_query

gate_cases = [
    ("who was the first president",     False),   # no forensic tokens
    ("how do I cook pasta",             False),   # no forensic tokens
    ("show me failed logon events",     True),    # failed + logon = 2
    ("what process used port 4444",     True),    # process + port = 2
    ("find mimikatz malware artifacts", True),    # 3 matches
    ("network traffic analysis",        True),    # network + traffic = 2
]
for question, expected in gate_cases:
    result = is_forensic_query(question)
    check(f"gate({question!r}) == {expected}  (got {result})", result == expected)


# ─── TEST 3: Citation validator ───────────────────────────────────────────────
print("\n=== TEST 3: Citation validator ===")
from llm import validate_citations

sources_3 = [{"num": i} for i in range(1, 4)]  # Sources 1, 2, 3

# 3a: Valid citation — no warning
answer_ok = "The attacker used mimikatz [Source 2] to dump credentials."
validated, had_bad = validate_citations(answer_ok, sources_3)
check("Valid [Source 2] — had_bad=False", not had_bad)
check("Valid [Source 2] — original text preserved", "[Source 2]" in validated)

# 3b: Hallucinated [Source 99] — flagged inline, warning at top
answer_bad = "Something happened [Source 99] somewhere."
validated, had_bad = validate_citations(answer_bad, sources_3)
check("[Source 99] — had_bad=True", had_bad)
check("[Source 99] — flagged inline", "INVALID CITATION" in validated)
check("[Source 99] — warning at TOP of answer", "CITATION WARNING" in validated[:120])

# 3c: Mixed valid + invalid
answer_mix = "Event [Source 1] then [Source 0] and [Source 3]."
validated, had_bad = validate_citations(answer_mix, sources_3)
check("Mixed — had_bad=True (Source 0 is invalid)", had_bad)
check("Mixed — [Source 1] kept valid", "[Source 1]" in validated and "INVALID" not in validated.split("[Source 1]")[1][:5])
check("Mixed — [Source 0] flagged", "INVALID CITATION" in validated)
check("Mixed — [Source 3] kept valid", "[Source 3]" in validated)

# 3d: Empty sources — validator skips, no crash
validated_empty, _ = validate_citations("Some answer [Source 1].", [])
check("Empty sources — no crash, no INVALID in output", "INVALID" not in validated_empty)


# ─── TEST 4: Audit log ────────────────────────────────────────────────────────
print("\n=== TEST 4: Audit log creation ===")
from config import AUDIT_LOG
from query import _log_query
import logging

# Close all handlers on the audit logger so we can inspect/remove the file
_audit_logger = logging.getLogger("forensic.audit")
for h in list(_audit_logger.handlers):
    h.flush()
    h.close()
    _audit_logger.removeHandler(h)

# Remove stale log from previous test run if present
if os.path.exists(AUDIT_LOG):
    try:
        os.remove(AUDIT_LOG)
    except PermissionError:
        pass   # file still locked — just append to it, test still valid

# Re-setup logger and log a test entry
from query import _setup_audit_logger
_audit_logger2 = _setup_audit_logger()

_log_query(
    "test question about malware",
    "The malware was [Source 1].",
    [{"source_file": "test.evtx", "event_id": "4688"}],
)

# Flush so content is on disk before we read it
for h in _audit_logger2.handlers:
    h.flush()

exists = os.path.exists(AUDIT_LOG)
check("Audit log file created", exists)
if exists:
    with open(AUDIT_LOG, encoding="utf-8") as f:
        content = f.read()
    check("Audit log contains QUERY entry", "test question about malware" in content)
    check("Audit log contains ANSWER entry", "The malware was" in content)
    check("Audit log contains SOURCES entry", "test.evtx" in content)


# ─── TEST 5: Evidence Inventory (deduplicated) ────────────────────────────────
print("\n=== TEST 5: Evidence Inventory builder (dedup) ===")
from report_generator import _build_evidence_inventory

mock_history = [
    {"question": "q1", "answer": "a1", "sources": [
        {"num": 1, "source_file": "mimikatz.evtx", "event_id": "4688",
         "event_type": "New Process Created",
         "timestamp": "2019-04-30T18:08:29Z", "hostname": "DC01"},
        {"num": 2, "source_file": "psexec.evtx", "event_id": "7045",
         "event_type": "New Service Installed",
         "timestamp": "2019-04-30T18:09:00Z", "hostname": "DC01"},
    ]},
    {"question": "q2", "answer": "a2", "sources": [
        # Exact duplicate of the first source — must be deduped
        {"num": 1, "source_file": "mimikatz.evtx", "event_id": "4688",
         "event_type": "New Process Created",
         "timestamp": "2019-04-30T18:08:29Z", "hostname": "DC01"},
        {"num": 2, "source_file": "new_user.evtx", "event_id": "4720",
         "event_type": "User Account Created",
         "timestamp": "2019-04-30T18:10:00Z", "hostname": "DC01"},
    ]},
]

inventory = _build_evidence_inventory(mock_history)
check("Inventory table has header row", "| # | Source File" in inventory)
check("mimikatz.evtx appears exactly once (deduped)", inventory.count("mimikatz.evtx") == 1)
check("psexec.evtx present", "psexec.evtx" in inventory)
check("new_user.evtx present", "new_user.evtx" in inventory)
# 3 unique sources across both turns
unique_rows = [l for l in inventory.splitlines() if l.startswith("| ") and "Source File" not in l and "---" not in l]
check(f"Exactly 3 unique rows (got {len(unique_rows)})", len(unique_rows) == 3)


# ─── TEST 6: config values reachable from all modules ─────────────────────────
print("\n=== TEST 6: Config values consistency ===")
import config, store, retriever, llm, query, ingest, report_generator

check("store.COLLECTION_NAME == config.COLLECTION_NAME",
      store.COLLECTION_NAME == config.COLLECTION_NAME)
check("config.SNIPPET_CHARS == 1500", config.SNIPPET_CHARS == 1500)
check("config.MAX_RECORDS == 50000", config.MAX_RECORDS == 50_000)
check("config.REPORT_MAX_TURNS == 20", config.REPORT_MAX_TURNS == 20)
check("config.REPORT_NUM_CTX == 16384", config.REPORT_NUM_CTX == 16384)
check("config.RRF_K == 60", config.RRF_K == 60)
check("retriever.FORENSIC_KEYWORDS does not contain 'who'",
      "who" not in retriever.FORENSIC_KEYWORDS)
check("retriever.FORENSIC_KEYWORDS does not contain 'was'",
      "was" not in retriever.FORENSIC_KEYWORDS)
check("retriever.FORENSIC_KEYWORDS does not contain 'how'",
      "how" not in retriever.FORENSIC_KEYWORDS)


# ─── TEST 7: RRF fusion — hand-computed expected ranking ──────────────────────────────────
print("\n=== TEST 7: RRF fusion (hand-computed) ===")
from retriever import rrf_fuse

# Setup:
#   dense : chunk 10 rank-0, chunk 20 rank-1, chunk 30 rank-2
#   sparse: chunk 10 rank-0, chunk 20 rank-1
#   k=60
#
# Expected RRF scores (exact fractions):
#   score(10) = 1/(60+0+1) + 1/(60+0+1) = 2/61 ~ 0.032787
#   score(20) = 1/(60+1+1) + 1/(60+1+1) = 2/62 ~ 0.032258
#   score(30) = 1/(60+2+1)               = 1/63 ~ 0.015873
#
# Correct order: 10 > 20 > 30
# top_n=2 must return [10, 20]
# top_n=3 must return [10, 20, 30]

dense_results  = [(10, 0.9, {}), (20, 0.8, {}), (30, 0.7, {})]
sparse_results = [(10, 5.0),    (20, 3.0)]

fused_top2 = rrf_fuse(dense_results, sparse_results, k=60, top_n=2)
fused_top3 = rrf_fuse(dense_results, sparse_results, k=60, top_n=3)

check("RRF top-2 first  = chunk 10 (highest combined rank)", fused_top2[0] == 10)
check("RRF top-2 second = chunk 20",                         fused_top2[1] == 20)
check("RRF top-2 length = 2",                                len(fused_top2) == 2)
check("RRF top-3 order  = [10, 20, 30]",                     fused_top3 == [10, 20, 30])

# Edge: only-dense, no sparse
dense_only = rrf_fuse([(5, 0.9, {}), (7, 0.8, {})], [], k=60, top_n=2)
check("RRF dense-only:  top=[5,7]", dense_only == [5, 7])

# Edge: only-sparse, no dense
sparse_only = rrf_fuse([], [(3, 5.0), (9, 2.0)], k=60, top_n=2)
check("RRF sparse-only: top=[3,9]", sparse_only == [3, 9])

# Edge: top_n larger than candidates => return all available
small_fuse = rrf_fuse([(1, 0.9, {})], [(2, 1.0)], k=60, top_n=10)
check("RRF top_n > candidates returns all", len(small_fuse) == 2)


# ─── TEST 8: dense_search filter_params path (mocked Qdrant) ────────────────────
print("\n=== TEST 8: dense_search filter_params (mocked Qdrant) ===")
from unittest.mock import MagicMock, patch
from retriever import dense_search

# Build a mock Qdrant result
mock_hit         = MagicMock()
mock_hit.id      = 42
mock_hit.score   = 0.91
mock_hit.payload = {"chunk_text": "EventID:4625 | Host:DC01", "event_type": "Failed Logon"}

mock_client          = MagicMock()
mock_client.query_points.return_value = MagicMock(points=[mock_hit])

# Patch embed_text so we don't need Ollama running
with patch("retriever.embed_text", return_value=[0.0] * 768):
    # 8a: No filter_params -> query_filter must be None
    results_no_filter = dense_search(mock_client, "failed login", top_k=5)
    call_kwargs_no   = mock_client.query_points.call_args[1]
    check("No filter_params -> query_filter=None in Qdrant call",
          call_kwargs_no.get("query_filter") is None)
    check("No filter_params -> returns list of tuples",
          isinstance(results_no_filter, list) and results_no_filter[0][0] == 42)

    # 8b: event_id filter -> Qdrant Filter object must be passed (not None)
    results_filtered = dense_search(
        mock_client, "failed login", top_k=5,
        filter_params={"event_id": "4625"}
    )
    call_kwargs_filt = mock_client.query_points.call_args[1]
    check("event_id filter -> query_filter is not None",
          call_kwargs_filt.get("query_filter") is not None)
    check("event_id filter -> still returns results",
          isinstance(results_filtered, list) and len(results_filtered) == 1)

    # 8c: source_file filter -> also produces a non-None filter
    dense_search(
        mock_client, "psexec activity", top_k=5,
        filter_params={"source_file": "psexec.evtx"}
    )
    call_kwargs_src = mock_client.query_points.call_args[1]
    check("source_file filter -> query_filter is not None",
          call_kwargs_src.get("query_filter") is not None)

    # 8d: Both filters -> still non-None and two conditions
    dense_search(
        mock_client, "psexec 4688", top_k=5,
        filter_params={"event_id": "4688", "source_file": "psexec.evtx"}
    )
    call_kwargs_both = mock_client.query_points.call_args[1]
    filt_obj = call_kwargs_both.get("query_filter")
    check("Both filters -> query_filter is not None",
          filt_obj is not None)
    # The Filter's must list should have 2 conditions
    check("Both filters -> 2 conditions in must list",
          len(filt_obj.must) == 2)


# ─── TEST 9: End-to-end run_query integration (mocked) ────────────────────
print("\n=== TEST 9: End-to-end run_query integration (mocked) ===")
from query import run_query
import query

mock_bm25 = MagicMock()
mock_corpus = []
mock_chunk_ids = []
mock_payloads = {
    10: {"source_file": "test.evtx", "timestamp": "2024", "hostname": "PC1", "event_type": "Logon", "chunk_text": "text1", "data": {"EventID": "4624"}}
}

test_chat_history = []

with patch("query.is_forensic_query_with_fallback", return_value=(True, "keyword")), \
     patch("query.dense_search", return_value=[(10, 0.9, {})]), \
     patch("query.sparse_search", return_value=[(10, 5.0)]), \
     patch("query.generate_answer_validated", return_value="Mocked Answer"), \
     patch("query.print"):  # silence output during test
    
    run_query("did anyone logon?", mock_client, mock_bm25, mock_corpus, mock_chunk_ids, mock_payloads, test_chat_history)
    
    check("chat_history length increased", len(test_chat_history) == 1)
    if test_chat_history:
        check("chat_history contains answer", test_chat_history[0]["answer"] == "Mocked Answer")
        check("chat_history contains sources", len(test_chat_history[0]["sources"]) == 1)
        check("chat_history source is correct", test_chat_history[0]["sources"][0]["source_file"] == "test.evtx")


# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  RESULTS:  {PASS} PASSED   {FAIL} FAILED")
print(f"{'='*50}\n")
sys.exit(0 if FAIL == 0 else 1)
