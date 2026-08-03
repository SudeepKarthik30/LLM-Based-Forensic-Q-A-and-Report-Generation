"""
retriever.py  —  Phase 2: Hybrid Retrieval (Dense + Sparse) + RRF Fusion + Context Builder
─────────────────────────────────────────────────────────────────────────────
This file is the "search engine" of the RAG pipeline.  Given a user question:

  1.  Dense search  — cosine-similarity lookup in Qdrant (semantic meaning)
  2.  Sparse search — BM25 keyword lookup (exact IPs, hashes, process names)
  3.  RRF fusion    — merge both ranked lists into one using Reciprocal Rank Fusion
  4.  Context build — assemble a numbered EVIDENCE block the LLM will cite

It also contains the forensic relevance gate that rejects off-topic questions
before wasting a round-trip to the LLM.

Changes from original
─────────────────────
• Forensic relevance gate tightened: now requires >= 2 keyword matches (was 1).
  Generic question words (when, who, was, how, did, found, where, which, what)
  removed from the keyword set — they trivially matched any question before.
• Context snippet limit raised from 300 → 400 chars.
  NOTE: this reduces the risk of truncating critical EventData fields but does
  NOT eliminate it for very long payloads.  Known limitation — documented
  intentionally in this file and in the report.
• All retrieval constants imported from config.py.
─────────────────────────────────────────────────────────────────────────────
"""

import re
import logging
from store import embed_text, tokenize, COLLECTION_NAME
from config import (
    DENSE_TOP_K, SPARSE_TOP_K, RRF_K, RRF_TOP_N, SNIPPET_CHARS,
    BM25_FALLBACK_THRESHOLD,
    CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MED_THRESHOLD,
)

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FORENSIC RELEVANCE GATE
# ─────────────────────────────────────────────────────────────────────────────

# A vocabulary of forensic / security / log-analysis terms.
#
# Design notes (post-review):
#   • Generic question words (who, what, when, where, how, was, did, found,
#     which) have been REMOVED.  They previously caused any question containing
#     those words to trivially pass the gate (e.g. "who was the first president"
#     matched "who" and "was").
#   • Gate now requires >= 2 matches (see is_forensic_query).
#   • This makes the gate a real semantic filter, not a cosmetic one.
FORENSIC_KEYWORDS = {
    # authentication & access
    "login", "logon", "logoff", "authentication", "credential", "password",
    "failed", "success", "account", "user", "admin", "privilege", "sudo",
    "access", "authorization", "session", "token", "kerberos", "ntlm",
    # process & file system
    "process", "executable", "binary", "script", "powershell", "cmd",
    "command", "shell", "wscript", "cscript", "registry", "service",
    "task", "scheduled", "dll", "exe", "bat", "ps1", "vbs",
    # network
    "network", "ip", "port", "tcp", "udp", "icmp", "http", "https",
    "dns", "firewall", "connection", "traffic", "packet", "payload",
    "c2", "beacon", "lateral", "movement", "exfil", "tunnel",
    # event log & forensics
    "event", "log", "evtx", "sysmon", "audit", "cleared", "artifact",
    "forensic", "investigation", "incident", "anomaly", "suspicious",
    "threat", "attack", "exploit", "malware", "ransomware", "backdoor",
    # specific tools / techniques
    "mimikatz", "metasploit", "psexec", "cobalt", "sliver", "meterpreter",
    "wmi", "dcom", "smb", "rdp", "ssh", "hash", "dump", "inject",
    "persistence", "escalation", "group", "domain", "dc", "security",
    # investigative but specific — NOT generic question words
    "detected", "occurred", "happened", "timestamp", "hostname",
    "eventid", "logfile", "pcap", "capture", "ioc", "indicator",
}


def is_forensic_query(question):
    """
    Quick pre-flight check: is this question forensics-relevant?

    Requires at LEAST 2 keyword matches from FORENSIC_KEYWORDS.
    Generic question words (who, what, when, how, was, etc.) have been
    removed from the keyword set so they no longer trivially pass any question.

    Input  : user question string
    Output : True if forensic-relevant, False if off-topic
    """
    tokens  = set(re.findall(r'[a-zA-Z]+', question.lower()))
    overlap = tokens & FORENSIC_KEYWORDS
    return len(overlap) >= 2


def is_forensic_query_with_fallback(question, bm25, corpus_tokens, chunk_ids):
    """
    Two-stage forensic relevance gate.

    Stage 1 — Keyword check (fast, O(1)):
        Runs is_forensic_query.  If it passes, return immediately.
        Most real analyst queries pass here.

    Stage 2 — BM25 score fallback (runs only if Stage 1 fails):
        Real analyst phrasing like "any brute force attempts?" can score 0
        keyword matches and be wrongly rejected.  We run a quick BM25 search;
        if the top score exceeds BM25_FALLBACK_THRESHOLD, the query has genuine
        forensic signal and we let it through.

    [WARNING]  BM25_FALLBACK_THRESHOLD in config.py is a placeholder calibrated at 2.0.
        After first successful ingest, run 10-15 sample queries, print the raw
        sparse_results[0][1] scores, and adjust the threshold to match your corpus.

    Inputs
    ------
    question     : user question string
    bm25         : BM25Okapi model from load_bm25()
    corpus_tokens: list of token lists (same order as BM25 corpus)
    chunk_ids    : list of chunk IDs (same order)

    Output : (allowed: bool, reason: str)
        reason is one of: 'keyword', 'bm25_fallback', 'rejected'
    """
    # Stage 1: fast keyword check
    if is_forensic_query(question):
        return True, "keyword"

    # Stage 2: BM25 fallback
    sparse_results = sparse_search(bm25, corpus_tokens, chunk_ids, question, top_k=5)
    if sparse_results and sparse_results[0][1] > BM25_FALLBACK_THRESHOLD:
        _log.info(
            "Gate: keyword check failed, BM25 fallback PASSED (score=%.3f)",
            sparse_results[0][1]
        )
        return True, "bm25_fallback"

    return False, "rejected"


# ─────────────────────────────────────────────────────────────────────────────
# IOC EXTRACTOR  (helps BM25 zero-in on specific indicators)
# ─────────────────────────────────────────────────────────────────────────────

def extract_ioc_terms(query):
    """
    Scans the question for concrete Indicators of Compromise (IOCs) so BM25 can
    do exact matching on them — these are the things semantic search often misses.

    What we look for:
      - IPv4 addresses  (e.g. "192.168.1.45")
      - MD5/SHA1/SHA256 hashes  (hex strings of length >= 32)
      - Executables / scripts   (anything ending in .exe .ps1 .bat .vbs .dll)

    Input  : question string
    Output : list of IOC strings found in the question
    """
    iocs = []

    # IPv4 — four dot-separated decimal octets
    iocs += re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', query)

    # Hashes — long hex strings (>= 32 chars = at least MD5 length)
    iocs += re.findall(r'\b[0-9a-fA-F]{32,}\b', query)

    # File names with known dangerous extensions
    iocs += re.findall(
        r'\b[\w\-]+\.(?:exe|ps1|bat|cmd|vbs|dll|sys|scr|hta)\b',
        query,
        re.IGNORECASE,
    )

    return iocs


# ─────────────────────────────────────────────────────────────────────────────
# DENSE RETRIEVAL  (Qdrant cosine similarity)
# ─────────────────────────────────────────────────────────────────────────────

def dense_search(client, question, top_k=None, filter_params=None):
    """
    Semantic vector search inside Qdrant.

    We embed the question with nomic-embed-text and find the `top_k` stored
    chunks whose embeddings are closest by cosine similarity.  This is great
    for catching semantically related events even when they don't share exact
    keywords (e.g. "lateral movement" matches events about PSExec or RDP).

    Payload indexes on data.EventID, hostname, event_type, and timestamp
    (created by ensure_collection in store.py) ensure that filter_params
    queries hit the index rather than full-scanning all payloads.

    Inputs
    ------
    client        : QdrantClient (passed in from query.py — one shared connection)
    question      : user question string
    top_k         : number of results to return from Qdrant (default: config value)
    filter_params : optional dict with keys:
                      "event_id"       → filter by exact EventID string
                      "source_file"    → filter by source filename

    Output : list of (chunk_id, similarity_score, payload_dict) tuples
    """
    if top_k is None:
        top_k = DENSE_TOP_K

    query_vector = embed_text(question)

    # Build an optional Qdrant payload filter
    qdrant_filter = None
    if filter_params:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = []
        if filter_params.get("event_id"):
            conditions.append(
                FieldCondition(key="data.EventID",
                               match=MatchValue(value=str(filter_params["event_id"])))
            )
        if filter_params.get("source_file"):
            conditions.append(
                FieldCondition(key="source_file",
                               match=MatchValue(value=filter_params["source_file"]))
            )
        if conditions:
            qdrant_filter = Filter(must=conditions)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    ).points

    return [(r.id, r.score, r.payload) for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE RETRIEVAL  (BM25 keyword matching)
# ─────────────────────────────────────────────────────────────────────────────

def sparse_search(bm25, corpus_tokens, chunk_ids, question, top_k=None):
    """
    BM25 (Best Match 25) keyword search over every chunk_text in the corpus.

    BM25 excels where dense search can struggle:
      - Exact IP addresses:  "192.168.1.45"
      - Exact process names: "mimikatz.exe"
      - Hash values:         "aabb1122ccdd…"
      - EventID numbers:     "4625"

    We also inject any IOCs we extracted from the question as extra query tokens
    to boost recall on those specific indicators.

    Inputs
    ------
    bm25          : BM25Okapi index (from load_bm25 / build_and_save_bm25)
    corpus_tokens : list of token lists (one per chunk, same order as chunk_ids)
    chunk_ids     : list of integer chunk IDs matching corpus position
    question      : user question string
    top_k         : max results to return (default: config value)

    Output : list of (chunk_id, bm25_score) tuples, sorted by score desc
    """
    if top_k is None:
        top_k = SPARSE_TOP_K

    # Start with normal word tokens from the question
    query_tokens = tokenize(question)

    # Append any IOC tokens found — helps exact-match on IPs / hashes / exes
    for ioc in extract_ioc_terms(question):
        query_tokens += tokenize(ioc)

    scores = bm25.get_scores(query_tokens)

    # Sort indices by score descending, keep only non-zero matches
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = [i for i in top_indices[:top_k] if scores[i] > 0.0]

    return [(chunk_ids[i], float(scores[i])) for i in top_indices]


# ─────────────────────────────────────────────────────────────────────────────
# RRF FUSION  (Reciprocal Rank Fusion)
# ─────────────────────────────────────────────────────────────────────────────

def rrf_fuse(dense_results, sparse_results, k=None, top_n=None):
    """
    Merges dense and sparse result lists using Reciprocal Rank Fusion.

    RRF score formula for each document:
        score(d) = Σ  1 / (k + rank_in_list)
    where the sum is over every list the document appears in.

    A document that ranks #1 in dense AND #1 in sparse gets the highest
    combined score.  Documents appearing in only one list still get a partial
    score, so nothing is completely discarded.

    The k=60 constant is the standard RRF default — it dampens the impact of
    very high positions vs very low positions, making the merge robust.

    Inputs
    ------
    dense_results  : list of (chunk_id, score, payload) from dense_search
    sparse_results : list of (chunk_id, score)          from sparse_search
    k              : RRF constant (default: config.RRF_K = 60)
    top_n          : how many final results to return (default: config.RRF_TOP_N)

    Output : list of chunk_ids in fused ranking order (best first)
    """
    if k is None:
        k = RRF_K
    if top_n is None:
        top_n = RRF_TOP_N

    fused_scores = {}

    # Contribution from the dense (semantic) ranking
    for rank, (chunk_id, _score, _payload) in enumerate(dense_results):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    # Contribution from the sparse (keyword) ranking
    for rank, (chunk_id, _score) in enumerate(sparse_results):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    # Sort by combined RRF score and return top_n chunk IDs
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, _ in ranked[:top_n]]


def compute_retrieval_confidence(dense_results, sources):
    """
    Derives a grounded confidence level from retrieval signals.

    This replaces (or constrains) the LLM's self-reported CONFIDENCE:
    HIGH|MEDIUM|LOW with a value computed from actual retrieval evidence.

    Signals used
    ------------
    top_dense_score : cosine similarity of the top Qdrant hit (0.0–1.0)
    source_count    : how many unique sources the LLM has to cite

    Thresholds (CALIBRATE AFTER INGEST — see config.py)
    -----------
    HIGH   : top score ≥ CONFIDENCE_HIGH_THRESHOLD  AND  sources ≥ 3
    MEDIUM : top score ≥ CONFIDENCE_MED_THRESHOLD   OR   sources ≥ 2
    LOW    : below both thresholds

    Inputs
    ------
    dense_results : list of (chunk_id, score, payload) tuples from dense_search
    sources       : list of source dicts from build_context_and_sources

    Output : (level: str, signals: dict)
        level   : 'HIGH' | 'MEDIUM' | 'LOW'
        signals : {'top_dense_score': float, 'source_count': int}
    """
    top_dense_score = dense_results[0][1] if dense_results else 0.0
    source_count    = len(sources)

    signals = {
        "top_dense_score": round(top_dense_score, 4),
        "source_count":    source_count,
    }

    if top_dense_score >= CONFIDENCE_HIGH_THRESHOLD and source_count >= 3:
        level = "HIGH"
    elif top_dense_score >= CONFIDENCE_MED_THRESHOLD or source_count >= 2:
        level = "MEDIUM"
    else:
        level = "LOW"

    return level, signals


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDER  (assembles the EVIDENCE block for the LLM)
# ─────────────────────────────────────────────────────────────────────────────

def build_context_and_sources(fused_ids, payloads_map):
    """
    Turns the top fused chunk IDs into a numbered EVIDENCE block for the LLM.

    Prompt injection guard
    ----------------------
    Every evidence snippet is raw attacker-controlled data (process names,
    CommandLine values, registry entries).  An attacker can name a process
    'Ignore_previous_instructions.exe' or embed EVIDENCE marker strings.

    Protection applied here:
      1. Snippet is sanitised: the delimiter strings <<<EVIDENCE_START>>> and
         <<<EVIDENCE_END>>> are stripped from the snippet content so an attacker
         cannot break out of the evidence block early.
      2. Each snippet is wrapped between explicit <<<EVIDENCE_START>>> and
         <<<EVIDENCE_END>>> markers.
      3. The system prompt (llm.py) instructs the LLM to treat content inside
         these markers as opaque data, never as instructions.

    Snippet truncation
    ------------------
    chunk_text is clipped to SNIPPET_CHARS (400 chars from config.py).
    Full chunk_text is always in the Qdrant payload for post-hoc inspection.

    Inputs
    ------
    fused_ids   : ordered list of chunk_ids (from rrf_fuse)
    payloads_map: dict  chunk_id → Qdrant payload dict

    Output : (context_string, sources_list)
    """
    context_lines = ["EVIDENCE:"]
    sources       = []

    for i, chunk_id in enumerate(fused_ids, start=1):
        payload = payloads_map.get(chunk_id)
        if payload is None:
            continue

        source_file = payload.get("source_file", "unknown")
        timestamp   = payload.get("timestamp",   "unknown")
        hostname    = payload.get("hostname",     "unknown")
        event_type  = payload.get("event_type",   "unknown")
        chunk_text  = payload.get("chunk_text",   "")
        data        = payload.get("data", {})
        event_id    = data.get("EventID", "")

        # Build one-line citation header
        header = f"[{i}] {source_file}"
        if event_id:
            header += f" | EventID {event_id}"
        header += f" | {timestamp} | Host: {hostname}"

        # ── Injection guard: sanitise before wrapping ───────────────────────────
        # Strip delimiter strings from attacker-controlled content first.
        # Prevents early breakout if CommandLine/ProcessName contains the
        # literal delimiter (e.g. attacker names process <<<EVIDENCE_END>>>).
        snippet = (
            chunk_text[:SNIPPET_CHARS]
            .replace("<<<EVIDENCE_START>>>", "")
            .replace("<<<EVIDENCE_END>>>",   "")
            .replace("\n", " ")
        )

        context_lines.append(header)
        context_lines.append("    <<<EVIDENCE_START>>>")
        context_lines.append(f'    "{event_type}: {snippet}"')
        context_lines.append("    <<<EVIDENCE_END>>>")
        context_lines.append("")   # blank line between entries

        sources.append({
            "num":         i,
            "source_file": source_file,
            "event_id":    event_id,
            "timestamp":   timestamp,
            "hostname":    hostname,
            "event_type":  event_type,
        })

    return "\n".join(context_lines), sources
