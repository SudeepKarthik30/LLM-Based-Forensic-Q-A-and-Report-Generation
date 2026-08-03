"""
app.py — Aeternus Forensis :: Streamlit UI
Wires directly into the existing backend: config.py, store.py, retriever.py,
llm.py, report_generator.py. No mocked data, no new pipeline logic — this
file only orchestrates real function calls and renders them.

Run:
    streamlit run app.py

Logos: place control.png, info.png, ingest.png, report.png, system_status.png
(and their _bg_black variants if you want a dark-mode swap later) in ./logos

Round 2 notes
-------------
- The forensic pipeline now runs in a background thread (see _pipeline_worker
  / start_query / poll_active_job). Streamlit cancels an in-progress script
  run the instant you interact with anything else — including clicking a
  different section tab — so a query that was still inline inside
  render_investigate_tab() would simply die if you navigated away. Moving
  the actual retrieval + LLM call into a plain Python thread means it keeps
  running regardless of which tab you're looking at; poll_active_job() picks
  up the finished result on the next rerun, from any tab.
- The chat input is centered on the page before the first message (Streamlit
  can't reposition st.chat_input itself — it's always pinned to the bottom
  of the viewport — so the "centered" state uses a plain st.form styled to
  match, and control hands off to the real st.chat_input once a
  conversation exists).
"""

import os
import time
import re
import sys
import html
import uuid
import pickle
import tempfile
import shutil
import threading
from datetime import datetime

import streamlit as st
from rank_bm25 import BM25Okapi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    COLLECTION_NAME, BM25_CACHE, DENSE_TOP_K, SPARSE_TOP_K, RRF_K, RRF_TOP_N,
    LLM_MODEL, EMBED_MODEL, LLM_TEMPERATURE, OUTPUT_DIR,
)
from store import (
    get_qdrant_client, load_bm25, build_chunk_text, embed_batch,
    ensure_collection, upsert_to_qdrant, tokenize,
)
from retriever import (
    is_forensic_query_with_fallback, dense_search, sparse_search, rrf_fuse,
    build_context_and_sources, compute_retrieval_confidence,
)
from llm import generate_answer_validated
import llm as llm_module  # used to patch LLM_TEMPERATURE at runtime from Settings
from report_generator import generate_report
from query import load_all_payloads_from_qdrant, _log_query
from parsers import parse_all_files
import session_store as sess_store  # persisted chat history — survives restarts/refresh

LOGO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logos")

# Sidebar (Session Info / Filters / Investigation History) is only relevant
# while in the Investigate section — collapsed everywhere else. Read directly
# from session_state here since this runs before the ss=st.session_state
# block below, but session_state itself already persists across reruns.
_active_section_now = st.session_state.get("active_section", "investigate")
st.set_page_config(
    page_title="Aeternus Forensis", page_icon="🛡", layout="wide",
    initial_sidebar_state="expanded" if _active_section_now == "investigate" else "collapsed",
)

with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "style.css"),
    encoding="utf-8",
) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Search-bar redesign overrides (GPT / Gemini / Claude style pill input) ──
# Kept separate from assets/style.css on purpose: this is new, still-settling
# styling for Ask #3, easy to tweak or rip out without touching the base sheet.
st.markdown(
    """<style>
    /* Pre-chat centered box (render_centered_input) */
    .af-centered-wrap { max-width: 700px; margin: 64px auto 12px auto; }
    .af-centered-heading {
        text-align: center; font-size: 1.9rem; font-weight: 600;
        margin-bottom: 26px; color: var(--textColor, #E9E6DC);
        letter-spacing: .2px;
    }
    .af-centered-wrap div[data-testid="stForm"] {
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 30px !important;
        background: var(--secondaryBackgroundColor, #121613) !important;
        padding: 6px 8px 6px 22px !important;
        box-shadow: 0 6px 24px rgba(0,0,0,0.35);
    }
    .af-centered-wrap div[data-testid="stForm"] input[type="text"] {
        background: transparent !important; border: none !important;
        font-size: 1.02rem !important; box-shadow: none !important;
    }
    .af-centered-wrap div[data-testid="stForm"] button {
        border-radius: 50% !important; width: 42px !important; height: 42px !important;
        padding: 0 !important; min-height: 42px !important;
    }
    .af-centered-examples { max-width: 700px; margin: 0 auto; }
    .af-centered-examples button {
        border-radius: 18px !important; font-size: .82rem !important;
    }

    /* Bottom-pinned chat_input once a conversation has started — styled to
       match the GPT/Claude/Gemini composer look as closely as a native
       Streamlit widget allows (can't add a custom "+" icon inside this
       specific widget — flagged separately in chat). */
    div[data-testid="stChatInput"] {
        max-width: 760px; margin: 0 auto;
    }
    div[data-testid="stChatInput"] > div {
        border-radius: 28px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        background: var(--secondaryBackgroundColor, #121613) !important;
        overflow: hidden !important;
        transition: border-color .15s ease, box-shadow .15s ease;
    }
    div[data-testid="stChatInput"]:focus-within > div {
        border-color: var(--primaryColor, #3FA88C) !important;
        box-shadow: 0 0 0 3px rgba(63,168,140,0.18) !important;
    }
    div[data-testid="stChatInput"] textarea {
        resize: none !important;
        border-radius: 28px !important;
        box-shadow: none !important;
        padding: 12px 16px !important;
        font-size: .96rem !important;
    }
    div[data-testid="stChatInput"] button {
        border-radius: 50% !important;
        background: var(--primaryColor, #3FA88C) !important;
    }

    /* Chat thread — user turn as a right-aligned bubble (Claude-style).
       Assistant content deliberately keeps the existing .af-finding card
       styling untouched, per explicit request — only the layout around it
       changes to read as a conversation instead of a stacked Q/A list. */
    .af-chat-turn { margin: 22px 0 4px; }
    .af-user-row { display: flex; justify-content: flex-end; margin-bottom: 14px; }
    .af-user-bubble {
        max-width: 72%;
        background: var(--secondaryBackgroundColor, #121613);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px 20px 4px 20px;
        padding: 10px 18px;
        color: var(--textColor, #E9E6DC);
        font-size: .96rem;
        line-height: 1.5;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    /* Section nav — one column per button now (not split icon+button),
       force single-line labels so they never wrap and misalign. */
    .af-nav-icon-wrap { text-align: center; margin-bottom: 4px; }
    .af-nav-icon { width: 16px; height: 16px; opacity: .85; }
    .st-key-af_nav_row button {
        white-space: nowrap !important;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: .82rem !important;
        letter-spacing: .03em;
        padding-top: 4px !important;
        padding-bottom: 4px !important;
        min-height: 34px !important;
    }
    .st-key-af_nav_row { margin-top: -4px; }

    /* Header: shrunk + pinned so it stays put while chat content scrolls
       beneath it, and frees up vertical room for the conversation itself.
       NOTE: position:sticky was silently failing before this fix because
       Streamlit's own internal wrapper divs (stAppViewContainer / stMain)
       run their OWN internal scroll context — sticky only works within the
       nearest scrolling ancestor, and that ancestor was clipping it. Forcing
       those wrappers back to normal (non-scrolling) flow lets the real page
       scroll instead, which is what sticky actually needs to work here. */
    div[data-testid="stAppViewContainer"],
    section[data-testid="stMain"],
    div[data-testid="stMain"],
    .main {
        overflow: visible !important;
    }
    .block-container { padding-top: 1.2rem !important; }
    .st-key-af_top_header {
        position: sticky; top: 0; z-index: 999;
        background: var(--backgroundColor, #0A0D0C);
        padding-bottom: 2px;
    }
    .af-page-title {
        font-size: 1.15rem; font-weight: 700;
        color: var(--textColor, #E9E6DC);
        margin-top: 4px;
    }
    </style>""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND PIPELINE — plain-thread job queue, independent of which tab is open
# ─────────────────────────────────────────────────────────────────────────────
# _JOBS holds ALL in-flight/finished jobs, keyed by job_id. The worker thread
# only touches this dict and plain backend functions — it never calls any
# st.* function, so it's safe to run off the Streamlit script thread.
#
# IMPORTANT: this must be created via st.cache_resource, not a bare module-
# level "_JOBS = {}". Streamlit re-executes the ENTIRE script top-to-bottom
# on every rerun (every button click, every st.rerun()) — a plain assignment
# here would get wiped back to an empty dict on the very next rerun, which
# is exactly what was happening: start_query() populated _JOBS and kicked
# off a background thread, then immediately called st.rerun(), which
# re-ran this file from the top and reset _JOBS to {} before the thread
# had a chance to write its result — so the answer vanished into a KeyError
# inside the thread with nothing shown to the user. cache_resource returns
# the SAME dict/lock across reruns (and across sessions, which is fine
# here since job_ids are unique) instead of recreating them.

@st.cache_resource(show_spinner=False)
def _get_job_store():
    return {}, threading.Lock()

_JOBS, _JOBS_LOCK = _get_job_store()


def _set_stage(job_id, stage):
    with _JOBS_LOCK:
        if job_id in _JOBS:
            _JOBS[job_id]["stage"] = stage


def _pipeline_worker(job_id, question, filter_params, backend, chat_history_snapshot, cfg):
    try:
        client, bm25 = backend["client"], backend["bm25"]
        corpus_tokens, chunk_ids, payloads = backend["corpus_tokens"], backend["chunk_ids"], backend["payloads"]

        _set_stage(job_id, "Checking forensic relevance")
        allowed, gate_reason = is_forensic_query_with_fallback(question, bm25, corpus_tokens, chunk_ids)
        if not allowed:
            with _JOBS_LOCK:
                _JOBS[job_id].update(status="rejected", stage="Rejected — not forensic-relevant")
            return

        _set_stage(job_id, "Dense retrieval — Qdrant cosine search")
        dense_results = dense_search(client, question, top_k=cfg["dense_top_k"], filter_params=filter_params)

        _set_stage(job_id, f"Sparse retrieval — BM25 keyword match ({len(dense_results)} dense hits)")
        sparse_results = sparse_search(bm25, corpus_tokens, chunk_ids, question, top_k=cfg["sparse_top_k"])

        _set_stage(job_id, f"Fusing dense + sparse results (RRF, k={cfg['rrf_k']})")
        fused_ids = rrf_fuse(dense_results, sparse_results, k=cfg["rrf_k"], top_n=cfg["rrf_top_n"])

        _set_stage(job_id, "Assembling evidence context")
        context, sources = build_context_and_sources(fused_ids, payloads)

        if not sources:
            with _JOBS_LOCK:
                _JOBS[job_id].update(status="empty", stage="No matching evidence found")
            return

        evidence_texts = [payloads[cid]["chunk_text"] for cid in fused_ids if cid in payloads]
        retrieval_confidence, conf_signals = compute_retrieval_confidence(dense_results, sources)

        _set_stage(job_id, f"Generating answer — {cfg['llm_model']} (temp={cfg['llm_temp']})")
        llm_module.LLM_TEMPERATURE = cfg["llm_temp"]
        t0 = time.time()
        answer = generate_answer_validated(
            question, context, sources, max_retries=1,
            chat_history=chat_history_snapshot,
            retrieval_confidence=retrieval_confidence,
        )
        elapsed = time.time() - t0

        with _JOBS_LOCK:
            _JOBS[job_id].update(
                status="done",
                stage=f"Done in {elapsed:.1f}s · {len(sources)} sources · {retrieval_confidence} confidence",
                result={
                    "question": question, "answer": answer, "sources": sources,
                    "retrieval_confidence": retrieval_confidence,
                    "evidence_texts": evidence_texts,
                    "gate_reason": gate_reason, "conf_signals": conf_signals,
                },
            )
    except Exception as exc:
        with _JOBS_LOCK:
            _JOBS[job_id].update(status="error", stage="Pipeline failed", error=str(exc))


def start_query(question, backend, filter_params):
    """Kicks off the pipeline in a background thread and returns immediately."""
    job_id = f"job-{int(time.time() * 1000)}"
    with _JOBS_LOCK:
        _JOBS[job_id] = {"status": "running", "stage": "Starting…", "question": question}
    cfg = dict(
        dense_top_k=ss.cfg_dense_top_k, sparse_top_k=ss.cfg_sparse_top_k,
        rrf_k=ss.cfg_rrf_k, rrf_top_n=ss.cfg_rrf_top_n,
        llm_temp=ss.cfg_llm_temp, llm_model=LLM_MODEL,
    )
    t = threading.Thread(
        target=_pipeline_worker,
        args=(job_id, question, filter_params, backend, list(ss.chat_history), cfg),
        daemon=True,
    )
    t.start()
    ss.active_job_id = job_id


def poll_active_job():
    """Called at the top of every rerun, from every tab. If the background
    job finished (or failed) since the last time we looked, it gets folded
    into chat_history here — this is what makes the answer survive a
    section switch instead of vanishing."""
    job_id = ss.get("active_job_id")
    if not job_id:
        return
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            ss.active_job_id = None
            return
        status = job["status"]
        if status == "done":
            r = job["result"]
            del _JOBS[job_id]
        elif status in ("rejected", "empty", "error"):
            ss.job_notice = dict(job)
            del _JOBS[job_id]
            ss.active_job_id = None
            return
        else:
            return  # still running — nothing to fold in yet

    _log_query(r["question"], r["answer"], r["sources"],
               gate_reason=r["gate_reason"], conf_signals=r["conf_signals"])
    ss.chat_history.append({
        "question": r["question"], "answer": r["answer"], "sources": r["sources"],
        "retrieval_confidence": r["retrieval_confidence"],
    })
    ss.evidence_texts_by_turn.append(r["evidence_texts"])
    ss.active_job_id = None

    # Save this investigation to disk now that it has at least one real turn —
    # this is what makes it show up in the "Investigation History" sidebar and
    # survive a server restart / page refresh.
    sess_store.save_session(ss.session_id, ss.active_case_id,
                             ss.chat_history, ss.evidence_texts_by_turn)


def check_ollama():
    import ollama
    try:
        ollama.embeddings(model=EMBED_MODEL, prompt="health-check")
        return True
    except Exception:
        return False


def logo_path(name):
    path = os.path.join(LOGO_DIR, f"{name}.png")
    return path if os.path.exists(path) else None


def sidebar_head(name, title):
    """Icon + title as one atomic unit (columns), not a div opened/closed
    across separate st.* calls — that split was what made section labels
    and captions render invisible before."""
    p = logo_path(name)
    if p:
        c1, c2 = st.columns([1, 6], gap="small")
        with c1:
            st.image(p, width=18)
        with c2:
            st.markdown(f'<div class="af-sb-title" style="margin-top:2px;">{title}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="af-sb-head"><span class="af-sb-title">{title}</span></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND BOOTSTRAP (cached — runs once per server process)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_backend():
    """Connects to the real Qdrant + BM25 + payload cache. Returns None fields
    (not an exception) on failure so the UI can show a precise error state."""
    result = {"client": None, "bm25": None, "corpus_tokens": None,
              "chunk_ids": None, "payloads": None, "vec_count": 0, "error": None}
    try:
        client = get_qdrant_client()
        client.get_collections()  # raises if unreachable
        result["client"] = client
        result["vec_count"] = client.count(collection_name=COLLECTION_NAME).count
    except Exception as exc:
        result["error"] = f"Qdrant unreachable: {exc}. Start it with: docker run -p 6333:6333 qdrant/qdrant"
        return result

    try:
        bm25, corpus_tokens, chunk_ids = load_bm25(BM25_CACHE)
        result["bm25"], result["corpus_tokens"], result["chunk_ids"] = bm25, corpus_tokens, chunk_ids
    except Exception as exc:
        result["error"] = f"BM25 cache missing/unreadable at {BM25_CACHE}: {exc}. Run ingest.py first."
        return result

    try:
        result["payloads"] = load_all_payloads_from_qdrant(client)
    except Exception as exc:
        result["error"] = f"Could not load payloads from Qdrant: {exc}"
        return result

    # Case-ID inventory — the original CLI ingest.py never tagged a case_id,
    # so any data loaded that way shows up here as "untagged" until someone
    # assigns one via the Ingest tab. This is what replaces the old permanent
    # "No case ingested yet this session" message with real info about what's
    # actually sitting in the collection.
    payloads = result["payloads"] or {}
    result["case_ids"] = sorted({p.get("case_id") for p in payloads.values() if p.get("case_id")})
    result["untagged_count"] = sum(1 for p in payloads.values() if not p.get("case_id"))

    return result


def tag_untagged_events(client, payloads, case_id):
    """
    Retroactively assigns a case_id to every event currently in Qdrant that
    doesn't have one yet (i.e. data ingested via the original CLI ingest.py,
    before case tracking existed). Updates both Qdrant's stored payload and
    the in-memory payloads cache so the UI reflects it immediately.

    Returns the number of events tagged.
    """
    ids = [cid for cid, p in payloads.items() if not p.get("case_id")]
    if not ids:
        return 0
    client.set_payload(collection_name=COLLECTION_NAME, payload={"case_id": case_id}, points=ids)
    for cid in ids:
        payloads[cid]["case_id"] = case_id
    return len(ids)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

ss = st.session_state
ss.setdefault("page", "landing")
ss.setdefault("active_section", "investigate")
ss.setdefault("session_id", uuid.uuid4().hex[:8])  # identity for THIS chat — fresh
                                                    # on every restart/refresh, persisted
                                                    # to disk once its first turn lands
ss.setdefault("chat_history", [])           # [{question, answer, sources}]
ss.setdefault("evidence_texts_by_turn", []) # parallel chunk_text lists per turn
ss.setdefault("filter_event_id", "")
ss.setdefault("filter_source_file", "")
ss.setdefault("active_job_id", None)        # background pipeline job, if any
ss.setdefault("job_notice", None)           # one-shot rejected/empty/error message
ss.setdefault("draft_case_id", f"case-{datetime.now().strftime('%Y%m%d-%H%M')}")
ss.setdefault("active_case_id", None)       # most recently ingested case this session
# adjustable retrieval / generation config (defaults = config.py values)
ss.setdefault("cfg_dense_top_k", DENSE_TOP_K)
ss.setdefault("cfg_sparse_top_k", SPARSE_TOP_K)
ss.setdefault("cfg_rrf_k", RRF_K)
ss.setdefault("cfg_rrf_top_n", RRF_TOP_N)
ss.setdefault("cfg_llm_temp", LLM_TEMPERATURE)


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER PARSING (FINDING / EVIDENCE / ANSWER / CONFIDENCE contract from llm.py)
# ─────────────────────────────────────────────────────────────────────────────

_FIND_RE = re.compile(r"FINDING:\s*(.+?)(?=\n[A-Z]+:|\Z)", re.DOTALL)
_ANS_RE  = re.compile(r"ANSWER:\s*(.+?)(?=\nCONFIDENCE:|\Z)", re.DOTALL)
_CONF_RE = re.compile(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", re.IGNORECASE)
_CITE_RE = re.compile(r"\[Source\s+(\d+)(\s*—\s*INVALID CITATION)?\]", re.IGNORECASE)


def linkify_citations(text):
    def _sub(m):
        if m.group(2):
            return f'<span class="af-cite-bad">Source {m.group(1)} · invalid</span>'
        return f'<span class="af-cite">Source {m.group(1)}</span>'
    return _CITE_RE.sub(_sub, text)


def render_answer(answer_text, retrieval_confidence):
    if answer_text.startswith("IRRELEVANT QUERY"):
        st.warning(answer_text)
        return
    if answer_text.startswith("INSUFFICIENT EVIDENCE") or "INSUFFICIENT EVIDENCE" in answer_text[:40]:
        st.info(answer_text)
        return

    warning_block = ""
    if answer_text.startswith("[WARNING]  CITATION WARNING"):
        parts = answer_text.split("─" * 72, 1)
        warning_block = parts[0].strip()
        answer_text = parts[1].strip() if len(parts) > 1 else answer_text

    finding_m = _FIND_RE.search(answer_text)
    answer_m  = _ANS_RE.search(answer_text)
    conf_m    = _CONF_RE.search(answer_text)

    finding = finding_m.group(1).strip() if finding_m else ""
    body    = answer_m.group(1).strip() if answer_m else answer_text
    conf    = (conf_m.group(1).upper() if conf_m else retrieval_confidence) or "LOW"

    badge_class = {"HIGH": "ok", "MEDIUM": "warn", "LOW": "bad"}.get(conf, "bad")
    badge_text = f"CONFIDENCE: {conf}"
    if retrieval_confidence and retrieval_confidence not in ("UNKNOWN", conf):
        badge_text += f" (retrieval: {retrieval_confidence})"

    if warning_block:
        st.error(warning_block)

    st.markdown(
        f"""<div class="af-finding">
        <div style="display:flex;justify-content:space-between;align-items:start;gap:12px;">
          <h4>{finding or "Analysis result"}</h4>
          <span class="af-badge {badge_class}">{badge_text}</span>
        </div>
        <div>{linkify_citations(body)}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_live_status(stage_text):
    st.markdown(
        f'<div class="af-live-status"><span class="af-pulse-dot"></span><span>{stage_text}</span></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# INGEST PIPELINE (new — reuses parsers.py / store.py functions as-is)
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_EXT = {".evtx", ".pcap", ".pcapng", ".csv", ".log"}


def run_ingest(uploaded_files, case_id, backend):
    bad = [f.name for f in uploaded_files if os.path.splitext(f.name)[1].lower() not in ALLOWED_EXT]
    if bad:
        st.error(f"Unsupported file type(s): {', '.join(bad)}. Allowed: {', '.join(sorted(ALLOWED_EXT))}")
        return

    tmp_dir = tempfile.mkdtemp(prefix="af_ingest_")
    out_dir = os.path.join(OUTPUT_DIR, f"ui_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    try:
        for f in uploaded_files:
            with open(os.path.join(tmp_dir, f.name), "wb") as out:
                out.write(f.getbuffer())

        with st.status(f"Ingesting into `{COLLECTION_NAME}` (case: {case_id})...", expanded=True) as status:
            st.write("① Parsing raw files")
            records = parse_all_files(tmp_dir, out_dir)
            if not records:
                status.update(label="No events parsed", state="error")
                st.error("No events could be parsed from the uploaded file(s). Check they aren't corrupted/empty.")
                return

            for r in records:
                r["case_id"] = case_id

            st.write(f"② Parsed {len(records):,} events — building chunks")
            start_id = backend["vec_count"]  # offset so new point IDs never collide with existing ones
            chunks = [
                {"chunk_id": start_id + i, "chunk_text": build_chunk_text(e), "payload": e}
                for i, e in enumerate(records)
            ]

            st.write("③ Generating embeddings (nomic-embed-text via Ollama)")
            embeddings = embed_batch([c["chunk_text"] for c in chunks])
            good = sum(1 for e in embeddings if e is not None)
            if good < len(chunks):
                st.warning(f"{len(chunks) - good} chunk(s) failed embedding and were skipped (still BM25-searchable).")

            st.write(f"④ Storing {good:,} vectors in Qdrant")
            ensure_collection(backend["client"], force_recreate=False)
            upsert_to_qdrant(backend["client"], chunks, embeddings)

            st.write("⑤ Merging into BM25 keyword index")
            new_tokens = [tokenize(c["chunk_text"]) for c in chunks]
            new_ids = [c["chunk_id"] for c in chunks]
            combined_tokens = (backend["corpus_tokens"] or []) + new_tokens
            combined_ids = (backend["chunk_ids"] or []) + new_ids
            new_bm25 = BM25Okapi(combined_tokens)
            with open(BM25_CACHE, "wb") as f:
                pickle.dump({"corpus_tokens": combined_tokens, "chunk_ids": combined_ids}, f)

            # update the live in-memory backend cache so it's queryable immediately,
            # without needing to restart the Streamlit server
            backend["bm25"] = new_bm25
            backend["corpus_tokens"] = combined_tokens
            backend["chunk_ids"] = combined_ids
            for c, v in zip(chunks, embeddings):
                if v is not None:
                    payload = dict(c["payload"])
                    payload["chunk_text"] = c["chunk_text"]
                    payload["chunk_id"] = c["chunk_id"]
                    backend["payloads"][c["chunk_id"]] = payload
            backend["vec_count"] = backend["client"].count(collection_name=COLLECTION_NAME).count

            status.update(
                label=f"Done — {good:,}/{len(records):,} events embedded and stored (case: {case_id})",
                state="complete",
            )

        ss.active_case_id = case_id
        ss.draft_case_id = f"case-{datetime.now().strftime('%Y%m%d-%H%M')}"  # fresh default for next ingest
        st.success(f"Ingested {good:,} new events into case `{case_id}`. Corpus is now {backend['vec_count']:,} events total — ready to query.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────

def render_landing(backend, ollama_ok):
    st.markdown(
        """<div class="af-nav">
            <div class="af-brand">AETERNUS FORÊNSIS</div>
            <div class="af-navlinks"><span>Capabilities</span><span>Research</span><span>Security</span><span>Documentation</span></div>
            <div class="af-navver">v1.0.0-CAPSTONE</div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="af-hero">
            <h1>AI-POWERED FORENSIC INVESTIGATION,<br>GROUNDED IN EVIDENCE</h1>
            <p>Ingest forensic logs (EVTX, PCAP, CSV) with natural-language reasoning and
            citation-backed findings. Every claim traces to a real logged event.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    chips = ["HYBRID SEARCH", "CITATION-ENFORCED", "AIR-GAPPED OPS", "AUTO REPORT GEN"]
    chip_html = '<div style="display:flex;gap:16px;margin:28px auto;max-width:920px;">' + "".join(
        f'<div class="af-chip" style="flex:1;"><div class="t">{label}</div></div>'
        for label in chips
    ) + '</div>'
    st.markdown(chip_html, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("BEGIN INVESTIGATION", type="primary", use_container_width=True):
            ss.page = "investigate"
            st.rerun()

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""<div class="af-panel">
                <span class="af-label">Research Capstone Project</span>
                <p style="color:var(--ink-soft);margin:10px 0;">
                Local open-source models only — nomic-embed-text (embeddings) + {LLM_MODEL}
                (generation), both served through Ollama. No cloud calls, no external transfer.
                </p>
                <div class="af-mono" style="font-size:.72rem;color:var(--ink-soft);">
                🔒 FULLY LOCAL &nbsp;&nbsp; ↔ NO EXTERNAL XFER &nbsp;&nbsp; 📜 FULL AUDIT LOGS
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with right:
        if backend["error"]:
            status_badge = '<span class="af-badge bad">NEEDS SETUP</span>'
            status_lines = f'<div style="color:var(--ink-soft);font-size:.85rem;margin-top:10px;">{backend["error"]}</div>'
        else:
            status_badge = '<span class="af-badge ok">OPERATIONAL</span>'
            status_lines = (
                f'<div style="font-family:var(--mono);font-size:.8rem;color:var(--ink-soft);margin-top:10px;">'
                f'Qdrant: {backend["vec_count"]:,} vectors in `{COLLECTION_NAME}`<br>'
                f'Ollama: {"reachable" if ollama_ok else "NOT reachable — start Ollama"}'
                f'</div>'
            )
        st.markdown(
            f"""<div class="af-panel">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span class="af-label">System Status</span>{status_badge}
                </div>
                {status_lines}
            </div>""",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# INVESTIGATION PAGE
# ─────────────────────────────────────────────────────────────────────────────

def _start_new_chat():
    """Resets to a brand-new, empty investigation with its own session_id —
    exactly what happens automatically on a fresh restart/refresh, exposed
    here as an explicit button so the user can do it mid-session too."""
    ss.session_id = uuid.uuid4().hex[:8]
    ss.chat_history = []
    ss.evidence_texts_by_turn = []
    ss.active_job_id = None
    ss.job_notice = None


def _load_session(session_id):
    """Loads a previously persisted investigation back into session_state
    so the user can pick up exactly where they left off."""
    rec = sess_store.load_session(session_id)
    if not rec:
        st.sidebar.error("That session could not be found — it may have been deleted.")
        return
    ss.session_id = rec["session_id"]
    ss.chat_history = rec.get("chat_history", [])
    ss.evidence_texts_by_turn = rec.get("evidence_texts_by_turn", [])
    ss.active_case_id = rec.get("case_id")
    ss.active_job_id = None
    ss.job_notice = None
    ss.active_section = "investigate"


def render_sidebar(backend, ollama_ok):
    with st.sidebar:
        sidebar_head("info", "Investigation History")
        if st.button("＋ New Chat", use_container_width=True, key="af_new_chat_btn"):
            _start_new_chat()
            st.rerun()

        past_sessions = sess_store.list_sessions(limit=30)
        if not past_sessions:
            st.markdown(
                '<div style="font-size:.78rem;color:var(--ink-soft);padding:4px 0 8px;">'
                'No saved investigations yet — they appear here after your first '
                'question in a chat.</div>', unsafe_allow_html=True,
            )
        else:
            for s in past_sessions:
                is_current = s["session_id"] == ss.session_id
                label_case = s["case_id"] or "no case tag"
                btn_label = f'{"● " if is_current else ""}{s["first_question"]}'
                st.caption(f'{label_case} · {s["turn_count"]} turn(s) · {s["updated_at"]}')
                if st.button(
                    btn_label, key=f"hist_{s['session_id']}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                    disabled=is_current,
                ):
                    _load_session(s["session_id"])
                    st.rerun()

        st.divider()
        sidebar_head("info", "Session Info")
        st.markdown(
            f"""<div style="font-size:.85rem;color:var(--ink-soft);line-height:1.9;margin-bottom:6px;">
            Collection: <span class="af-cite">{COLLECTION_NAME}</span><br>
            Events indexed: <b style="color:var(--ink);">{backend['vec_count']:,}</b><br>
            Turns this session: <b style="color:var(--ink);">{len(ss.chat_history)}</b>
            </div>""",
            unsafe_allow_html=True,
        )

        st.divider()
        sidebar_head("ingest", "Filters")
        ss.filter_event_id = st.text_input("EventID", value=ss.filter_event_id, placeholder="e.g. 4625")
        source_files = sorted({p.get("source_file", "") for p in (backend["payloads"] or {}).values()})
        ss.filter_source_file = st.selectbox("Source file", [""] + source_files)
        st.markdown('<div style="font-size:.75rem;color:var(--ink-soft);">Applied to dense retrieval only, via Qdrant payload index.</div>', unsafe_allow_html=True)

        st.divider()
        sidebar_head("control", "Session Controls")
        if st.button("Start New Investigation", use_container_width=True, key="af_clear_session_btn"):
            _start_new_chat()
            st.rerun()

        st.divider()
        sidebar_head("system_status", "System Status")
        st.markdown(
            f"""<div style="font-family:var(--mono);font-size:.78rem;color:var(--ink-soft);line-height:1.9;">
            Ollama: <b style="color:{'var(--accent)' if ollama_ok else 'var(--danger)'};">{'UP' if ollama_ok else 'DOWN'}</b><br>
            LLM model: {LLM_MODEL}<br>
            Embed model: {EMBED_MODEL}<br>
            BM25 docs: {len(backend['chunk_ids'] or []):,}
            </div>""",
            unsafe_allow_html=True,
        )


def render_section_nav():
    """Custom section nav — one column per section (not a split icon+button
    pair) so labels get real room and never wrap to two lines."""
    sections = [
        ("investigate", "Investigate", "info"),
        ("overview", "Overview", "system_status"),
        ("ingest", "Ingest", "ingest"),
        ("settings", "Settings", "control"),
        ("report", "Report", "report"),
    ]
    with st.container(key="af_nav_row"):
        cols = st.columns(len(sections))
        for col, (key, label, logo) in zip(cols, sections):
            with col:
                p = logo_path(logo)
                icon_html = ""
                if p:
                    import base64
                    b64 = base64.b64encode(open(p, "rb").read()).decode()
                    icon_html = f'<img src="data:image/png;base64,{b64}" class="af-nav-icon">'
                active = ss.active_section == key
                st.markdown(
                    f'<div class="af-nav-icon-wrap">{icon_html}</div>' if icon_html else "",
                    unsafe_allow_html=True,
                )
                if st.button(label.upper(), key=f"nav_{key}",
                             type="primary" if active else "secondary",
                             use_container_width=True):
                    ss.active_section = key
                    st.rerun()
    st.divider()


def render_overview_tab(backend):
    """Session metrics, pulled out of Investigate into their own section so
    the chat can stay a clean chat and not a dashboard-plus-chat hybrid."""
    st.markdown(
        '<div class="af-panel"><span class="af-label">Session overview</span><br>'
        '<span style="color:var(--ink-soft);font-size:.88rem;">'
        'A snapshot of the current corpus and this investigation\'s activity so far.'
        '</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        all_sources = [s for t in ss.chat_history for s in t["sources"]]
        unique_files = len({s["source_file"] for s in all_sources})
        m1.metric("EVENTS INDEXED", f"{backend['vec_count']:,}")
        m2.metric("SESSION TURNS", len(ss.chat_history))
        m3.metric("SOURCES CITED", len(all_sources))
        m4.metric("FILES REFERENCED", unique_files)

    st.write("")
    case_ids = backend.get("case_ids") or []
    untagged = backend.get("untagged_count") or 0
    st.markdown(
        f'<div class="af-panel" style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span class="af-label">CASE(S) IN THIS COLLECTION</span>'
        f'<span class="af-cite">{", ".join(case_ids) if case_ids else "none tagged yet"}'
        f'{f" (+{untagged:,} untagged)" if untagged else ""}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_investigate(backend, ollama_ok):
    if ss.active_section == "investigate":
        render_sidebar(backend, ollama_ok)

    with st.container(key="af_top_header"):
        hdr_l, hdr_r = st.columns([3, 1])
        with hdr_l:
            st.markdown(
                '<div class="af-page-title">Forensic Investigation Assistant</div>',
                unsafe_allow_html=True,
            )
        with hdr_r:
            badge = '<span class="af-badge ok">READY</span>' if (not backend["error"] and ollama_ok) else '<span class="af-badge bad">NEEDS SETUP</span>'
            st.markdown(badge, unsafe_allow_html=True)

        if backend["error"]:
            st.error(backend["error"])
            return
        if not ollama_ok:
            st.error(f"Ollama not reachable. Start Ollama and ensure `{EMBED_MODEL}` / `{LLM_MODEL}` are pulled.")
            return

        render_section_nav()

    if ss.active_section == "investigate":
        render_investigate_tab(backend)
    elif ss.active_section == "overview":
        render_overview_tab(backend)
    elif ss.active_section == "ingest":
        render_ingest_tab(backend)
    elif ss.active_section == "settings":
        render_settings_tab(backend, ollama_ok)
    elif ss.active_section == "report":
        render_report_tab()


def _build_filter_params():
    filter_params = {}
    if ss.filter_event_id:
        filter_params["event_id"] = ss.filter_event_id
    if ss.filter_source_file:
        filter_params["source_file"] = ss.filter_source_file
    return filter_params or None


def _submit_question(question_text, backend):
    """
    Fires a query. Used as an on_click/on_submit CALLBACK, not called inline
    after rendering — callbacks run before the script repaints, so
    start_query() (which sets ss.active_job_id) has already happened by the
    time the page draws again. That's what stops the old bug where the
    centered search box and the live status/pinned chat box both flashed on
    screen at once for a frame: previously the centered form got drawn once
    more on the same run a click was processed, THEN start_query ran and
    forced a second rerun right after — a visible double-render. Doing the
    call from inside the callback means the very first paint after a click
    already reflects "query running", nothing else.
    """
    question_text = (question_text or "").strip()
    if not question_text:
        return
    start_query(question_text, backend, _build_filter_params())


def render_centered_input(backend):
    """Pre-chat search box, centered on the page — mirrors the Claude/
    Gemini/ChatGPT 'empty state' pattern. Hands off to the real
    st.chat_input (pinned bottom) the moment a conversation starts."""
    st.markdown('<div class="af-centered-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="af-centered-heading">What are we investigating?</div>', unsafe_allow_html=True)
    with st.form("af_centered_form", clear_on_submit=True, border=False):
        c_plus, c_input, c_send = st.columns([1, 10, 1])
        with c_plus:
            add_data = st.form_submit_button("＋", use_container_width=True)
        with c_input:
            st.text_input(
                "question", placeholder="Ask a forensic question…",
                label_visibility="collapsed", key="centered_q",
            )
        with c_send:
            st.form_submit_button(
                "↑", use_container_width=True,
                on_click=lambda: _submit_question(st.session_state.get("centered_q", ""), backend),
            )
    st.markdown('</div>', unsafe_allow_html=True)

    if add_data:
        ss.active_section = "ingest"
        st.rerun()

    st.markdown('<div class="af-centered-examples">', unsafe_allow_html=True)
    examples = [
        "Show me failed logon events",
        "Was there evidence of lateral movement?",
        "What process created suspicious network connections?",
    ]
    cols = st.columns(len(examples))
    for c, ex in zip(cols, examples):
        c.button(
            ex, use_container_width=True, key=f"ex_{hash(ex)}",
            on_click=_submit_question, args=(ex, backend),
        )
    st.markdown('</div>', unsafe_allow_html=True)


def render_investigate_tab(backend):
    st.markdown(
        '<div class="af-panel"><span class="af-label">Ask a forensic question</span><br>'
        '<span style="color:var(--ink-soft);font-size:.88rem;">'
        'Runs hybrid retrieval (dense + BM25) over the ingested case, then a citation-checked '
        'local LLM answer — every claim traces back to a real logged event.'
        '</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    chat_started = bool(ss.chat_history) or ss.active_job_id

    if not chat_started:
        render_centered_input(backend)
    else:
        for turn, ev_texts in zip(ss.chat_history, ss.evidence_texts_by_turn):
            st.markdown('<div class="af-chat-turn">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="af-user-row"><div class="af-user-bubble">'
                f'{html.escape(turn["question"])}</div></div>',
                unsafe_allow_html=True,
            )
            render_answer(turn["answer"], retrieval_confidence=turn.get("retrieval_confidence", "UNKNOWN"))
            with st.expander(f"Source Evidence ({len(turn['sources'])})"):
                for src, text in zip(turn["sources"], ev_texts):
                    st.markdown(
                        f'<div class="af-ev-head">'
                        f'<span>📄 {src["source_file"]}</span>'
                        f'<span>EventID: {src.get("event_id") or "—"}</span>'
                        f'<span>Host: {src["hostname"]}</span>'
                        f'<span>{src["timestamp"]}</span>'
                        f'</div>', unsafe_allow_html=True,
                    )
                    st.code(text[:800], language=None)
            st.markdown('</div>', unsafe_allow_html=True)

        if ss.active_job_id:
            # While a query is running: ONLY the live status shows — no
            # input box, no notices, nothing else competing for attention.
            # It reads straight from _JOBS, so it survives a section switch.
            with _JOBS_LOCK:
                job = _JOBS.get(ss.active_job_id)
                stage = job["stage"] if job else "Starting…"
            render_live_status(stage)
        else:
            if ss.job_notice:
                notice = ss.job_notice
                if notice["status"] == "rejected":
                    st.warning(
                        "This system only answers forensic questions (logons, process creation, "
                        "network connections, malware activity, incident response)."
                    )
                elif notice["status"] == "empty":
                    st.info("No matching events found in the forensic database for this query.")
                elif notice["status"] == "error":
                    st.error(f"{notice.get('error')}\n\nMake sure Ollama is running and `{LLM_MODEL}` is pulled.")
                ss.job_notice = None

            st.chat_input(
                "Ask a forensic question...", key="bottom_chat_q",
                on_submit=lambda: _submit_question(st.session_state.get("bottom_chat_q", ""), backend),
            )

    # Light auto-refresh ONLY while sitting on this tab with a job running,
    # so the pulsing status + eventual answer show up without a manual
    # click. Navigating away simply stops this loop — the background
    # thread itself is unaffected and poll_active_job() still catches the
    # result on whatever rerun happens next, from any tab.
    if ss.active_job_id and ss.active_section == "investigate":
        time.sleep(1)
        st.rerun()


def render_ingest_tab(backend):
    case_ids = backend.get("case_ids") or []
    untagged = backend.get("untagged_count") or 0

    if ss.active_case_id:
        active_label = ss.active_case_id
    elif case_ids:
        active_label = ", ".join(case_ids) if len(case_ids) <= 3 else f"{len(case_ids)} cases tagged"
    elif untagged:
        active_label = f"{untagged:,} events — no case tag yet"
    else:
        active_label = "No case ingested yet this session"

    st.markdown(
        f'<div class="af-panel" style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span class="af-label">CURRENT ACTIVE CASE</span>'
        f'<span class="af-cite">{active_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if untagged:
        st.markdown(
            f'<div class="af-panel" style="border-color:rgba(230,160,60,.4);">'
            f'<span class="af-label">⚠ Untagged legacy data</span><br>'
            f'<span style="color:var(--ink-soft);font-size:.88rem;">'
            f'{untagged:,} events in the <code>{COLLECTION_NAME}</code> collection have no case ID — '
            f'these came from the original CLI <code>ingest.py</code> run, which predates case tracking. '
            f'Give them a case ID below so they show up properly instead of as "no case ingested".'
            f'</span></div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            legacy_case_id = st.text_input(
                "Case ID for existing untagged events",
                value=f"legacy-{datetime.now().strftime('%Y%m%d')}",
                key="legacy_case_id_input",
            )
        with c2:
            st.write("")
            st.write("")
            if st.button("Tag Existing Events", use_container_width=True, key="tag_legacy_btn"):
                n = tag_untagged_events(backend["client"], backend["payloads"], legacy_case_id)
                backend["case_ids"] = sorted(set((backend.get("case_ids") or []) + [legacy_case_id]))
                backend["untagged_count"] = 0
                ss.active_case_id = legacy_case_id
                st.success(f"Tagged {n:,} existing events with case `{legacy_case_id}`.")
                st.rerun()
        st.write("")

    st.markdown(
        '<div class="af-panel"><span class="af-label">Add raw forensic artifacts</span><br>'
        '<span style="color:var(--ink-soft);font-size:.88rem;">'
        'Allowed file types: .evtx, .pcap, .pcapng, .csv, .log. Each upload is parsed, '
        'embedded, and stored in the same `forensic_events` Qdrant collection, tagged with a case ID.'
        '</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")

    case_mode = st.radio("Case", ["New case", "Existing case"], horizontal=True)
    existing_cases = sorted({p.get("case_id", "") for p in (backend["payloads"] or {}).values() if p.get("case_id")})
    if case_mode == "New case":
        case_id = st.text_input("New case ID", value=ss.draft_case_id, key="new_case_id_input")
        ss.draft_case_id = case_id
    else:
        case_id = st.selectbox("Select existing case", existing_cases) if existing_cases else None
        if not existing_cases:
            st.info("No existing cases found yet — previously ingested data has no case_id tag. Use 'New case' instead.")

    uploaded = st.file_uploader(
        "Upload raw artifacts", type=["evtx", "pcap", "pcapng", "csv", "log"], accept_multiple_files=True
    )

    if st.button("Validate & Ingest", type="primary", disabled=not uploaded or not case_id):
        run_ingest(uploaded, case_id, backend)


def render_report_tab():
    st.markdown(
        '<div class="af-panel"><span class="af-label">Generate session report</span><br>'
        '<span style="color:var(--ink-soft);font-size:.88rem;">'
        'Builds a Markdown + PDF report from every Q&amp;A turn in this session, with a '
        'programmatic Evidence Inventory table (not LLM-written) for the citation list.'
        '</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if not ss.chat_history:
        st.info("No Q&A turns yet this session — ask something in Investigate first.")
        return
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating Markdown + PDF report (LLM call)..."):
            paths = generate_report(ss.chat_history, output_dir=OUTPUT_DIR)
        if paths:
            c1, c2 = st.columns(2)
            if paths.get("md") and os.path.exists(paths["md"]):
                with open(paths["md"], "rb") as f:
                    c1.download_button("⬇ Download .md", f, file_name=os.path.basename(paths["md"]), use_container_width=True)
            if paths.get("pdf") and os.path.exists(paths["pdf"]):
                with open(paths["pdf"], "rb") as f:
                    c2.download_button("⬇ Download .pdf", f, file_name=os.path.basename(paths["pdf"]), use_container_width=True)


def render_settings_tab(backend, ollama_ok):
    st.markdown(
        '<div class="af-panel"><span class="af-label">Pipeline configuration</span><br>'
        '<span style="color:var(--ink-soft);font-size:.88rem;">'
        'Tune retrieval depth and generation temperature for this session. Defaults come '
        'from config.py — changes here apply immediately but aren\'t saved to disk.'
        '</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown('<span class="af-label">Retrieval configuration</span>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ss.cfg_dense_top_k = st.slider("Dense top-k (Qdrant)", 5, 50, ss.cfg_dense_top_k)
        ss.cfg_rrf_k = st.slider("RRF constant (k)", 10, 100, ss.cfg_rrf_k)
    with c2:
        ss.cfg_sparse_top_k = st.slider("Sparse top-k (BM25)", 5, 50, ss.cfg_sparse_top_k)
        ss.cfg_rrf_top_n = st.slider("Fused results sent to LLM", 1, 10, ss.cfg_rrf_top_n)

    st.write("")
    st.markdown('<span class="af-label">LLM generation</span>', unsafe_allow_html=True)
    ss.cfg_llm_temp = st.slider("Temperature", 0.0, 1.0, ss.cfg_llm_temp, step=0.05)
    st.caption(
        f"Model: {LLM_MODEL}  ·  Embedding: {EMBED_MODEL}  ·  Ollama: {'UP' if ollama_ok else 'DOWN'}"
    )
    st.caption(
        "Defaults come from config.py. Raising temperature above 0 trades away the "
        "project's zero-hallucination determinism guarantee — change for testing only, "
        "not for graded runs."
    )

    if st.button("Reset to defaults"):
        ss.cfg_dense_top_k, ss.cfg_sparse_top_k = DENSE_TOP_K, SPARSE_TOP_K
        ss.cfg_rrf_k, ss.cfg_rrf_top_n = RRF_K, RRF_TOP_N
        ss.cfg_llm_temp = LLM_TEMPERATURE
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    poll_active_job()  # runs on EVERY rerun, from EVERY tab — this is what
                        # makes a finished query show up even if you switched
                        # away from Investigate while it was running.
    backend = load_backend()
    ollama_ok = check_ollama()

    if ss.page == "landing":
        render_landing(backend, ollama_ok)
    else:
        render_investigate(backend, ollama_ok)


if __name__ == "__main__":
    main()