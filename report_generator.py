"""
report_generator.py  —  Phase 4: Forensic Report Generation
─────────────────────────────────────────────────────────────────────────────
Implements the Phase 4 report generator described in the project spec:
  - Executive Summary
  - Technical Timeline (chronological evidence from the session)
  - Evidence Inventory (table of all sources cited, with metadata)

Outputs
-------
  forensic_report_YYYYMMDD_HHMMSS.md   — Markdown version (human-readable)
  forensic_report_YYYYMMDD_HHMMSS.pdf  — PDF version  (spec Week 6 requirement)

Usage
-----
This module is called automatically from query.py when the user types
'report' at the Q&A prompt.

Design decisions
----------------
• chat_history is a list of dicts: {question, answer, sources}
  where sources is the list of source metadata dicts from
  build_context_and_sources().  This structured format enables the
  Evidence Inventory table to be built with real file/eventid/timestamp/host
  data, not just answer strings.

• num_ctx is set to REPORT_NUM_CTX (16384, from config.py) — double the
  Q&A window — to accommodate a full session's worth of Q&A pairs in one
  LLM call.

• chat_history is capped at REPORT_MAX_TURNS (20, from config.py) before
  being sent to the LLM.  If the session had more turns, the most recent
  20 are used and a warning is included in the report header.  This prevents
  silent mid-prompt truncation if the context window is exhausted.

• The Evidence Inventory is built programmatically from the sources lists
  (not from LLM output), so it is always accurate regardless of what the
  LLM says.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
from datetime import datetime, timezone

import ollama

from config import (
    LLM_MODEL, REPORT_NUM_CTX, REPORT_MAX_TURNS, OUTPUT_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT FOR REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

_REPORT_SYSTEM_PROMPT = """You are a senior digital forensics analyst writing a formal incident report.
You will be given a log of forensic Q&A pairs from an investigation session.
Write a concise, professional forensic report with EXACTLY these three sections:

EXECUTIVE SUMMARY
(1–2 paragraphs): Describe what happened overall, what threat actor activity was found,
and the likely scope of the incident.  Be precise and factual.

TECHNICAL TIMELINE
(bulleted or numbered list): List key events in chronological order.
Use timestamps where available.  Cite evidence references as [Source N] from the Q&A.

Each section must be clearly headed.  Do NOT add extra sections.
Do NOT speculate beyond the evidence provided.  Be concise.
"""


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE INVENTORY BUILDER  (programmatic — not LLM-generated)
# ─────────────────────────────────────────────────────────────────────────────

def _build_evidence_inventory(chat_history):
    """
    Builds the Evidence Inventory section as a Markdown table by collecting
    every unique source cited across all Q&A turns in the session.

    Sources are deduplicated by (source_file, event_id, timestamp) so the
    same log entry cited in multiple questions only appears once.

    This section is built entirely from structured source metadata — not from
    the LLM narrative — so the table is always accurate.

    Input  : list of {question, answer, sources} dicts
    Output : Markdown table string
    """
    seen     = set()
    rows     = []
    row_num  = 1

    for turn in chat_history:
        for src in turn.get("sources", []):
            key = (
                src.get("source_file", ""),
                src.get("event_id",    ""),
                src.get("timestamp",   ""),
            )
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "num":         row_num,
                "source_file": src.get("source_file", "—"),
                "event_id":    src.get("event_id",    "—"),
                "event_type":  src.get("event_type",  "—"),
                "timestamp":   src.get("timestamp",   "—"),
                "hostname":    src.get("hostname",     "—"),
            })
            row_num += 1

    if not rows:
        return "_No sources were cited during this session._\n"

    header = (
        "| # | Source File | EventID | Event Type | Timestamp (UTC) | Host |\n"
        "|---|-------------|---------|------------|-----------------|------|\n"
    )
    body = "\n".join(
        f"| {r['num']} | `{r['source_file']}` | {r['event_id']} "
        f"| {r['event_type']} | {r['timestamp']} | {r['hostname']} |"
        for r in rows
    )
    return header + body + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# SESSION SUMMARY FOR LLM
# ─────────────────────────────────────────────────────────────────────────────

def _format_session_for_llm(chat_history, max_turns):
    """
    Formats the chat history into a compact text block for the LLM report call.

    Caps at max_turns most-recent turns to avoid context overflow.
    Returns (formatted_text, was_capped, original_turn_count).
    """
    original_count = len(chat_history)
    was_capped     = original_count > max_turns
    turns          = chat_history[-max_turns:] if was_capped else chat_history

    lines = []
    for i, turn in enumerate(turns, start=1):
        lines.append(f"--- QUESTION {i} ---")
        lines.append(f"Q: {turn.get('question', '')}")
        lines.append(f"A: {turn.get('answer', '')}")
        lines.append("")

    return "\n".join(lines), was_capped, original_count


# ─────────────────────────────────────────────────────────────────────────────
# SHARED ROW EXTRACTOR  (used by both MD table builder and PDF renderer)
# ─────────────────────────────────────────────────────────────────────────────

def _get_inventory_rows(chat_history):
    """
    Returns the deduplicated list of source row dicts across the session.
    Extracted as a standalone helper so both the Markdown table builder and
    the PDF renderer can call it without duplicating dedup logic.

    Deduplication key: (source_file, event_id, timestamp)

    Input  : list of {question, answer, sources} dicts
    Output : list of row dicts with keys: num, source_file, event_id,
             event_type, timestamp, hostname
    """
    seen    = set()
    rows    = []
    row_num = 1
    for turn in chat_history:
        for src in turn.get("sources", []):
            key = (
                src.get("source_file", ""),
                src.get("event_id",    ""),
                src.get("timestamp",   ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "num":         row_num,
                "source_file": src.get("source_file", "—"),
                "event_id":    src.get("event_id",    "—"),
                "event_type":  src.get("event_type",  "—"),
                "timestamp":   src.get("timestamp",   "—"),
                "hostname":    src.get("hostname",     "—"),
            })
            row_num += 1
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXPORT  (fpdf2 — pure Python, no external binaries required)
# ─────────────────────────────────────────────────────────────────────────────

_UNICODE_REPLACEMENTS = {
    "\u2022": "-",   # • bullet
    "\u2023": "-",   # triangular bullet
    "\u25e6": "-",   # white bullet
    "\u2013": "-",   # – en dash
    "\u2014": "--",  # — em dash
    "\u2018": "'",   # ' left single quote
    "\u2019": "'",   # ' right single quote
    "\u201c": '"',   # " left double quote
    "\u201d": '"',   # " right double quote
    "\u2026": "...", # … ellipsis
}


def _safe(text):
    """Normalize common Unicode punctuation (bullets, smart quotes, dashes)
    to ASCII equivalents, then strip any remaining non-latin-1 characters
    that fpdf2 core fonts can't encode (falls back to '?' only as a last
    resort for truly unmappable characters, e.g. CJK or emoji)."""
    for uni, ascii_equiv in _UNICODE_REPLACEMENTS.items():
        text = text.replace(uni, ascii_equiv)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _save_as_pdf(pdf_path, now_str, original_count, unique_files,
                 narrative, inventory_rows, chat_history, cap_warning=""):
    """
    Renders the forensic report as a multi-page PDF using fpdf2.

    Sections
    --------
    1.  Title page  — report title, metadata table
    2.  Narrative   — Executive Summary + Technical Timeline (LLM text)
    3.  Evidence Inventory — formatted table of all cited sources
    4.  Q&A Log     — full transcript, one turn per block

    Uses fpdf2's built-in core fonts (Helvetica/Courier) — no font files
    needed, no external binaries, works on any OS.

    Input  : all section data pre-computed by generate_report()
    Output : True if PDF was saved successfully, False on error
    """
    try:
        from fpdf import FPDF
    except ImportError:
        print("  [WARNING]  fpdf2 not installed — PDF export skipped.")
        print("      Install with:  pip install fpdf2==2.8.3")
        return False

    try:
        pdf = FPDF(unit="mm", format="A4")   # A4 = 210mm; margins 20+20 -> usable W=170mm
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_margins(left=20, top=20, right=20)
        W = 170   # explicit usable width — never use 0 (causes position corruption)

        # ── Page 1: Title + Metadata ──────────────────────────────────────────
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(W, 12, _safe("FORENSIC INVESTIGATION REPORT"),
                 new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        pdf.set_fill_color(235, 235, 235)
        meta = [
            ("Generated (UTC)",           now_str),
            ("Session Q&A Turns",         str(original_count)),
            ("Unique Source Files Cited", str(unique_files)),
            ("LLM Model",                 "llama3:8b"),
            ("Retrieval",                 "nomic-embed-text + BM25 + RRF"),
        ]
        col_w = [75, 95]   # sum = 170 mm
        for label, value in meta:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(col_w[0], 7, _safe(label), border=1, fill=True, new_x="RIGHT", new_y="TOP")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(col_w[1], 7, _safe(value), border=1, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        if cap_warning:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(180, 60, 0)
            clean_warn = (cap_warning.replace("> ", "").replace("**", "")
                          .replace("\n", " ").strip())
            pdf.multi_cell(W, 5, _safe(clean_warn[:300]), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

        # ── Page 2: Narrative ────────────────────────────────────────────────
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(W, 10, _safe("Narrative"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

        for line in narrative.splitlines():
            stripped = line.strip()
            if not stripped:
                pdf.ln(3)
                continue
            if stripped.isupper() or stripped.startswith("##") or stripped.startswith("**"):
                pdf.set_font("Helvetica", "B", 11)
                clean = stripped.strip("#").strip("*").strip()
                pdf.multi_cell(W, 7, _safe(clean), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
            else:
                pdf.multi_cell(W, 5, _safe(stripped), new_x="LMARGIN", new_y="NEXT")

        # ── Page 3: Evidence Inventory table ─────────────────────────────────
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(W, 10, _safe("Evidence Inventory"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        if not inventory_rows:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(W, 7, _safe("No sources cited during this session."),
                     new_x="LMARGIN", new_y="NEXT")
        else:
            cw      = [8, 42, 14, 36, 36, 34]   # sum = 170 mm [OK]
            headers = ["#", "Source File", "EID", "Event Type", "Timestamp (UTC)", "Host"]

            # Header row (dark background, white text)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(50, 50, 50)
            pdf.set_text_color(255, 255, 255)
            last = len(headers) - 1
            for i, h in enumerate(headers):
                nx = "LMARGIN" if i == last else "RIGHT"
                ny = "NEXT"    if i == last else "TOP"
                pdf.cell(cw[i], 7, _safe(h), border=1, fill=True, new_x=nx, new_y=ny)

            # Data rows (alternating grey/white)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(0, 0, 0)
            max_chars = [4, 28, 8, 22, 22, 20]
            for row_idx, row in enumerate(inventory_rows):
                fill = row_idx % 2 == 0
                if fill:
                    pdf.set_fill_color(245, 245, 245)
                else:
                    pdf.set_fill_color(255, 255, 255)
                vals = [
                    str(row["num"]),
                    row["source_file"],
                    row["event_id"],
                    row["event_type"],
                    row["timestamp"],
                    row["hostname"],
                ]
                for i, v in enumerate(vals):
                    nx = "LMARGIN" if i == last else "RIGHT"
                    ny = "NEXT"    if i == last else "TOP"
                    pdf.cell(cw[i], 6, _safe(str(v))[:max_chars[i]],
                             border=1, fill=fill, new_x=nx, new_y=ny)

        # ── Page 4: Q&A Transcript ────────────────────────────────────────────
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(W, 10, _safe("Session Q&A Transcript"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        for i, turn in enumerate(chat_history, start=1):
            q = turn.get("question", "")
            a = turn.get("answer",   "")

            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(W, 6, _safe(f"Q{i}: {q}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            # NOTE: previously this was a[:1200], which hard-truncated any
            # answer over 1200 chars mid-word/mid-sentence. Confirmed via
            # isolated testing that fpdf2's multi_cell auto-paginates long
            # text cleanly across pages on its own — no cap needed at all.
            pdf.multi_cell(W, 5, _safe(a), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y())
            pdf.ln(3)

        pdf.output(pdf_path)
        return True

    except Exception as exc:
        print(f"  WARNING: PDF export failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────


def generate_report(chat_history, output_dir=None):
    """
    Generates a Markdown forensic report from the current Q&A session and
    saves it to disk.

    Report sections
    ---------------
    1. Report header   (session metadata, timestamp, file counts)
    2. Executive Summary + Technical Timeline  (LLM-generated)
    3. Evidence Inventory  (programmatic Markdown table from source metadata)

    Context overflow protection
    ---------------------------
    chat_history is capped at REPORT_MAX_TURNS (20) before the LLM call.
    If the session has more turns, the most recent 20 are used and a
    WARNING is included in the report header.  The full session is still
    reflected in the Evidence Inventory (all turns' sources are included).

    Inputs
    ------
    chat_history : list of {question, answer, sources} dicts from query.py
    output_dir   : directory to save the .md file (default: config.OUTPUT_DIR)

    Output : absolute path to the saved .md report file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    if not chat_history:
        print("  [WARNING]  No Q&A turns in session — nothing to report.")
        return None

    # ── 1. Prepare session text for LLM ───────────────────────────────────────
    session_text, was_capped, original_count = _format_session_for_llm(
        chat_history, max_turns=REPORT_MAX_TURNS
    )

    cap_warning = ""
    if was_capped:
        cap_warning = (
            f"\n> [WARNING]  **Context Cap Warning**: This session had {original_count} Q&A turns. "
            f"Only the most recent {REPORT_MAX_TURNS} turns were sent to the LLM for "
            f"narrative generation to avoid context overflow.  "
            f"All {original_count} turns are reflected in the Evidence Inventory below.\n"
        )

    # ── 2. LLM call for narrative sections ────────────────────────────────────
    print("    Generating report narrative (LLM call)...", end=" ", flush=True)
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': _REPORT_SYSTEM_PROMPT},
                {'role': 'user',   'content': session_text},
            ],
            options={
                'temperature': 0.0,
                'num_ctx':     REPORT_NUM_CTX,
            },
        )
        narrative = response['message']['content']
        print("done")
    except Exception as exc:
        print(f"FAILED\n  [WARNING]  Ollama error: {exc}")
        narrative = (
            "_Report narrative could not be generated — Ollama was unavailable._\n\n"
            f"Error: {exc}"
        )

    # ── 3. Build evidence inventory programmatically ───────────────────────────
    inventory_table = _build_evidence_inventory(chat_history)

    # ── 4. Assemble full Markdown report ──────────────────────────────────────
    now_str   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fname     = f"forensic_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
    out_path  = os.path.join(output_dir, fname)

    # Count unique source files cited
    all_sources = [s for turn in chat_history for s in turn.get("sources", [])]
    unique_files = len({s.get("source_file", "") for s in all_sources})

    report_md = f"""# Forensic Investigation Report

| Field | Value |
|-------|-------|
| Generated | {now_str} |
| Session Q&A Turns | {original_count} |
| Unique Source Files Cited | {unique_files} |
| LLM Model | {LLM_MODEL} |
| Retrieval | nomic-embed-text + BM25 + RRF |
{cap_warning}
---

{narrative}

---

## Evidence Inventory

Complete list of all unique log entries cited during this investigation session.
This table is generated programmatically from source metadata — not from LLM output.

{inventory_table}

---

## Session Q&A Log

<details>
<summary>Click to expand full Q&A transcript</summary>

"""
    for i, turn in enumerate(chat_history, start=1):
        q = turn.get("question", "")
        a = turn.get("answer",   "")
        report_md += f"**Q{i}:** {q}\n\n"
        report_md += f"**A{i}:**\n\n{a}\n\n"
        report_md += "---\n\n"

    report_md += "</details>\n"

    # ── 5. Save Markdown ───────────────────────────────────────────────────────
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"  [OK]  Markdown report saved  → {out_path}")

    # ── 6. Save PDF ──────────────────────────────────────────────────────────────────────
    pdf_path = out_path.replace(".md", ".pdf")
    pdf_ok   = _save_as_pdf(
        pdf_path       = pdf_path,
        now_str        = now_str,
        original_count = original_count,
        unique_files   = unique_files,
        narrative      = narrative,
        inventory_rows = _get_inventory_rows(chat_history),
        chat_history   = chat_history,
        cap_warning    = cap_warning,
    )
    if pdf_ok:
        print(f"  [OK]  PDF report saved       → {pdf_path}")

    return {"md": out_path, "pdf": pdf_path if pdf_ok else None}