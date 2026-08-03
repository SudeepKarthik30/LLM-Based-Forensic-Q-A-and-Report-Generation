"""
session_store.py — Persisted Investigation Sessions
─────────────────────────────────────────────────────────────────────────────
Streamlit's session_state (ss.chat_history) only lives as long as the browser
tab / server process does — refresh the page or restart Streamlit and it's
gone. This module gives every investigation a durable identity on disk so:

  • Every chat gets a session_id the moment its first turn completes.
  • A fresh browser session / app restart = a brand new session_id + empty
    chat, matching the "new chat every restart" behaviour of ChatGPT/Claude.
  • The sidebar can list every past investigation (case, first question,
    turn count, last-updated time) and reload any of them back into
    ss.chat_history so the user can keep going where they left off.

Storage format
--------------
One JSON file per session at:  {SESSIONS_DIR}/{session_id}.json

    {
      "session_id": "a1b2c3d4",
      "case_id": "case-20260725-1223" | null,
      "created_at": "2026-07-25T06:07:25Z",
      "updated_at": "2026-07-25T06:12:03Z",
      "chat_history": [ {question, answer, sources, retrieval_confidence}, ... ],
      "evidence_texts_by_turn": [ [chunk_text, ...], ... ]
    }

No database needed at this scale (a capstone demo will have dozens of
sessions, not millions) — flat JSON files keep this dependency-free and easy
to inspect/debug by hand.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
from datetime import datetime, timezone

from config import SESSIONS_DIR


def _path(session_id):
    # session_id is always our own uuid4 hex slice — safe to use directly,
    # but strip any path separators defensively anyway.
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
    return os.path.join(SESSIONS_DIR, f"{safe_id}.json")


def save_session(session_id, case_id, chat_history, evidence_texts_by_turn):
    """
    Writes/overwrites the full session record to disk. Called after every
    completed Q&A turn so nothing is lost if the app crashes or restarts.
    """
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    path = _path(session_id)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    created_at = now_str
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                created_at = json.load(f).get("created_at", now_str)
        except Exception:
            pass

    record = {
        "session_id": session_id,
        "case_id": case_id,
        "created_at": created_at,
        "updated_at": now_str,
        "chat_history": chat_history,
        "evidence_texts_by_turn": evidence_texts_by_turn,
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)  # atomic-ish swap, avoids truncated files on crash


def load_session(session_id):
    """Returns the full session record dict, or None if it doesn't exist."""
    path = _path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_session(session_id):
    path = _path(session_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def list_sessions(limit=50):
    """
    Returns lightweight summaries for every persisted session, most-recently
    updated first — exactly what the sidebar history list needs, without
    loading full chat transcripts into memory.

    Output: list of dicts with session_id, case_id, updated_at, turn_count,
            first_question (truncated to 60 chars for display).
    """
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    summaries = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(SESSIONS_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue  # skip corrupt/partial files rather than crashing the sidebar

        turns = rec.get("chat_history", [])
        first_q = turns[0]["question"] if turns else "(empty session)"
        summaries.append({
            "session_id":     rec.get("session_id", fname[:-5]),
            "case_id":        rec.get("case_id"),
            "created_at":     rec.get("created_at", ""),
            "updated_at":     rec.get("updated_at", ""),
            "turn_count":     len(turns),
            "first_question": (first_q[:60] + "…") if len(first_q) > 60 else first_q,
        })

    summaries.sort(key=lambda s: s["updated_at"], reverse=True)
    return summaries[:limit]