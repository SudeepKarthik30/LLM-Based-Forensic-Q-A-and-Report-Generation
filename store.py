"""
store.py  —  Phase 2: Chunking · Embedding · Qdrant Storage · BM25 Index
─────────────────────────────────────────────────────────────────────────────
This file is the "memory builder" for the RAG pipeline.
It takes the parsed events from Phase 1 and:

  1.  Turns each event into a searchable "chunk" (one chunk = one event).
  2.  Generates a 768-dim embedding for each chunk using nomic-embed-text.
  3.  Stores those vectors + full metadata in a local Qdrant collection.
  4.  Builds a BM25 keyword index and saves it to disk for fast query-time use.

Changes from original
─────────────────────
• All constants imported from config.py — no more inline literals.
• ensure_collection: creates Qdrant payload indexes on data.EventID,
  event_type, hostname, and timestamp after collection creation so
  filtered queries don't full-scan at scale.
• Docstring accuracy fix: load_bm25 now correctly documents that the
  BM25Okapi model IS rebuilt from persisted tokens each session start
  (not pre-fitted and saved) — about ~1s for 50k docs, acceptable.

Nothing here touches Phase-1 code — it only reads the output of parsers.py.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import logging
import pickle

from tqdm import tqdm
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    PayloadSchemaType,
)
from rank_bm25 import BM25Okapi

from config import (
    QDRANT_HOST, QDRANT_PORT,
    COLLECTION_NAME, EMBED_MODEL, EMBED_DIM,
    MAX_CHUNK_CHARS, EMBED_CHECKPOINT,
)

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING  (one chunk per Phase-1 event)
# ─────────────────────────────────────────────────────────────────────────────

def build_chunk_text(event):
    """
    Converts one Phase-1 event dict into a single flat, human-readable string.

    This string is what we embed and later search.  The more detail we cram in,
    the better the embedding captures the event's semantics.  Think of it as
    writing a one-line detective note about every entry in the log.

    Input  : one event dict  (source_file, format, timestamp, hostname, event_type, data)
    Output : a pipe-separated string containing every meaningful field
    """
    parts = []

    # Top-level Phase-1 fields
    parts.append(f"Format:{event.get('format', '')}")
    parts.append(f"Source:{event.get('source_file', '')}")
    parts.append(f"Time:{event.get('timestamp', '')}")
    parts.append(f"Host:{event.get('hostname', '')}")
    parts.append(f"Type:{event.get('event_type', '')}")

    data = event.get('data', {})

    # Common structured fields (present in evtx events)
    for key in ('EventID', 'Channel', 'ProcessID', 'UserSID'):
        val = data.get(key, '')
        if val:
            parts.append(f"{key}:{val}")

    # Flatten the nested EventData sub-dict (evtx)
    edata = data.get('EventData', {})
    if isinstance(edata, dict):
        for k, v in edata.items():
            if v and str(v).strip() not in ('', '-', '%%1793', '%%1794'):
                parts.append(f"{k}:{str(v).strip()}")

    # For pcap events the interesting fields are at the top-level data dict
    for key in ('src_ip', 'dst_ip', 'protocol', 'src_port', 'dst_port',
                'alert', 'http_method', 'http_uri', 'ttl', 'packet_length'):
        val = data.get(key)
        if val is not None and str(val).strip():
            parts.append(f"{key}:{val}")

    # For CSV/log events store the raw line
    if data.get('raw_line'):
        parts.append(f"raw:{data['raw_line'][:300]}")

    text = " | ".join(parts)

    # ── Hard cap to prevent Ollama HTTP 500 on huge EventData blobs ──────────
    # nomic-embed-text context ≈ 8192 tokens (~32k chars). 6000 chars is safe.
    # Payloads that exceed this are almost always registry exports or verbose
    # CommandLine strings that carry little extra semantic signal beyond 6k.
    if len(text) > MAX_CHUNK_CHARS:
        text = text[:MAX_CHUNK_CHARS] + " ...[TRUNCATED]"

    return text


def make_chunks(events):
    """
    Converts the full list of Phase-1 events into chunks ready for embedding.

    One chunk = one event.  Each chunk gets:
      - chunk_id   : integer index (used as Qdrant point ID)
      - chunk_text : flat string built by build_chunk_text()
      - payload    : the original event dict (preserved for citation display)

    Input  : list of event dicts (output of parse_all_files)
    Output : list of chunk dicts
    """
    chunks = []
    for idx, event in enumerate(events):
        chunks.append({
            "chunk_id":   idx,
            "chunk_text": build_chunk_text(event),
            "payload":    event,          # keep the full original event intact
        })
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING  (nomic-embed-text via Ollama)
# ─────────────────────────────────────────────────────────────────────────────

def embed_text(text):
    """
    Sends a single string to Ollama and returns a 768-dim float vector.

    Fault isolation (two-stage fallback)
    -------------------------------------
    Some EventData payloads still exceed Ollama's per-request context even after
    the MAX_CHUNK_CHARS cap (e.g. multi-line CommandLine blocks with Unicode).
    Rather than crash the whole ingest run, we try progressively shorter inputs:

      Attempt 1: full text  (should succeed 99.9% of the time after the cap)
      Attempt 2: first 3000 chars  (well inside any model's context window)
      Failure:   return None  — caller skips this chunk from dense indexing
                               but still keeps it in BM25 (keyword-findable)

    [WARNING]  Zero vectors are NOT used as fallback.  Qdrant cosine similarity =
        dot(a,b)/(|a|×|b|).  A zero vector gives |a|=0 → division by zero →
        NaN scores that poison the entire result set for that query.

    Input  : any string (already capped by build_chunk_text)
    Output : list of 768 floats, OR None on total failure
    """
    for limit in [None, 3000]:
        try:
            t = text if limit is None else text[:limit]
            return ollama.embeddings(model=EMBED_MODEL, prompt=t)['embedding']
        except Exception as exc:
            _log.warning("embed_text attempt (limit=%s) failed: %s", limit, exc)
            continue

    _log.error(
        "embed_text: all attempts failed — chunk excluded from dense index. "
        "Snippet: %.80s", text
    )
    return None   # caller (embed_batch) handles this


# Checkpoint save interval — every N completed embeddings
_CHECKPOINT_EVERY = 500


def embed_batch(texts, max_workers=4, checkpoint_file=None):
    """
    Embeds a list of strings concurrently using a thread pool with checkpointing.

    Concurrency
    -----------
    Each text becomes one HTTP call to Ollama.  max_workers=4 parallel calls
    give a real ~3-4x speedup over sequential since wall-clock time is mostly
    I/O waiting for the HTTP response.  Results are written to a pre-allocated
    list by index to preserve order regardless of completion order.

    Checkpointing
    -------------
    Every _CHECKPOINT_EVERY completions the done-dict is pickled to
    checkpoint_file (default: EMBED_CHECKPOINT from config.py).  On restart,
    already-embedded indices are loaded and skipped, so a crash at 81% doesn't
    mean losing 2 hours of work.

    Known IO tradeoff: the full done-dict (~180 MB for 30k×768 floats) is
    rewritten to disk on each save.  Near the end of the run this means
    rewriting ~180 MB every 500 chunks.  Acceptable for 30k-event corpus;
    document as known limitation for 150 GB scale.

    None handling
    -------------
    embed_text returns None on total failure.  embed_batch preserves those
    None values in all_embeddings.  upsert_to_qdrant skips None entries so
    zero vectors never reach Qdrant.

    Input  : list of strings
             max_workers      : parallel Ollama HTTP calls (default 4)
             checkpoint_file  : path to checkpoint pickle (default from config)
    Output : list where each element is either a 768-float vector or None
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if checkpoint_file is None:
        checkpoint_file = EMBED_CHECKPOINT

    # ── Load existing checkpoint ──────────────────────────────────────────────
    done = {}   # {idx: embedding_or_None}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                done = pickle.load(f)
            print(f"   Resuming from checkpoint: {len(done)}/{len(texts)} already embedded")
        except Exception as exc:
            _log.warning("Could not load checkpoint (starting fresh): %s", exc)
            done = {}

    # Restore already-done results into the output array
    all_embeddings = [None] * len(texts)
    for idx, emb in done.items():
        all_embeddings[idx] = emb

    todo_indices = [i for i in range(len(texts)) if i not in done]

    if not todo_indices:
        print("  [OK] All chunks already embedded (checkpoint complete).")
        return all_embeddings

    # ── Concurrent embedding with checkpoint saves ────────────────────────────
    pbar = tqdm(
        total=len(todo_indices),
        desc="   [EMBED] Embedding",
        unit="chunk",
    )

    completed_since_save = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(embed_text, texts[i]): i
            for i in todo_indices
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()          # None on embed_text failure
            all_embeddings[idx] = result
            done[idx] = result
            completed_since_save += 1
            pbar.update(1)

            # Save checkpoint every N completions
            if completed_since_save % _CHECKPOINT_EVERY == 0:
                try:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(done, f)
                except Exception as exc:
                    _log.warning("Checkpoint save failed: %s", exc)

    pbar.close()

    # ── Final checkpoint save ─────────────────────────────────────────────────
    try:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(done, f)
    except Exception as exc:
        _log.warning("Final checkpoint save failed: %s", exc)

    # Report failed embeddings
    failed = sum(1 for e in all_embeddings if e is None)
    if failed:
        print(f"  [WARNING]  {failed} chunk(s) failed embedding — excluded from Qdrant "
              f"dense index, still in BM25 keyword index.")

    return all_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# QDRANT OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_qdrant_client():
    """
    Creates and returns a QdrantClient.

    Local dev  : connects to Qdrant via TCP (Docker or native binary).
                 Host/port from config.py (QDRANT_HOST / QDRANT_PORT).

    Colab mode : if env var QDRANT_PATH is set, switches to local-file mode
                 (QdrantClient(path=...)) which needs no Docker and persists
                 to the specified directory (ideally a Google Drive path).

    Example Colab cell:
        import os
        os.environ['QDRANT_PATH'] = '/content/drive/MyDrive/RV/qdrant_data'
        from store import get_qdrant_client
        client = get_qdrant_client()   # → local-file mode, no Docker needed
    """
    qdrant_path = os.environ.get("QDRANT_PATH", None)
    if qdrant_path:
        os.makedirs(qdrant_path, exist_ok=True)
        return QdrantClient(path=qdrant_path)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def ensure_collection(client, force_recreate=False):
    """
    Makes sure the 'forensic_events' Qdrant collection exists before we try to
    insert vectors.  If it already exists and force_recreate=False, we leave it
    alone (so you don't accidentally wipe previous ingestions).

    Set force_recreate=True only when you want to start completely fresh.

    Payload indexes created
    -----------------------
    After creating a fresh collection, we create keyword payload indexes on:
      • data.EventID    — used by dense_search filters in retriever.py
      • event_type      — useful for type-filtered queries
      • hostname        — useful for host-scoped queries
      • timestamp       — useful for time-range queries (ISO-8601 UTC strings)

    Without these indexes, Qdrant must full-scan the payload on every filtered
    query.  With them, filters are applied before HNSW traversal, which keeps
    performance acceptable as the evidence set grows.

    Inputs
    ------
    client         : QdrantClient
    force_recreate : bool — wipe and recreate if True

    Output : True if a fresh collection was created, False if it already existed
    """
    existing_names = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing_names:
        if force_recreate:
            print(f"   ️  Deleting existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"   ℹ️  Collection '{COLLECTION_NAME}' already exists — adding to it.")
            return False   # collection already there, nothing to create

    # Create with cosine distance — best for semantic text embeddings
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"   [OK]  Created collection '{COLLECTION_NAME}'  (dim={EMBED_DIM}, cosine)")

    # Create payload indexes for the fields we filter on in retriever.py
    _index_fields = [
        ("data.EventID", PayloadSchemaType.KEYWORD),
        ("event_type",   PayloadSchemaType.KEYWORD),
        ("hostname",     PayloadSchemaType.KEYWORD),
        ("timestamp",    PayloadSchemaType.KEYWORD),
    ]
    for field_name, schema_type in _index_fields:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=schema_type,
        )
        print(f"   [OK]  Payload index created on '{field_name}'")

    return True


def upsert_to_qdrant(client, chunks, embeddings, batch_size=256):
    """
    Stores (chunk, embedding) pairs as points in Qdrant, skipping None embeddings.

    Each Qdrant point has:
      - id      : the chunk_id integer
      - vector  : the 768-dim embedding
      - payload : the full original event + chunk_text (for citation display)

    We upsert in batches of 256 to avoid hitting Qdrant's request-size limits.

    None embedding handling
    -----------------------
    embed_batch returns None for chunks that failed all embedding attempts.
    These chunks are EXCLUDED from Qdrant (zero vectors must never be stored
    — they cause cosine division-by-zero NaN in search results).  They remain
    in the BM25 index and are keyword-searchable.  The count of excluded chunks
    is logged here.

    Inputs
    ------
    client     : QdrantClient
    chunks     : list of chunk dicts from make_chunks()
    embeddings : list from embed_batch() — may contain None entries
    batch_size : how many points to send per Qdrant request
    """
    # ── Filter out None embeddings ────────────────────────────────────────────
    good_pairs   = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]
    skipped      = len(chunks) - len(good_pairs)
    if skipped:
        print(f"  [WARNING]  Skipped {skipped} chunk(s) from Qdrant (embedding failed).")
    _log.info("upsert_to_qdrant: %d good, %d skipped", len(good_pairs), skipped)

    # ── Build PointStruct objects ─────────────────────────────────────────────
    points = []
    for chunk, vector in good_pairs:
        payload = dict(chunk["payload"])
        payload["chunk_text"] = chunk["chunk_text"]
        payload["chunk_id"]   = chunk["chunk_id"]
        points.append(PointStruct(
            id=chunk["chunk_id"],
            vector=vector,
            payload=payload,
        ))

    # ── Upsert in batches ─────────────────────────────────────────────────────
    for i in tqdm(
        range(0, len(points), batch_size),
        desc="    Upserting to Qdrant",
        unit="batch",
    ):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size],
        )


# ─────────────────────────────────────────────────────────────────────────────
# BM25 KEYWORD INDEX
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text):
    """
    Breaks a string into lowercase alphanumeric tokens for BM25 indexing.

    We keep alphanumeric runs only (numbers are important — IPs, EventIDs, ports).
    Example: "EventID:4625 | Host:DC01" → ["eventid", "4625", "host", "dc01"]

    Input  : any string
    Output : list of lowercase token strings
    """
    return re.findall(r'[a-zA-Z0-9]+', text.lower())


def build_and_save_bm25(chunks, cache_path):
    """
    Tokenises every chunk_text, builds a BM25Okapi index over the corpus, and
    saves the tokenized corpus + chunk IDs to a pickle file on disk.

    BM25 is great for exact keyword hits that semantic search might miss:
      - IP addresses  ("192.168.1.45")
      - Process names ("mimikatz.exe")
      - Hashes        ("aabbccdd…")
      - EventID numbers ("4625")

    What is persisted vs. rebuilt
    ------------------------------
    We persist the raw corpus_tokens list (not the BM25Okapi object itself,
    because BM25Okapi is not reliably pickleable across library versions).
    load_bm25() refits BM25Okapi from those tokens at session start — this
    takes ~1–2 seconds for 50k documents and is acceptable.

    Inputs
    ------
    chunks     : list of chunk dicts (same as make_chunks output)
    cache_path : where to write the pickle file (e.g. code/bm25_cache.pkl)

    Output : (BM25Okapi index, list of token lists)
    """
    print("   Building token corpus...", end=" ", flush=True)
    corpus_tokens = [tokenize(c["chunk_text"]) for c in chunks]
    chunk_ids     = [c["chunk_id"] for c in chunks]
    print("done")

    print("   Fitting BM25 index...", end=" ", flush=True)
    bm25 = BM25Okapi(corpus_tokens)
    print("done")

    # Persist the raw corpus + IDs so we can refit BM25 at query time in ~1s
    with open(cache_path, 'wb') as f:
        pickle.dump({"corpus_tokens": corpus_tokens, "chunk_ids": chunk_ids}, f)

    print(f"   [OK]  BM25 token corpus saved → {cache_path}")
    return bm25, corpus_tokens


def load_bm25(cache_path):
    """
    Loads the pre-built BM25 corpus from disk and reconstructs the BM25Okapi
    index.  This is fast (< 2 seconds for tens of thousands of docs).

    Important: what is saved to disk is the raw corpus_tokens list, NOT a
    pre-fitted BM25Okapi object.  The BM25Okapi model is always rebuilt from
    tokens at session start.  This is intentional — it avoids pickle version
    compatibility issues at the cost of ~1–2s startup time.

    Input  : path to the pickle file saved by build_and_save_bm25()
    Output : (BM25Okapi index, corpus token lists, chunk ID list)
    """
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)

    corpus_tokens = data["corpus_tokens"]
    chunk_ids     = data["chunk_ids"]
    bm25          = BM25Okapi(corpus_tokens)   # refit from tokens (~1-2s)

    return bm25, corpus_tokens, chunk_ids
