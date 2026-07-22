#!/usr/bin/env python3
"""
timed_test.py — Non-interactive timing harness for the RAG pipeline.

Asks three questions in sequence with NO gap between them:
  Q1 & Q2 : identical pcap question  ->  reveals model-reload overhead
  Q3       : ransomware/exfil         ->  tests Rule 8 context-bleed fix

Prints [TIME] Retrieval / LLM / Total for each turn.
Run from:  d:\\Desktop\\RV_internship\\code
    ..\\venv\\Scripts\\python.exe timed_test.py
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force UTF-8 output so box-drawing chars don't crash on cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config    import COLLECTION_NAME, BM25_CACHE, DENSE_TOP_K, SPARSE_TOP_K, RRF_K, RRF_TOP_N, LLM_MODEL
from store     import get_qdrant_client, load_bm25
from retriever import (is_forensic_query_with_fallback, dense_search,
                       sparse_search, rrf_fuse, build_context_and_sources,
                       compute_retrieval_confidence)
from llm       import generate_answer_validated

QUESTIONS = [
    "What network traffic occurred between the two hosts in the pcap file?",
    "What network traffic occurred between the two hosts in the pcap file?",   # repeat — isolates reload cost
    "Is there evidence of ransomware or data exfiltration in this dataset?",
]

def load_all_payloads(client):
    payloads, offset = {}, None
    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME, offset=offset,
            limit=1000, with_payload=True, with_vectors=False,
        )
        for p in results:
            payloads[p.id] = p.payload
        if next_offset is None:
            break
        offset = next_offset
    return payloads


def run_one(question, client, bm25, corpus_tokens, chunk_ids, payloads_map, chat_history, q_num):
    print(f"\n{'='*66}")
    print(f"Q{q_num}: {question}")
    print('='*66)

    t_retrieval_start = time.time()

    allowed, gate_reason = is_forensic_query_with_fallback(question, bm25, corpus_tokens, chunk_ids)
    if not allowed:
        print("  GATE: REJECTED -- not forensic")
        return

    dense_results    = dense_search(client, question, top_k=DENSE_TOP_K)
    sparse_results   = sparse_search(bm25, corpus_tokens, chunk_ids, question, top_k=SPARSE_TOP_K)
    fused_ids        = rrf_fuse(dense_results, sparse_results, k=RRF_K, top_n=RRF_TOP_N)
    context, sources = build_context_and_sources(fused_ids, payloads_map)
    retrieval_confidence, _ = compute_retrieval_confidence(dense_results, sources)

    t_retrieval_elapsed = time.time() - t_retrieval_start
    print(f"  [TIME]  Retrieval : {t_retrieval_elapsed:.2f}s  ({len(sources)} sources, gate={gate_reason})")

    if not sources:
        print("  INSUFFICIENT EVIDENCE -- no sources found")
        return

    t_llm_start = time.time()
    answer = generate_answer_validated(
        question, context, sources, max_retries=1,
        chat_history=chat_history,
        retrieval_confidence=retrieval_confidence,
    )
    t_llm_elapsed = time.time() - t_llm_start

    print(f"  [TIME]  LLM gen   : {t_llm_elapsed:.2f}s")
    print(f"  [TIME]  Total     : {t_retrieval_elapsed + t_llm_elapsed:.2f}s")
    print(f"\n--- ANSWER PREVIEW (first 800 chars) ---")
    print(answer[:800])
    if len(answer) > 800:
        print("...")

    chat_history.append({"question": question, "answer": answer, "sources": sources})


def main():
    print("Loading components...")

    client    = get_qdrant_client()
    vec_count = client.count(collection_name=COLLECTION_NAME).count
    print(f"  Qdrant  : {vec_count:,} vectors")

    bm25, corpus_tokens, chunk_ids = load_bm25(BM25_CACHE)
    print(f"  BM25    : {len(chunk_ids):,} docs")

    payloads_map = load_all_payloads(client)
    print(f"  Payloads: {len(payloads_map):,} events")
    print(f"  Model   : {LLM_MODEL}")
    print()

    chat_history = []
    for i, q in enumerate(QUESTIONS, 1):
        run_one(q, client, bm25, corpus_tokens, chunk_ids, payloads_map, chat_history, i)

    print(f"\n{'='*66}")
    print("Timing test complete.")


if __name__ == "__main__":
    main()
