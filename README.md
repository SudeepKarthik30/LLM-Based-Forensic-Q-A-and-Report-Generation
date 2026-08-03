# Aeternus Forensis — AI-Powered Forensic Investigation Pipeline

A local, citation-enforced RAG system for digital forensic investigation. It ingests raw forensic artifacts (Windows Event Logs, packet captures, CSV/syslog exports), indexes them with hybrid dense \+ sparse retrieval, and lets an investigator ask natural-language questions and get answers that trace back to real, cited log entries — not hallucinated summaries.

Built as a research internship project at RV University. Everything — ingestion, retrieval, generation, the Streamlit UI, and report generation — runs locally against open-source models via Ollama and a local Qdrant instance. No data leaves the machine.

---

## What it does

Given raw forensic evidence (`.evtx`, `.pcap`, `.csv`, `.log`), the pipeline:

1. Parses every artifact into a unified event schema (Phase 1\)  
2. Embeds and indexes every event for hybrid semantic \+ keyword search (Phase 2\)  
3. Answers investigator questions with citation-checked, confidence-graded responses (Phase 2, query-time)  
4. Generates a structured Markdown \+ PDF investigation report from the session (Phase 4\)  
5. Exposes all of the above through a Streamlit web UI, with case tagging and persisted sessions (Phase 3\)

The project is designed around one non-negotiable constraint: **every factual claim in an answer must cite a real source event, and if it can't, the system says so instead of guessing.**

---

## Architecture

Raw artifacts (.evtx / .pcap / .csv / .log)

        │

        ▼

   parsers.py            Phase 1 — parse into a unified event schema

        │

        ▼

   store.py              Phase 2 — chunk, embed (nomic-embed-text), store in Qdrant, build BM25 index

        │

        ▼

   retriever.py           forensic relevance gate → dense search (Qdrant) → sparse search (BM25) → RRF fusion → evidence context

        │

        ▼

   llm.py                 citation-enforced answer generation (llama3:8b) → citation validation → confidence override

        │

        ├──► query.py            CLI investigation terminal

        ├──► app.py               Streamlit UI (ingest / investigate / overview / settings / report)

        └──► report\_generator.py  Phase 4 — Markdown \+ PDF report from the session

### Retrieval pipeline, step by step

1. **Forensic relevance gate** (`retriever.py`) — rejects off-topic questions before they reach the LLM. Two stages: a keyword-overlap check requiring at least 2 forensic-domain terms, with a BM25-score fallback for legitimate questions that don't happen to use those exact keywords.  
2. **Dense retrieval** — Qdrant cosine similarity search over `nomic-embed-text` embeddings (768-dim), with optional payload filters on EventID / source file.  
3. **Sparse retrieval** — BM25 keyword search over the same corpus, boosted with extracted IOCs (IPs, hashes, suspicious file extensions) pulled straight from the question text.  
4. **RRF fusion** — Reciprocal Rank Fusion merges the two ranked lists into one, so an event that's a strong keyword hit but a weak semantic match (or vice versa) doesn't get dropped.  
5. **Context assembly** — builds a numbered EVIDENCE block for the LLM, with explicit `<<<EVIDENCE_START>>>` / `<<<EVIDENCE_END>>>` delimiters to guard against prompt injection from attacker-controlled log content (e.g. a process literally named to look like an instruction).  
6. **Generation** — `llama3:8b-instruct-q4_K_M` at temperature 0, given the evidence, a retrieval-confidence hint, and the last two turns of conversation for follow-up continuity.  
7. **Citation validation** — every `[Source N]` in the answer is checked against the actual number of sources provided. Invalid citations are flagged inline as `[Source N — INVALID CITATION]` and the response is retried once.  
8. **Confidence override** — the LLM's self-reported `CONFIDENCE: HIGH/MEDIUM/LOW` can never exceed what the retrieval signal (top similarity score \+ source count) actually supports. This is enforced in code, not left to the model's judgment.

---

## Features

- **Multi-format ingestion**: `.evtx` (Windows Event Log), `.pcap`/`.pcapng` (network captures), `.csv`/`.log` (syslog-style exports), with format-specific field extraction for each  
- **Hybrid retrieval**: dense semantic search \+ BM25 keyword search, merged with RRF — catches both "what kind of attack is this" queries and exact-match queries like a specific IP or hash  
- **Citation-enforced answers**: every claim traces to a numbered source; hallucinated citations are detected and flagged, never silently passed through  
- **Grounded confidence**: HIGH/MEDIUM/LOW is computed from retrieval signals, not just asserted by the model  
- **Multi-turn memory**: follow-up questions ("what about that admin account?") get prior-turn context without re-running retrieval  
- **Prompt injection guarding**: evidence snippets are sanitized and delimiter-wrapped before being sent to the LLM  
- **Forensic relevance gate**: off-topic questions ("what's the capital of France?") are rejected before wasting an LLM call  
- **Automated report generation**: Markdown \+ PDF reports with an LLM-written executive summary and a programmatically-built (non-hallucinated) evidence inventory table  
- **Persisted investigation sessions**: every chat is saved to disk with a session ID, survives app restarts, and shows up in a sidebar history list  
- **Case tagging**: ingested events can be tagged with a case ID; legacy untagged data can be retroactively tagged from the UI  
- **Chain-of-custody audit log**: every question, answer, source list, gate decision, and confidence signal is logged with a UTC timestamp  
- **Crash-resilient embedding**: checkpointed embedding batches mean a crash at 80% doesn't mean re-embedding from zero  
- **Background-threaded query execution**: in the Streamlit UI, queries run in a worker thread so switching tabs mid-query doesn't kill it

---

## Tech stack

| Component | Choice |
| :---- | :---- |
| Embedding model | `nomic-embed-text` (768-dim) via Ollama |
| Generation model | `llama3:8b-instruct-q4_K_M` via Ollama |
| Vector database | Qdrant (local, cosine distance) |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Web UI | Streamlit |
| PDF generation | fpdf2 |
| EVTX parsing | python-evtx |
| PCAP parsing | dpkt |

Everything runs locally. No API keys, no cloud calls — that's a design constraint of the project, not an accident.

---

## Project structure

RV\_internship/

├── config.py               central configuration — all constants, paths, thresholds

├── config.toml              Streamlit theme config

├── parsers.py               Phase 1 — EVTX / PCAP / CSV / log parsing

├── store.py                 Phase 2 — chunking, embedding, Qdrant storage, BM25 index

├── retriever.py              Phase 2 — forensic gate, dense/sparse search, RRF fusion, context builder

├── llm.py                   citation-enforced LLM generation \+ validation

├── ingest.py                 CLI ingestion script (Phase 1 \+ Phase 2, end to end)

├── query.py                  CLI interactive investigation terminal

├── report\_generator.py       Phase 4 — Markdown \+ PDF report generation

├── app.py                    Streamlit UI — ingest / investigate / overview / settings / report

├── session\_store.py          persisted investigation sessions (JSON, disk-backed)

├── verify.py                  automated functional test suite

├── timed\_test.py              non-interactive timing/performance harness

├── pdf\_debug.py                standalone PDF renderer debug script

├── requirements.txt

└── .gitignore

---

## Setup

### Prerequisites

- Python 3.12  
- [Ollama](https://ollama.com) installed locally  
- Docker (for Qdrant), or use Qdrant's local-file mode (see below)

### 1\. Clone and install

git clone \<your-repo-url\>

cd RV\_internship

python \-m venv venv

venv\\Scripts\\activate        \# Windows

\# source venv/bin/activate   \# macOS/Linux

pip install \-r requirements.txt

### 2\. Start Qdrant

docker run \-p 6333:6333 qdrant/qdrant

Colab / no-Docker alternative: set `QDRANT_PATH` as an environment variable before importing `store.py`, and it switches to local-file mode automatically.

### 3\. Pull the Ollama models

ollama pull nomic-embed-text

ollama pull llama3:8b-instruct-q4\_K\_M

### 4\. Ingest forensic data

Either drop raw artifacts into `sample_data/` and run:

python ingest.py

or use the **Ingest** tab in the Streamlit UI (supports per-upload case tagging).

### 5\. Run it

CLI:

python query.py

Web UI:

streamlit run app.py

### 6\. Verify everything works

python \-X utf8 verify.py

Runs the full functional test suite: timestamp normalization, the forensic relevance gate, citation validation, audit logging, evidence inventory deduplication, RRF fusion correctness, mocked dense-search filtering, and an end-to-end mocked query run.

---

## Configuration

All tunable values live in `config.py`. The ones that matter most for tuning against real data:

| Setting | Default | Notes |
| :---- | :---- | :---- |
| `DENSE_TOP_K` / `SPARSE_TOP_K` | 20 / 20 | candidates pulled from each retrieval path before fusion |
| `RRF_TOP_N` | 3 | final number of sources sent to the LLM per query |
| `SNIPPET_CHARS` | 400 | evidence snippet length sent to the LLM per source |
| `CONFIDENCE_HIGH_THRESHOLD` | 0.62 | placeholder — calibrate against real dense score distribution after ingest |
| `CONFIDENCE_MED_THRESHOLD` | 0.55 | same |
| `BM25_FALLBACK_THRESHOLD` | 2.0 | placeholder — same calibration note |
| `MAX_RECORDS` | 50,000 | per-file ingestion cap (set to 0 to disable) |
| `LLM_TEMPERATURE` | 0.0 | deterministic generation |

The confidence and BM25 fallback thresholds are explicitly flagged in the code as placeholders that should be calibrated by running 10–15 sample queries against real ingested data and adjusting to match the actual score distribution — this hasn't been skipped, it's a documented manual step.

---

## Known limitations

These are documented in the code, not hidden:

- **In-memory payload cache**: all event payloads are loaded into memory at startup via Qdrant scroll. Works fine at 10k–50k event scale; would need per-query `retrieve()` calls instead of a full in-memory map to scale to millions of events.  
- **Snippet truncation**: evidence snippets sent to the LLM are capped at 400 characters, which reduces but doesn't eliminate the risk of truncating relevant fields on very verbose EventData payloads.  
- **Confidence thresholds are placeholders**: see Configuration above.  
- **Single-instance BM25 rebuild on startup**: BM25Okapi isn't reliably pickleable across library versions, so the raw token corpus is persisted and the index is refit at every session start (\~1–2s for 50k documents).

---

## Testing

`verify.py` covers:

- Timestamp normalization across multiple raw formats  
- Forensic relevance gate precision (tightened keyword set, no generic question words)  
- Citation validator: valid citations, hallucinated citations, mixed valid/invalid, empty-source edge case  
- Audit log creation and content  
- Evidence inventory deduplication logic  
- Config consistency across modules  
- RRF fusion — hand-computed expected rankings, dense-only and sparse-only edge cases  
- `dense_search` filter construction with mocked Qdrant  
- End-to-end `run_query` with mocked retrieval and LLM calls

`timed_test.py` is a separate, non-interactive harness for measuring retrieval and generation latency across repeated and varied queries — useful for isolating model-reload overhead from actual inference time.

---

## License

MIT — see [LICENSE](http://LICENSE).

---

## Author

Karthik Thirumalasetty — B.Tech CSE, IIIT Manipur  
