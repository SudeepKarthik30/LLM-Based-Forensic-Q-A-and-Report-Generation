# AI-Powered Forensic Investigation Assistant

This is a Q&A system for digital forensics. You feed it parsed forensic artifacts
(Windows Event Logs, network captures, CSV exports), it indexes them, and lets an
investigator ask plain-English questions about the incident. Every answer has to
point back to a specific log entry — if it can't find evidence, it says so instead
of guessing.

Built for an 8-week capstone project. Phase 1 and Phase 2 are done. This README
covers what exists right now and how to run it.

---

## What it actually does

1. Parses raw forensic files (.evtx, .pcap/.pcapng, .csv) into one consistent JSON format.
2. Loads that data into Qdrant (vector database) for semantic search, and also builds
   a BM25 keyword index for exact-match search (IPs, hashes, usernames, etc).
3. When you ask a question, it runs both searches, merges the results (RRF fusion),
   and hands the top matches to a local LLM running through Ollama.
4. The LLM has to cite which source ("[Source 1]", "[Source 3]", etc.) backs every
   claim it makes. The code checks those citations are real — if the model makes one
   up, it gets flagged in the output instead of silently passed through.
5. Report generation (Phase 4) is partially started — turns a Q&A session into a
   Markdown/PDF report.

Everything runs locally through Ollama. No API keys, no cloud calls, nothing leaves
the machine.

---

## Repo layout

Everything currently sits at the repo root, flat:

```
config.py             All settings live here. Model names, thresholds, file paths.
                       If you need to change anything, change it here, not inline
                       in other files.
parsers.py             Turns raw evtx/pcap/csv into the unified JSON schema.
store.py                Qdrant connection setup, plus saving/loading the BM25 index.
ingest.py                Embeds parsed records and pushes them into Qdrant, builds
                         the BM25 cache.
retriever.py             Dense search, sparse (BM25) search, RRF fusion, and the
                         confidence scoring logic.
llm.py                    The system prompt, the actual Ollama call, citation
                          checking, and confidence override logic.
query.py                   The interactive terminal you actually run to ask
                            questions.
report_generator.py         Phase 4 work — builds a report from chat history.
verify.py                    Some validation/sanity checks.
requirements.txt              Python packages needed.
```

Phase reports, the system design doc, and the mentor's original project plan aren't
in this repo — check the folder one level up from here, or ask whoever set this up
where they ended up.

---

## Getting it running

### What you need first
- Python 3.10 or newer
- Ollama installed and running (https://ollama.com)
- Qdrant running locally. Easiest way is Docker:
  ```
  docker run -p 6333:6333 qdrant/qdrant
  ```

### Pull the models
```
ollama pull nomic-embed-text
ollama pull llama3:8b-instruct-q4_K_M
```
If you end up using a different model or quantization, change `LLM_MODEL` in
`config.py`. Don't hardcode a model name anywhere else in the code.

### Install Python packages
```
pip install -r requirements.txt
```

### Get your sample data in place
Drop your raw .evtx / .pcap / .csv files into the folder `config.py` points to
under `INPUT_DIR`. Check that variable before running anything, since the exact
path depends on where you cloned this.

---

## Running it

**Parse the raw files:**
```
python parsers.py
```
Converts everything into structured JSON.

**Build the index:**
```
python ingest.py
```
Embeds the parsed data into Qdrant and builds the BM25 cache. If you need to
wipe and rebuild everything from scratch, run `python ingest.py --force` instead.

**Ask questions:**
```
python query.py
```
This drops you into an interactive terminal. Type a question, get an answer.
A couple of special commands work here too:
- type `report` to generate a report from everything you've asked so far
- type `exit` or `quit` to leave

---

## Why this isn't just a wrapper around a chatbot

The whole point of this project is that the LLM isn't allowed to just say whatever
sounds right. A few things enforce that:

- Every claim in an answer needs a `[Source N]` tag pointing at a real retrieved
  event. The code double-checks these are valid — see `validate_citations()` in
  `llm.py`. If the model cites something that doesn't exist, it gets flagged
  right there in the output, not buried somewhere.
- The confidence level (HIGH / MEDIUM / LOW) isn't something the model just decides
  on its own. It's computed from how strong the actual retrieval match was, and if
  the model tries to claim a higher confidence than the evidence supports, the code
  overrides it back down.
- If there isn't enough evidence to answer something, it says "insufficient evidence"
  instead of making something up.
- Questions that have nothing to do with forensics get rejected before they ever
  reach the LLM.

---

## Things that are known limitations right now, not hidden problems

- All the event data gets loaded into memory when you start `query.py`. This is
  fine up to something like 50,000 events. Past that, it needs to switch to
  fetching things from Qdrant on demand instead of holding everything in RAM.
- If you're running this on a laptop with no dedicated GPU, expect answers to take
  anywhere from 20 to 90+ seconds depending on what else your machine is doing.
  That's a hardware limitation, not something wrong with the code — a GPU would
  bring this down to a few seconds. Worth mentioning upfront if you're demoing this
  live so it doesn't look like it's frozen.
- The confidence thresholds in `config.py` are placeholder numbers right now. Once
  you've ingested real data, run a handful of test questions, look at the actual
  similarity scores being returned, and adjust the thresholds to match what you're
  actually seeing.

---

## What's coming next

Phase 3 is a Streamlit interface — a proper dashboard instead of a terminal, with
a case switcher so you can work across more than one forensic case, source
citations you can expand and collapse, and something that actually shows what
step the pipeline is on while you wait for an answer, instead of a plain spinner.

There's no login system planned. Given the timeline, a case dropdown gets you the
same practical benefit without the extra engineering a real auth system would need.

---

## Who to ask

If something in here doesn't make sense or doesn't match what you're seeing when
you run it, ask before assuming it's broken — there's a decent chance it's a config
path issue, not a code bug.
