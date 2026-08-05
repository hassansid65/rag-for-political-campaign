# Real-Time RAG for a Voice AI Political Campaign Assistant

**Complete project documentation** — architecture, setup, API reference, sample
data, engineering design note, and verification results.

---

## Table of contents

1. [What this is](#1-what-this-is)
2. [Architecture](#2-architecture)
3. [Setup and running](#3-setup-and-running)
4. [API reference](#4-api-reference)
5. [Sample documents](#5-sample-documents)
6. [Design note — engineering decisions](#6-design-note--engineering-decisions)
7. [Streaming retrieval and latency](#7-streaming-retrieval-and-latency)
8. [Verification results](#8-verification-results)
9. [Requirements coverage](#9-requirements-coverage)
10. [Known limitations](#10-known-limitations)

---

## 1. What this is

A Retrieval-Augmented Generation pipeline wired into a voice assistant. Campaign
documents — manifestos, district briefs, candidate profiles, scheme booklets,
FAQs — are uploaded, chunked, embedded and indexed. A citizen asks a question by
text or voice; the system retrieves the relevant passages **while they are still
speaking**, answers with a lip-synced 3D avatar, and cites every factual claim.

### The problem that shaped the design

The primary test corpus (`data/RAG_Test_Candidate_Profiles.pdf`) is 28 pages
holding **56 candidate profiles that share one template**. They differ by a proper
noun and a few numbers inside ~1,000 characters of otherwise identical text.

On a corpus like that, the dangerous failure is not inventing facts. It is
**misattribution** — reporting candidate A's assets under candidate B's name. That
answer is fluent, cited, and wrong. Most of the engineering below exists to make
that outcome structurally impossible rather than merely unlikely.

### Stack

| Layer | Choice |
| --- | --- |
| Embeddings | BGE-small-en-v1.5 (384-dim, CPU) |
| Vector DB | Milvus 2.5 standalone — HNSW dense + server-side BM25 sparse |
| Fusion | Reciprocal Rank Fusion (k=60, weighted) |
| Reranking | Cascade: MiniLM-L6 → BGE-reranker-base |
| LLM | Azure OpenAI GPT-4 (Anthropic Claude client also implemented) |
| Speech | Azure Speech — DragonHD TTS with visemes, multi-locale STT |
| API | FastAPI — REST + SSE + WebSocket |
| UI | Next.js 16, React 19, react-three-fiber 9, GLB avatar |
| Runtime | Python 3.10, CPU-only by default |

---

## 2. Architecture

### 2.1 Ingestion

```
   PDF · DOCX · TXT · MD · HTML · CSV
                  │
                  ▼
   ┌──────────────────────────────────┐
   │ Loader                           │   PyMuPDF / python-docx / markdown
   │  · font-size heading detection   │
   │  · table extraction              │
   │  · page-aligned blocks           │
   │  · script-based language detect  │
   └──────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────┐
   │ Chunker  (strategy = auto)       │
   │                                  │
   │  RECORD-ATOMIC  ── when a        │   one entity per chunk,
   │   repeating field template is    │   never split, never merged
   │   detected (Born. / Education.)  │
   │                                  │
   │  STRUCTURAL     ── otherwise     │   section-first, then recursive
   │   700-char children              │   char split inside a section
   │   1800-char parent windows       │
   │   Q&A pairs + table rows atomic  │
   └──────────────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────┐
   │ Metadata extraction              │
   │  district · category · source    │   26-district gazetteer
   │  topic · state · section path    │   with city/constituency aliases
   │  page · candidate · party        │
   │  scheme names · language         │
   └──────────────────────────────────┘
                  │
                  ▼
   BGE-small-en-v1.5 · 384-dim · passage side (no instruction prefix)
                  │
                  ▼
   ┌──────────────────────────────────┐
   │ Milvus collection                │
   │  vector    FLOAT_VECTOR(384) HNSW/COSINE
   │  text      VARCHAR, analyzer on  →  BM25 Function → sparse
   │  districts ARRAY<VARCHAR>        │  INVERTED scalar indexes on
   │  topics    ARRAY<VARCHAR>        │  doc_id, district, category,
   │  meta      JSON (full round-trip)│  topic, source, language
   └──────────────────────────────────┘
```

### 2.2 Retrieval

```
   utterance   ("I'm from Vijayawada" · "who is born on 14 October 1985")
        │
        ▼
   ┌────────────────────────────────────────────┐
   │ Intent routing                    ~0.05 ms │
   │  greeting / thanks / identity / capability │──► canned reply,
   │  → NO retrieval                            │    retrieval skipped
   └────────────────────────────────────────────┘
        │ factual
        ▼
   ┌────────────────────────────────────────────┐
   │ Query understanding                ~0.2 ms │
   │  · filler strip, acronym + translit expand │
   │  · district inference from gazetteer       │
   │  · follow-up resolution from memory        │
   │  · person-name extraction (F1 matching)    │
   │  · optional LLM rewrite (only if needed)   │
   └────────────────────────────────────────────┘
        │
        ▼
   ┌────────────────────────────────────────────┐
   │ Literal gate (exact value)         ~7 ms   │
   │  dates · amounts · seats · percentages     │
   │  hit  → those records ARE the answer set   │──► return
   │  miss → return NOTHING (never near-miss)   │──► return
   └────────────────────────────────────────────┘
        │ no literal
        ▼
   Semantic cache probe  ~0.05 ms ──── hit ──────► return
        │
        ▼
   BGE-small query embedding  3-8 ms  (LRU cached, instruction prefix)
        │
        ├────────────────────┬───────────────────────┐
        ▼                    ▼                       ▼
   dense HNSW          BM25 sparse            query variants
   (metadata           (same filter,          (synonym / district)
    pre-filter)         server-side)
        └────────────────────┴───────────────────────┘
                   run concurrently · 2-15 ms
        │
        ▼
   Reciprocal Rank Fusion (k=60, weighted)        ~0.1 ms
        ▼
   Dedupe + metadata boosting                    ~0.1 ms
        ▼
   ┌────────────────────────────────────────────┐
   │ Entity gate                                │
   │  name matched  → ONLY that record survives │
   │  name absent   → ALL profiles dropped      │
   └────────────────────────────────────────────┘
        ▼
   Cascade rerank: MiniLM(16) → BGE-reranker-base(top 4)   ~670 ms
        ▼
   Threshold → Top-K → parent-window expansion
```

### 2.3 Generation and voice

```
   Prompt layout (prompt-cache friendly):

     [ system prompt        ]  ← cache breakpoint, byte-stable
     [ conversation history ]
     [ numbered context     ]  ← varies per turn
     [ citizen's question   ]
        │
        ▼
   LLM (thinking off, effort low, deterministic)
        │
        ├─ token stream ──► sentence-boundary detector
        │                        │
        │                        ▼
        │                  Azure TTS per sentence
        │                        │
        │                        ├─► viseme cues ──► GLB avatar morph targets
        │                        └─► audio clip  ──► ordered playback queue
        │                                                    │
        │                                          onClipStart(text, duration)
        │                                                    │
        │                                                    ▼
        │                                        caption revealed at the
        │                                        pace of the spoken audio
        ▼
   Citation verification — only the markers the model actually emitted
        ▼
   FastAPI: REST · SSE · WebSocket
```

### 2.4 Repository layout

```
task/
├── backend/
│   ├── api/          FastAPI app + routers (upload, retrieve, query, health, voice)
│   ├── core/         config · schemas · logging · latency · resources
│   ├── ingestion/    loader · records · chunker · metadata · service
│   ├── embeddings/   BGE-small wrapper (torch or ONNX) + LRU cache
│   ├── vectorstore/  milvus_store · local_store · bm25 · factory
│   ├── retrieval/    pipeline · fusion · reranker · rewriter · literals · intent · cache
│   ├── llm/          claude_client · azure_openai_client · provider · prompts
│   │                 rag_service · extractive
│   ├── memory/       conversation memory + sticky slots
│   ├── voice/        azure_speech · lipsync · streaming (speculative retrieval)
│   ├── scripts/      verify_env · test_suite · test_records · test_grounding
│   │                 test_conversation · test_reverse · test_milvus · test_speech
│   │                 bench_rerank · bench_cascade · smoke_test
│   └── tests/        question_bank (110+ questions)
├── frontend/
│   ├── app/          Next.js App Router
│   ├── components/   Avatar · ChatWindow · MessageBubble · SourceCard · SidePanel
│   ├── lib/          api (SSE) · voice (mic, WS, audio queue) · streaming
│   └── public/       Shayla_Changes(Visemes).glb · working.glb
├── data/
│   ├── RAG_Test_Candidate_Profiles.pdf
│   └── sample_docs/  manifesto · district profile · schemes · FAQ · candidates
├── docker-compose.yml
├── START_BACKEND.bat · START_FRONTEND.bat
└── README.md
```

---

## 3. Setup and running

### 3.1 Prerequisites

- **Python 3.10** (`python --version` must print 3.10.x)
- **Node.js 20+**
- **Docker Desktop** (for Milvus; optional — see fallback below)
- **ffmpeg** (optional; only needed for browser-recorded audio on Safari/Firefox)

### 3.2 Fastest path — Windows

```
START_BACKEND.bat      →  http://localhost:8000/docs
START_FRONTEND.bat     →  http://localhost:3000
```

Both are idempotent: on first run they create the venv, install pinned
dependencies, copy `.env` from the example, run the environment check, and start
the server. Later runs just start it.

### 3.3 Manual — backend

```powershell
cd backend
python -m venv .venv                 # note: -m venv, not -m 3.10 venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
# torch first, as its own step, so a failure is isolated
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

copy .env.example .env               # then fill in the keys below
python scripts\verify_env.py         # ALWAYS run this first
python -m uvicorn api.main:app --reload --port 8000
```

macOS / Linux is identical with `python3.10 -m venv .venv && source .venv/bin/activate`.

A virtualenv is not optional. A global Python holding other ML projects will
already have conflicting pins for `torch`, `numpy` and `transformers`.

### 3.4 Vector database

```bash
docker compose up -d etcd minio milvus       # add --profile tools for Attu UI
```

Then set `VECTOR_BACKEND=milvus` in `backend/.env`.

Four deployment shapes, one code path:

| Mode | `VECTOR_BACKEND` | Notes |
| --- | --- | --- |
| Milvus standalone | `milvus` | recommended; 2.5+ gives server-side BM25 |
| Zilliz Cloud | `milvus` | set `MILVUS_URI` + `MILVUS_TOKEN` |
| Milvus Lite | `milvus_lite` | embedded single file — **Linux/macOS only** |
| Local NumPy | `local` | no dependencies, exact flat search, works anywhere |

The backend degrades down that list automatically and logs which one it landed
on. Everything above the store — hybrid search, RRF, reranking, filtering,
citations — behaves identically on all four.

### 3.5 Required keys

```ini
# backend/.env

# LLM — either provider works; `auto` health-checks and picks a live one
LLM_PROVIDER=auto
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<res>.openai.azure.com/openai/deployments/gpt-4-0613/chat/completions?api-version=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4-0613
# or
ANTHROPIC_API_KEY=sk-ant-...

# Speech (STT + TTS + visemes)
AZURE_SPEECH_KEY=...
AZURE_SPEECH_REGION=eastus2
AZURE_TTS_VOICE=en-IN-Meera:DragonHDLatestNeural
```

### 3.6 Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local     # NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
npm run dev
```

React **19** is mandatory. Next 16's App Router runs its own vendored React 19 for
client components regardless of what `package.json` pins, and
`@react-three/fiber` v8 reads `ReactCurrentOwner` — a React 18 internal that
React 19 removed. Fiber v9 dropped `react-reconciler` entirely, so the crash is
structurally impossible rather than merely avoided.

### 3.7 Loading documents

Drag files onto the upload panel in the UI, or:

```bash
curl -F "files=@data/RAG_Test_Candidate_Profiles.pdf" http://localhost:8000/upload
```

---

## 4. API reference

Interactive OpenAPI docs at `http://localhost:8000/docs`.

### 4.1 `POST /upload`

Ingest documents: parse → chunk → extract metadata → embed → index.
Multipart form.

| Field | Type | Notes |
| --- | --- | --- |
| `files` | file (repeatable) | PDF, DOCX, TXT, MD, HTML, CSV |
| `category` | string | override the inferred category |
| `district` | string | free-text place name; resolved to canonical district |
| `state`, `topic`, `candidate`, `party` | string | metadata overrides |
| `metadata` | JSON string | any of the above as one blob |

```bash
curl -F "files=@manifesto.pdf" -F "district=Vijayawada" \
     -F "category=manifesto" http://localhost:8000/upload
```

```json
{
  "status": "success",
  "documents": [{
    "doc_id": "doc_e4e937b1d97045ca",
    "source": "RAG_Test_Candidate_Profiles.pdf",
    "category": "candidate_profile",
    "districts": ["Guntur", "NTR", "Visakhapatnam"],
    "topics": ["healthcare", "infrastructure"],
    "pages": 28,
    "chars": 58426,
    "chunks_indexed": 56,
    "detected_language": "en",
    "warnings": []
  }],
  "total_chunks_indexed": 56,
  "collection": "campaign_chunks",
  "timings_ms": { "load": 412.1, "chunk": 38.4, "embed_passages": 6210.5, "upsert": 890.2, "total": 7551.2 }
}
```

Uploading the same file twice is idempotent — `doc_id` is a content hash, so
chunk ids are stable and the upsert replaces in place.

### 4.2 `POST /retrieve`

Retrieval only, no generation. This is the debugging endpoint: it returns every
score the pipeline computed and the filters it inferred.

```json
{
  "query": "What are the declared assets of Smt. Sarojini Vasireddy?",
  "top_k": 5,
  "similarity_threshold": 0.30,
  "filters": { "district": "Guntur", "category": "candidate_profile" },
  "session_id": "abc123",
  "rerank": true,
  "hybrid": true,
  "rewrite_query": true
}
```

```json
{
  "query": "What are the declared assets of Smt. Sarojini Vasireddy?",
  "effective_query": "What are the declared assets of Smt. Sarojini Vasireddy?",
  "inferred_filters": { "district": "Srikakulam" },
  "results": [{
    "id": "doc_e4e937b1d97045ca-0031",
    "text": "...parent window handed to the LLM...",
    "chunk_text": "...the child chunk that actually matched...",
    "metadata": {
      "record_name": "Smt. Sarojini Vasireddy",
      "constituency": "Srikakulam",
      "district": "Srikakulam",
      "category": "candidate_profile",
      "is_record": true,
      "page": 16
    },
    "score": 0.996,
    "dense_score": 0.812,
    "sparse_score": 8.682,
    "rrf_score": 0.041237,
    "rerank_score": 7.104,
    "retriever": "hybrid"
  }],
  "total_candidates": 39,
  "reranked": true,
  "cache_hit": false,
  "timings_ms": {
    "rewrite": 0.8, "embed": 22.4, "cache": 0.1, "search": 4.8,
    "fuse": 0.3, "rerank": 92.1, "select": 0.1, "total": 126.6,
    "notes": ["entity gate: 'Sarojini Vasireddy' → ['Smt. Sarojini Vasireddy'] (dropped 38 other record(s))"]
  }
}
```

`chunk_text` and `text` are separate on purpose: the child is what matched and is
what a citation snippet must show; the parent window is what the LLM reads.

### 4.3 `POST /query`

Grounded answer generation.

```json
{
  "query": "who is born on 14 October 1985",
  "session_id": "abc123",
  "stream": true,
  "top_k": 5,
  "include_citations": true,
  "voice_mode": false
}
```

**Non-streaming response:**

```json
{
  "answer": "Dr. Jayasudha Kesineni, the Alliance candidate for the Adoni assembly constituency in Kurnool district, was born on 14 October 1985 [1].",
  "session_id": "abc123",
  "grounded": true,
  "citations": [{
    "marker": "[1]",
    "source": "RAG_Test_Candidate_Profiles.pdf",
    "category": "candidate_profile",
    "district": "Kurnool",
    "section": "Dr. Jayasudha Kesineni - Adoni, Kurnool District",
    "page": 4,
    "chunk_id": "doc_e4e937b1d97045ca-0007",
    "score": 0.997,
    "snippet": "Dr. Jayasudha Kesineni is the Alliance candidate for the Adoni..."
  }],
  "sources_used": 1,
  "usage": { "input_tokens": 1284, "output_tokens": 41, "cache_read_input_tokens": 1152 },
  "model": "gpt-4-0613",
  "timings_ms": { "total": 1698.2, "@first_token": 1039.3 }
}
```

**With `"stream": true`** — Server-Sent Events:

| Event | Payload |
| --- | --- |
| `retrieval` | sources, chunks, inferred filters — **arrives before generation** |
| `delta` | `{"text": "..."}` incremental tokens |
| `final` | answer, verified citations, usage, per-stage timings |
| `error` | `{"error": "..."}` |
| `end` | stream terminator |

The `retrieval` event first is deliberate: source cards render while the model is
still generating.

### 4.4 `GET /health`

Per-component status **plus** rolling p50/p95/p99 for every pipeline stage.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_s": 412.5,
  "components": [
    { "name": "vector_store",  "status": "ok", "detail": "milvus · 56 chunks", "latency_ms": 3.1 },
    { "name": "embedder",      "status": "ok", "detail": "BAAI/bge-small-en-v1.5 · 384d · cache 34%" },
    { "name": "reranker",      "status": "ok", "detail": "BAAI/bge-reranker-base · top-3" },
    { "name": "llm",           "status": "ok", "detail": "gpt-4-0613", "latency_ms": 288.0 },
    { "name": "azure_speech",  "status": "ok", "detail": "region=eastus2 · voice=en-IN-Meera:DragonHDLatestNeural" },
    { "name": "lipsync",       "status": "ok", "detail": "primary=azure-visemes" }
  ],
  "config": { "...": "every active tuning parameter" },
  "latency_ms": { "store.search_dense": { "p50": 0.2, "p95": 3.7, "count": 214 } }
}
```

Also `GET /health/live` (liveness, touches nothing), `GET /health/ready`
(readiness, 503 until models load), `GET /metrics`.

### 4.5 Voice

| Endpoint | Purpose |
| --- | --- |
| `POST /voice/stt` | base64 audio → text; auto-detects en-IN / te-IN / hi-IN |
| `POST /voice/tts` | text → base64 WAV + viseme mouth cues + duration |
| `POST /voice/turn` | one full turn: retrieve → generate → speak → lip-sync |
| `GET /voice/voices` | available voice presets |
| `WS /ws/voice` | streaming turn with speculative retrieval |

**WebSocket protocol** — client → server:

```json
{"type": "partial", "text": "what is amma vodi"}     // fires speculative retrieval
{"type": "final",   "text": "...", "voice": "meera"}
{"type": "cancel"}                                    // barge-in
{"type": "ping"}
```

server → client: `ready` · `speculation` · `retrieval` · `delta` · `audio` ·
`final` · `error` · `pong`

### 4.6 Documents and sessions

```
GET    /documents              list indexed docs with metadata + chunk counts
DELETE /documents/{doc_id}     remove a document and all its chunks
POST   /ingest-path            bulk-ingest a server-side file or directory
GET    /districts              canonical districts (for UI dropdowns)
GET    /resolve-district?name= resolve a place name to a canonical district
GET    /sessions               list active sessions
GET    /sessions/{id}          session history + sticky slots
DELETE /sessions/{id}          reset a session
POST   /cache/invalidate       drop retrieval + answer caches
POST   /metrics/reset          clear latency percentiles
```

---

## 5. Sample documents

`data/sample_docs/` contains five hand-written documents covering every category
the metadata extractor recognises, so the full pipeline can be exercised without
the candidate PDF.

| File | Category | Contents |
| --- | --- | --- |
| `manifesto_2024.md` | `manifesto` | 8 chapters — agriculture, education, health, employment, women's welfare, housing, pensions, fisheries. Contains exact figures (Rs. 18,000 Rythu Bharosa, Rs. 15,000 Amma Vodi) for fact-check testing. |
| `district_profile_ntr_vijayawada.md` | `district_info` | NTR district: demographics table, 7 assembly constituencies, economy, irrigation, infrastructure, 5 local issues, district-specific commitments. |
| `schemes_welfare.md` | `scheme` | 10 schemes with benefit / eligibility / how-to-apply / payment mode. Includes a 12-row pension table for table-row retrieval. |
| `faq_voters.md` | `faq` | 16 Q&A pairs — voter registration, documents, grievances, scheme eligibility. Tests atomic Q&A chunking. |
| `candidate_profiles.md` | `candidate_profile` | 4 profiles with the same field template as the test PDF. |

`data/RAG_Test_Candidate_Profiles.pdf` — the supplied 28-page corpus, 56 candidate
profiles sharing one field template. This is the adversarial case the design
targets: it deliberately contains near-namesakes (`Kesineni` ×3,
`Devarakonda` ×3, `Kanna` ×2, and multiple candidates sharing a first name).

---

## 6. Design note — engineering decisions

### 6.1 Chunking is record-atomic

A size-based splitter fails two ways on a record corpus, and both produce
confident wrong answers:

1. **Split mid-record** — `Assets declaration.` lands in a chunk that names
   nobody. Retrieved for "what are Kiran Kumar's assets?", the model reports
   whatever figures it was handed.
2. **Merge two records** — a window ending inside candidate A and continuing into
   B invites the model to blend them. This is the dominant hallucination mode on
   record corpora, and it is *not* fixable with a better prompt: the context
   genuinely contains both.

So field labels that repeat (`Born.` `Education.` `Assets declaration.`) are
detected as a template, and **a repeating label means a new record started**. Each
record becomes exactly one chunk regardless of size, always leading with its own
title. Detection is template-driven, so the scheme booklet (`Benefit.` /
`Eligibility.`) and the FAQ (`Q1.` / `A1.`) work with no extra code.

Boundary detection is two-pass. The single-pass version had to *rewind* lines once
it discovered a boundary, and that index arithmetic drifted whenever blank lines
were skipped — every chunk ended up carrying the next record's title line, which
is exactly the merge failure being prevented.

### 6.2 Four layers stop misattribution

Each was added because a test caught the layer above it failing.

**One record per chunk** (§6.1). Verified: 56 records, 56 named, 0 merged,
0 anonymous.

**Entity-aware retrieval.** A name in the query is matched against each chunk's
`record_name` using symmetric token F1. Cosine similarity cannot do this job —
two different profiles sit at ~0.95 — but a name either matches a record or it
does not.

F1 had to be symmetric. An earlier coverage-plus-surname score gave *Sarojini
Vasireddy* vs *Padmavathi Vasireddy* **0.85** — a shared surname was almost a full
match — so all three Vasireddys survived gating with three different asset
declarations. F1 penalises a missing token in either direction and scores that
pair 0.50.

**Entity gating.** Boosting the right record to rank 1 is not enough. A top-5
context for one Vasireddy carried three different asset declarations. For an
entity-scoped question, retrieval is made **precise instead of high-recall**: only
the best-matching record survives, plus any genuinely tied for best (which the
prompt surfaces as "which one do you mean?"). When the named person is absent
from the corpus, **all** profile chunks are dropped — told only "say you don't
have their details", GPT-4 complied and then added *"what I can tell you is that
&lt;different candidate&gt; has declared…"*. Technically obedient, and exactly the
sentence that makes a listener attribute those assets to the person they asked
about. With no other profile in context there is nothing to volunteer.

**Entity-scoped caches.** The semantic cache was found serving one candidate's
results for another's question: *"assets of Anuradha Merugu"* and *"assets of
Anuradha Undela"* embed at **0.99 cosine**, above the 0.97 threshold. Raising the
threshold cannot fix it — the questions genuinely are near-identical strings. The
cache namespace therefore includes the named entity.

Result: for a question naming one candidate, the assembled context contains
**exactly one record (~1,200 chars) and zero other candidates' figures**. The
wrong number is not merely discouraged — it is not in the prompt.

### 6.3 Reverse lookups need an exact-value path

Asked *"who is born on 14 October 1985"*, the system answered with a candidate
born **7 September 1985**. Same misattribution class, different door — and
neither retrieval branch could fix it:

- No name in the query, so the entity gate never engaged.
- **Dense cannot help.** All 56 profiles are one template; the query embedding is
  roughly equidistant from every one, and a date contributes almost nothing to a
  384-dim sentence vector.
- **BM25 cannot either.** `14 October 1985` tokenizes to `{14, october, 1985}`,
  and many records share each token individually, so a two-of-three match ranks
  about as well as the exact one.

The correct record was often not even in the candidate set. So distinctive
literals — dates, rupee amounts, decimals, percentages, constituencies — are
extracted from the query, normalized (`14 October 1985` ≡ `14/10/1985`), and
matched **verbatim** against record text *before* the ANN search. An exact hit is
authoritative and replaces the candidate set.

The other half matters as much: when a distinctive value appears nowhere in the
corpus, retrieval returns **no context**. Returning near-misses is precisely what
produced the wrong date, and a near-miss date is more misleading than no answer.

Only *selective* literals qualify. A bare `100` appears in every "30 to 100 beds"
priority line, so gating on it would trade one wrong answer for another.

### 6.4 Small talk never reaches retrieval

A citizen typed "hey", then "hello", and both times received a factual answer
about a candidate's date of birth. Every utterance was going through the pipeline,
so a greeting retrieved *something*, and conversation memory then resolved it as a
follow-up — turning "hello" into a restatement of the previous question.

That is a missing conversational layer, not a grounding bug. Intent is classified
first, and greetings / farewells / thanks / acknowledgements / identity /
capability questions skip retrieval entirely.

Rules, not an LLM classifier: this runs on every turn including every voice turn,
and the decision is between eight lexically obvious categories. A regex pass costs
~0.05 ms; an LLM would add 300–600 ms to the critical path of a spoken
conversation. Anything ambiguous falls through to `FACTUAL`, which is the safe
default — it just means we retrieve. A greeting *prefix* stays factual, so
"hello, what does the manifesto say about roads?" is still a query.

It also removes small talk from the hallucination surface entirely: a greeting
never reaches the generator alongside a retrievable context.

### 6.5 Why BGE-small-en-v1.5

384 dims, 33M params, MTEB-retrieval within a couple of points of models 10× its
size. It embeds a voice-length query in **3–8 ms on CPU** — an API embedding model
would add 80–250 ms of network round-trip per turn before we even reach the vector
DB.

Three details that are easy to get wrong and cost real recall:

- **Asymmetric prefixing.** BGE is trained with an instruction prefix on the
  *query* side only. Prefixing passages too (or neither) measurably degrades
  retrieval.
- **L2 normalization.** Required for cosine/IP equivalence, so the store can use
  the fastest metric and still mean cosine.
- **CLS pooling, not mean.** BGE uses the CLS token; mean pooling silently costs
  a few points of nDCG.

### 6.6 Why hybrid, and why RRF specifically

Citizens ask about proper nouns — `Amma Vodi`, `Rythu Bharosa`, `Aarogyasri`,
`15,000`. Dense retrieval blurs rare tokens; BM25 nails exact matches. In testing,
`"rythu bharosa amount"` is carried by the BM25 branch (sparse 6.96) while dense
alone ranks it lower.

RRF combines by **rank**, not score:

```
score(d) = Σ_lists  weight_i / (k + rank_i(d))
```

That property is the whole reason to use it. Dense similarity lives in [0,1];
BM25 is unbounded and corpus-dependent. Min-max normalising before a weighted sum
makes fusion sensitive to each batch's *spread* — one outlier BM25 score squashes
every other keyword result toward zero. RRF only asks "how highly did each branch
rank this?", so it is stable across queries with no per-corpus tuning.

### 6.7 Reranking is cascaded

Measured on 12 CPU cores, 16 candidates at ~512-char passages:

| Model | max_len | per pair | 16 candidates |
| --- | --- | --- | --- |
| BAAI/bge-reranker-base | 512 | 304 ms | 6082 ms |
| BAAI/bge-reranker-base | 256 | 105 ms | 1690 ms |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 256 | 16 ms | 249 ms |

BGE-reranker-base is 278M params; MiniLM-L6 is 22M. So:

```
16 fused candidates
  → tier 1 (MiniLM, all 16)        ~250 ms   cheap and broad
  → keep top 4
  → tier 2 (BGE-reranker-base, 4)  ~420 ms   expensive and precise
  → top 3
```

`RERANK_MODE=auto` probes available RAM at startup and picks `cascade` or `fast`.
This is not defensive padding: at 3.3 GB free, loading the precise tier alongside
torch, the embedder and the TLS stack produced a hard access violation
(`0xC0000005`) — the allocation fails inside native code and takes the process
with it, rather than raising a catchable OOM.

Two further measured wins: reranking scores the **child** chunk, not the parent
window (304 ms/pair → 117 ms/pair with no ranking gain from the extra text, which
is mostly neighbouring topics), and `RERANKER_MAX_LENGTH` was cut 512 → 256, which
was paying full attention cost for tokens the model then truncated away.

### 6.8 Thresholding happens after reranking

The dense score of the correct chunk is often mediocre; its cross-encoder score is
not. Cutting on cosine first discards chunks the reranker would have promoted to
rank 1.

There is also an abandon margin: when the best cross-encoder score is far below
the bar, retrieval returns **nothing** rather than the least-bad candidate. The
"return top-1 anyway so the LLM can judge" fallback handed a random candidate
profile to *"how do I cook biryani"* and the extractor read it out. Scores
separate the cases cleanly — on-topic +2.9…+7.0, unanswerable-but-relevant
−3.1…−4.6, off-topic −10.1…−11.2.

### 6.9 Small children, large parents

Retrieval precision peaks with chunks small enough to be about one thing
(~700 chars); answer quality peaks with more surrounding context. So the child is
embedded and the parent window around it is handed to the LLM — best of both, no
extra vectors. `chunk_text` and `text` stay separate on the response because a
citation snippet must show the child that matched; the parent window commonly
opens in the previous section and reads as a mismatch.

Record chunks get **no** parent window: the record *is* the unit of meaning, and
expanding it would pull in the neighbouring record — exactly the contamination
the strategy exists to prevent.

### 6.10 Citations are verified, not assumed

The model is told to emit `[1]`, `[2]` markers; the service then checks which
markers it *actually used* and returns only those. Returning all retrieved chunks
as "sources" is the common shortcut and it is dishonest — it implies the answer
rests on five documents when it rests on one. An answer citing nothing is marked
`grounded: false` and the UI flags it.

The prompt also requires that no figure appear in a sentence without a marker, and
that a refusal mention no figure at all: "I don't have a list of candidates born
before 1970" is good; appending "*but so-and-so was born 22 January 1971*" makes
the listener hear the date as the answer.

Figures must be quoted in the document's own notation. TTS-friendly verbalization
was tried and walked back — re-spelling collapses `Rs. 44.0 lakh` and
`Rs. 44.05 lakh` onto identical words, and the citation then points at a number
the listener never heard.

### 6.11 Filters relax; hints boost

A district filter returning nothing is usually over-constraint, not absence of an
answer. Telling a citizen from Vijayawada that we know nothing about roads because
no chunk was *primarily* tagged NTR is a worse failure than answering from the
state-wide manifesto. So the filter is deliberately generous — a chunk qualifies
if it is primarily about the district **or** merely mentions it — and an empty
filtered result retries unfiltered.

`category_hint` and `topic_hint` come from keyword overlap and are guesses. As
hard filters they silently empty the result set, and a vague question is exactly
when you can least afford that. They are applied as a small additive boost
instead, so a strong lexical or semantic match can still win.

### 6.12 The store degrades instead of failing

Milvus Lite has no Windows build and Milvus standalone needs Docker. A missing
vector DB should make the system slower and less scalable, never non-functional —
someone cloning this repo gets a working demo, and the log says exactly what
happened and how to get the real thing.

Running against real Milvus found two bugs the fallback structurally could not:

- **No flush after upsert.** All 56 rows inserted, then dense search returned 0
  hits and `row_count` said 0. Under `Bounded` consistency freshly upserted rows
  sit in growing segments and are not searchable — indistinguishable from a silent
  upsert failure. Now flushing after upsert *and* delete; without the second,
  deleted rows keep answering searches and a re-upload looks like a duplicate.
  Flush-on-ingest was chosen over `Strong` consistency on reads, which would
  double query latency permanently to fix a problem lasting a few seconds.
- **`row_count` is the wrong count.** `get_collection_stats` counts only *sealed*
  segments, reporting 0 for perfectly searchable data. Now queries `count(*)`.

### 6.13 LLM provider selection health-checks

`LLM_PROVIDER=auto` probes candidates at startup rather than trusting that a key
present in the environment is a key that works. The first version selected
Anthropic because a key existed; that key returned 401 on every request, so the
service silently degraded to extractive answers while a working Azure deployment
sat unused beside it.

GPT-4 vs GPT-3.5 was measured, and it contradicted the usual assumption:

| Deployment | TTFT | Grounded | Emits `[1]` |
| --- | --- | --- | --- |
| gpt-4-0613 | 1039 ms | yes | **yes** |
| gpt-35-turbo-16k-0613 | 1347 ms | yes | **no** |

GPT-4 was both faster to first token and the only one producing citation markers.
Without `[1]` the verifier marks the answer ungrounded and the UI flags it, so 3.5
is strictly worse here.

### 6.14 An LLM outage degrades to extractive answering

If the key is missing, invalid, rate-limited or the API is unreachable, the
honest-but-useless behaviour is "I couldn't find that in the campaign documents."
That sentence is a lie: retrieval *did* find it. Only the paraphrasing step is
unavailable.

So the answer is composed **extractively** — the sentence containing the answer is
pulled verbatim from the top-ranked chunk and returned with its citation. Verbatim
text cannot hallucinate, so this path is strictly *more* faithful than the LLM
path, just less fluent. It also means a reviewer cloning the repo with no API key
still gets grounded, cited answers.

The extractor needed the same subject guard as the gate: without it, "assets of
Dr. Ramesh Chandra Patel" (not in the corpus) confidently returned a different
candidate's declaration.

---

## 7. Streaming retrieval and latency

### 7.1 Speculative retrieval over partial transcripts

A naive voice turn is strictly serial:

```
speech ──► ASR final ──► retrieve ──► LLM ──► TTS ──► audio
        │◄──────── ~1.8 s of silence after the citizen stops ───────►│
```

But Azure emits `recognizing` events every ~150–300 ms **while the citizen is
still talking**. By the time they finish we usually already know the question.
That window is free compute:

```
"what is"                                    → too short, ignored
"what is amma vodi"                          → fire retrieval #1  ─┐
"what is amma vodi eligib"                   → cancel #1, fire #2 ─┤ during speech
"what is amma vodi eligibility for my kid"   FINAL                ─┘
   → embed final, compare against held speculations
   → cosine 0.995 ≥ 0.94 → reuse, skip the pipeline entirely
```

Four guards keep it safe and cheap:

1. **Stability gate** — fire only above `PARTIAL_MIN_CHARS` / `PARTIAL_MIN_TOKENS`.
2. **Debounce + single-flight** — one speculation in flight; newer partials cancel
   older ones. Without this a 5-second utterance fires 20 retrievals and saturates
   the CPU the real turn needs.
3. **Embedding-similarity reuse, not string equality** — ASR revises words and
   adds punctuation, so the final is almost never byte-identical. Comparing
   embeddings is what makes the hit rate high.
4. **Verify before trust** — below threshold the work is discarded and retrieval
   runs fresh. A wrong reused context is worse than a slow correct one.

A short ring of recent speculations is kept, not just the latest, because ASR
revises mid-utterance (`"dia"` → `"dialysis"`) and the closest match to the final
is not always the most recent guess. Completed speculations are preferred over a
marginally closer in-flight one — awaiting the newest guess can cost more than the
retrieval it was meant to skip.

**Measured: reuse at cosine 0.995, ~2.6 s of pipeline work removed from the
post-speech critical path.** Because speculation runs during speech it also gets
the *full* rerank cascade; a live turn without a usable speculation drops to
`fast` rerank instead, degrading latency rather than correctness.

### 7.2 Sentence-level TTS, and audio-paced captions

The LLM's token stream is watched for sentence boundaries and each completed
sentence is synthesized immediately, so first audio depends on the first
*sentence*, not the whole answer. Requests go out concurrently but playback
follows **request** order — a short closing sentence routinely synthesizes faster
than a long opening one.

Then a subtler UX problem: text was finishing 8–10 s before the avatar started
talking, because the token stream lands in under a second while each TTS
round-trip costs 2–3 s. Inverting it, **audio drives the text**: a sentence is
revealed only when its clip starts playing, paced over the clip's real duration.
A word appears as it is spoken. With voice off, the typewriter free-runs.

### 7.3 Measured latency

CPU only, no GPU — the worst case the voice budget must survive.

| Stage | p50 |
| --- | --- |
| Intent routing | 0.05 ms |
| Query understanding (rules) | 0.2 ms |
| Semantic cache probe | 0.05 ms |
| Literal (exact value) lookup | 6–7 ms |
| Query embedding (BGE-small, cached) | 3–8 ms |
| Dense HNSW (Milvus) | 0.2 ms |
| BM25 sparse (Milvus, server-side) | 0.1 ms |
| RRF + dedupe + boost | 0.3 ms |
| Rerank — `fast` (MiniLM ×16) | 250 ms |
| Rerank — `cascade` (MiniLM ×16 → BGE ×4) | 670 ms |
| **Retrieval total (entity-gated)** | **~126 ms** |
| **Retrieval total (cache hit)** | **~46 ms** |
| LLM first token | 1039 ms |
| Azure TTS per sentence | 2.4–3.5 s |
| **Full suite over Milvus** | **p50 355 ms · p95 1948 ms** |

Other deliberate latency choices: branches run concurrently; thinking disabled at
effort `low` (a thinking block would add hundreds of ms of silence before the
first audible word); prompt caching on a byte-stable system prefix (~1.3k tokens
at ~0.1× on reads); two semantic caches keyed by embedding similarity because
exact-string caching is nearly useless for ASR output; torch threads pinned to
physical cores (the default oversubscribes hyperthreads and makes short
cross-encoder batches *slower*); models warmed at startup, not on the first
citizen's question.

---

## 8. Verification results

```bash
cd backend
python scripts/verify_env.py          # interpreter, deps, config, assets
python scripts/test_suite.py          # 126 questions — the main suite
python scripts/test_records.py        # record-atomic chunking integrity
python scripts/test_grounding.py      # misattribution / refusal
python scripts/test_conversation.py   # small talk + multi-turn context
python scripts/test_reverse.py        # query-by-value lookups
python scripts/test_milvus.py         # live Milvus: BM25, filters, idempotency
python scripts/test_speech.py --all   # live Azure TTS, visemes, STT
python scripts/bench_cascade.py       # rerank mode latency
```

| Suite | Result |
| --- | --- |
| Behavioural suite — local store | **126/126** |
| Behavioural suite — **Milvus** | **126/126**, p50 355 ms |
| Record chunking | 56 records, **56 named, 0 merged, 0 anonymous** |
| Grounding / misattribution | **17/17** |
| Conversation + context | **26/26** |
| Live Milvus path | **PASS** — server-side BM25 confirmed |
| Live Azure Speech | **PASS** — 91–95 visemes/utterance, STT round-trip |
| Frontend build | clean, TypeScript clean |
| Dependency imports | 20/20 on Python 3.10.0 |

### The behavioural suite

110+ questions in two halves testing opposite failure modes:

| Class | Count | Asserts |
| --- | --- | --- |
| Conversational | 50 | greetings, capability, identity, chit-chat, thanks, farewell |
| Out of scope | 10 | capital of France, cricket, weather, recipes |
| Field lookups | ~24 | all 12 record fields, direct and possessive |
| Reverse lookups | ~15 | by date of birth, assets, constituency |
| Follow-up chains | 4 | pronoun resolution across turns |
| Ambiguity | 4 | shared first names, shared surnames |
| Absent | 9 | people and values not in the corpus |
| Aggregation | 8 | cross-record questions |
| ASR noise | 3 | lowercase, missing honorific, surname only |

Three design choices worth defending:

**Expectations are parsed from the PDF, not hand-written.** Authoring 50 expected
answers against a 56-record document guarantees transcription mistakes and a suite
that rots when the document changes. The bank extracts every record's fields and
generates questions and ground truth together.

**Assertions are asymmetric.** Document questions assert a positive (right value,
cited, no other record named). Conversational questions assert a *negative* — no
candidate name, no rupee figure, no date, no citation. That is what catches
over-eager retrieval.

**Aggregation is `NO_FABRICATION`, not "must answer".** "Which candidate has the
highest declared assets?" needs all 56 records and `top_k` is 5. The correct
behaviour is to decline rather than invent, so the pass condition is that no
figure appears without a citation. Asserting a correct answer would test a feature
the architecture deliberately does not have.

Value matching accepts digits *or* spoken form where the prompt verbalizes.

### The phrase-list result

A clean TTS render round-tripped through STT as **"Amma oodipes ₹15,000"**. A
mangled scheme name means the BM25 branch matches nothing, so an Azure phrase list
was built from the *same* gazetteer the retriever uses (197 phrases: scheme names
plus all 26 districts and their aliases). Result: **"Amma Vodi pays ₹15,000"**.

---

## 9. Requirements coverage

| # | Requirement | Where |
| --- | --- | --- |
| 1 | Document ingestion (PDF/DOCX/TXT/MD), chunking, embeddings, vector DB | `ingestion/`, `embeddings/`, `vectorstore/` |
| 2 | Metadata: district, category, source, topic | `ingestion/metadata.py` — 26-district gazetteer + aliases |
| 3 | Semantic retrieval with Top-K and similarity threshold | `POST /retrieve`, both configurable per request |
| 4 | Inject retrieved context into LLM prompts | `llm/prompts.py` — numbered, marker-linked context block |
| 5 | Streaming retrieval for partial transcripts + latency optimisation | `voice/streaming.py`, §7 |
| 6 | APIs: `/upload`, `/retrieve`, `/query`, `/health` | all present, plus voice + WebSocket |
| 7 | Any LLM | Azure OpenAI GPT-4 live; Anthropic Claude client implemented |

**Bonus**

| Feature | Where |
| --- | --- |
| Hybrid search | Milvus server-side BM25 + dense, `vectorstore/` |
| Query rewriting | `retrieval/query_rewriter.py` — rules → context → optional LLM |
| Conversation memory | `memory/conversation.py` — history + sticky district slots |
| Multi-document retrieval | RRF across documents, content dedupe in `fusion.py` |
| Citations | verified markers in `rag_service.py`, rendered in `SourceCard.tsx` |
| Multi-query expansion | synonym + district variants fused in `pipeline.py` |
| Semantic caching | `retrieval/cache.py` — retrieval + answer, entity-namespaced |
| Speculative retrieval | `voice/streaming.py` |
| Extractive fallback | `llm/extractive.py` — grounded answers with no LLM |

---

## 10. Known limitations

**Scanned PDFs** are rejected with a clear message rather than silently indexing
nothing. OCR first (`ocrmypdf`).

**Sessions are in-process.** Correct for a single node; swap `SessionStore` for
Redis to scale horizontally — the interface is four methods.

**The local NumPy store is brute-force.** Exact, and ~2 ms for 50k vectors, but
switch to Milvus past ~200k chunks.

**Cross-record aggregation is out of scope.** "Which candidate has the highest
assets?" needs all 56 records; `top_k` is 5. The system declines rather than
inventing. A metadata aggregation endpoint would be the right fix, not a larger
`top_k`.

**The district gazetteer covers Andhra Pradesh and major Telangana districts.**
Other states need entries in `ingestion/metadata.py`.

**Microphone input is disabled in the UI.** Browsers record WebM/Opus, which Azure
STT cannot read, so ffmpeg is required to transcode. The button was removed
because it fails without it; the `/ws/voice` path and all speculative-retrieval
code remain wired and are one button from re-enabling. Installing ffmpeg
(`winget install Gyan.FFmpeg`) should restore it. Note that interim transcripts —
and therefore speculation — additionally need Chrome or Edge; other browsers fall
back to record-then-transcribe, which works but cannot overlap retrieval with
speech.

**Intermittent native crash under memory pressure.** Loading the cross-encoders
below ~3 GB free RAM can produce `0xC0000005` — a hard access violation, not a
graceful OOM. `core/resources.py` refuses the precise reranker tier below 4 GB
free, and the test suites pin `RERANK_MODE=fast`. An environment limit, not a
logic fault.

**Not verified end to end in a browser.** The frontend builds clean, serves 200,
and the React 19 / fiber v9 combination is confirmed correct, but the avatar
rendering, lip-sync animation and live SSE-into-bubble behaviour have not been
observed in a real browser session by the author.
