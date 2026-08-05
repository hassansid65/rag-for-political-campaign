# Real-Time RAG for a Voice AI Political Campaign Assistant

A retrieval-augmented generation pipeline wired into a simulated voice assistant.
Campaign documents (manifestos, district briefs, candidate profiles, scheme
booklets, FAQs) are uploaded, chunked, embedded and indexed; a citizen speaks;
the assistant retrieves the relevant passages **while they are still talking** and
answers out loud with a lip-synced 3D avatar, citing its sources.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INGESTION                                                                  │
│                                                                             │
│  PDF · DOCX · TXT · MD · HTML · CSV                                         │
│         │                                                                   │
│         ▼                                                                   │
│  Loader (PyMuPDF / python-docx / markdown)                                  │
│    · font-size heading detection    · table extraction                      │
│    · page-aligned blocks            · script-based language detection       │
│         │                                                                   │
│         ▼                                                                   │
│  Structure-aware chunker                                                    │
│    · section-first, recursive char split inside sections only               │
│    · 700-char children + 1800-char parent windows                           │
│    · Q&A pairs and table rows kept atomic                                   │
│    · contextual header prepended for embedding                              │
│         │                                                                   │
│         ▼                                                                   │
│  Metadata extraction — district · category · source · topic                 │
│    (+ state, section path, page, candidate, party, schemes, language)        │
│         │                                                                   │
│         ▼                                                                   │
│  BGE-small-en-v1.5 · 384-dim · passage side (no prefix)                      │
│         │                                                                   │
│         ▼                                                                   │
│  Milvus — HNSW(COSINE) dense + BM25 sparse + scalar/ARRAY metadata indexes   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  RETRIEVAL                                                                  │
│                                                                             │
│  utterance              │
│         │                                                                   │
│         ▼                                                                   │
│  Query understanding          ~0.2 ms                                       │
│    · filler stripping · acronym/transliteration expansion                   │
│    · district inference from a 26-district gazetteer + aliases              │
│    · follow-up resolution from conversation memory                          │
│    · optional Haiku rewrite, only when the rules under-specify              │
│         │                                                                   │
│         ▼                                                                   │
│  Semantic cache probe         ~0.05 ms ──── hit ──► return                   │
│         │                                                                   │
│         ▼                                                                   │
│  BGE-small query embedding    3–8 ms (LRU cached)                           │
│         │                                                                   │
│         ├──────────────────────┬─────────────────────────┐                  │
│         ▼                      ▼                         ▼                  │
│   dense HNSW            BM25 sparse              variant queries            │
│   (metadata pre-filter) (same filter)            (synonym / district)       │
│         └──────────────────────┴─────────────────────────┘                  │
│                          run concurrently · 2–15 ms                         │
│         ▼                                                                   │
│  Reciprocal Rank Fusion (k=60, weighted)   ~0.1 ms                          │
│         ▼                                                                   │
│  Dedupe + metadata boosting                ~0.1 ms                          │
│         ▼                                                                   │
│  Cascade rerank: MiniLM-L6 (16) → BGE-reranker-base (top 4)   ~670 ms       │
│         ▼                                                                   │
│  Threshold → Top-K → parent-window expansion                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  GENERATION & VOICE                                                         │
│                                                                             │
│  Prompt: cached system prefix ‖ history ‖ numbered context ‖ question        │
│         ▼                                                                   │
│  Claude (thinking off, effort low, prompt caching on)                        │
│         ▼                                                                   │
│  Token stream ──► sentence boundary detector                                 │
│         │                    │                                              │
│         │                    ▼                                              │
│         │            Azure TTS per sentence ──► viseme cues ──► GLB avatar   │
│         ▼                                                                   │
│  Citation verification (only markers the model actually used)               │
│         ▼                                                                   │
│  FastAPI: REST + SSE + WebSocket                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick start

### Fastest path (Windows)

```
START_BACKEND.bat      →  http://localhost:8000/docs
START_FRONTEND.bat     →  http://localhost:3000
```

Both are idempotent: on first run they create `backend\.venv`, install pinned
dependencies, copy `.env` from the example, run the environment check, and start
the server. On later runs they just start it. The manual steps below are the same
thing, spelled out.

### 0. Prerequisite — Python 3.10

```bash
python --version        # must print 3.10.x
```

Everything is pinned to **CPython 3.10**. A virtualenv is not optional here: a
global Python with other ML projects in it will already hold conflicting pins for
`torch`, `numpy` and `transformers`, and pip will happily leave you with a broken
mix.

### 1. Backend

**Windows (PowerShell)**

```powershell
cd backend
python -m venv .venv                 # note: -m venv, not -m 3.10 venv
.\.venv\Scripts\Activate.ps1

# torch first, from the CPU index — its own step so a failure is isolated
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt

copy .env.example .env               # then fill in ANTHROPIC_API_KEY
python -m uvicorn api.main:app --reload --port 8000
```

**macOS / Linux**

```bash
cd backend
python3.10 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
cp .env.example .env
python -m uvicorn api.main:app --reload --port 8000
```

**Check the environment before anything else:**

```bash
python scripts/verify_env.py
```

Verifies the interpreter version, that the venv is actually active, that every
dependency imports, that all project modules load, which device and rerank mode
were resolved, and which keys are configured. Run this first whenever something
looks wrong — it separates "missing package" from "real bug" in one command.

First boot downloads ~500 MB of model weights and warms both cross-encoders,
so expect 30–60 s. Subsequent starts are ~5 s.

> **Runs on CPU by default.** `EMBEDDING_DEVICE` / `RERANKER_DEVICE` default to
> `cpu` — no CUDA runtime, no VRAM ceiling, no driver variance. Set either to
> `cuda` (or `auto` to probe) to opt in; the cross-encoders are 20–50× faster on
> a GPU, but the CPU numbers already fit the latency budget.

<details>
<summary>Two Windows install failures worth knowing about</summary>

**`Could not install packages due to an OSError … INSTALLER<random>.tmp`** —
Defender or a sync client holding a file in `site-packages` mid-write. It can
leave the venv without a `pyvenv.cfg`, after which every command reports
`No pyvenv.cfg file`. Delete `.venv` and recreate it; installing torch as a
separate step (as above) keeps the blast radius small.

**`ModuleNotFoundError: No module named 'pkg_resources'`** — `setuptools` 81
removed `pkg_resources`, which `pymilvus` still imports. `requirements.txt` pins
`setuptools<81` for exactly this. Without it, pymilvus fails at *import* and the
store silently falls back to the local backend, which looks like "Milvus is
down".
</details>

### 2. Vector store

Pick one:

| Mode | `VECTOR_BACKEND` | Notes |
| --- | --- | --- |
| **Milvus standalone** (recommended) | `milvus` | `docker compose up -d milvus` — 2.5+ gives server-side BM25 |
| **Zilliz Cloud** | `milvus` | set `MILVUS_URI` + `MILVUS_TOKEN` |
| **Milvus Lite** | `milvus_lite` | embedded, single file — **Linux/macOS only** |
| **Local NumPy** | `local` | no dependencies, exact flat search, works on Windows |

The backend degrades along that list automatically and logs which one it landed
on. If Docker isn't available (e.g. a bare Windows box), `VECTOR_BACKEND=local`
runs the full pipeline — hybrid search, RRF, reranking, filtering, citations — with
no vector server at all.

### 3. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev            # http://localhost:3000
```

Stack: **Next 16 · React 19.2 · @react-three/fiber 9 · @react-three/drei 10**.

React 19 is mandatory, and `package.json` is not what decides it. Next 16's App
Router runs its **own vendored React** for client components —
`next/dist/compiled/react` is `19.3.0-canary` — regardless of the version pinned
in the project. Pinning React 18 with fiber v8 therefore compiles fine and then
dies in the browser at module evaluation:

```
Uncaught TypeError: Cannot read properties of undefined (reading 'ReactCurrentOwner')
    at $$$reconciler (node_modules_react-reconciler…)
    at createRenderer (node_modules_@react-three_fiber_dist…)
```

`ReactCurrentOwner` is a React **18** internal that React 19 removed;
`react-reconciler` (which fiber v8 depends on) reads it. Fiber v9 dropped
`react-reconciler` altogether — it is absent from the tree now, so the failure is
structurally impossible rather than merely avoided. Fiber v9 declares
`react >=19 <19.3`, which 19.2.8 satisfies.

None of the avatar code needed changing: `Canvas`, `useFrame`, `useGLTF`,
`useAnimations`, `OrbitControls`, `Environment` and `Html` are API-compatible
across v8→v9 and v9→v10, and the GLB rig, morph-target names and viseme mapping
are untouched.

<details>
<summary><code>npm audit</code> — read before running <code>--force</code></summary>

`npm audit fix --force` will offer to "fix" the remaining `postcss` / `sharp`
advisories by installing **next@9.3.3** — a 2020 release. That is a downgrade of
seven major versions, not a fix; it would break the App Router entirely. Don't run
it. Those two packages are transitive *inside* Next and are only resolvable when
Next itself ships updated pins.

Upgrading 14 → 16 (which does clear the Next advisories) needs three changes,
already applied here:

1. `next.config.js` — Turbopack is the default bundler in 16, so a lingering
   `webpack` key raises a hard error. The `.glb` loader rule it contained was
   never needed: the avatar models live in `public/` and are fetched by URL, not
   imported. Replaced with an explicit empty `turbopack: {}`.
2. `app/page.tsx` — `dynamic(..., { ssr: false })` is a build error in a Server
   Component from Next 15 on. The page is now a thin `'use client'` boundary whose
   only job is to defer the three.js import.
3. `NODE_OPTIONS=--max-old-space-size=4096` — Next 16 type-checks in a separate
   worker that hard-crashes (`exit 3221226505`) when memory is tight. Observed at
   ~2 GB free; the launcher sets this for you.
</details>

### 4. Index the sample corpus

```bash
curl -F "files=@../data/sample_docs/manifesto_2024.md" \
     -F "files=@../data/sample_docs/schemes_welfare.md" \
     -F "files=@../data/sample_docs/district_profile_ntr_vijayawada.md" \
     -F "files=@../data/sample_docs/faq_voters.md" \
     -F "files=@../data/sample_docs/candidate_profiles.md" \
     http://localhost:8000/upload
```

…or just drag them onto the upload panel in the UI.

### Everything at once

```bash
export ANTHROPIC_API_KEY=sk-ant-…
export AZURE_SPEECH_KEY=…
docker compose up -d              # add `--profile tools` for the Attu Milvus UI
```

---

## Testing

```bash
cd backend

python scripts/verify_env.py       # interpreter, deps, config, assets
python scripts/test_suite.py       # 110+ questions, the main suite
python scripts/test_records.py     # record-atomic chunking integrity
python scripts/test_grounding.py   # misattribution / refusal
python scripts/test_conversation.py# small talk + multi-turn context
python scripts/test_speech.py --all# live Azure TTS, visemes, STT
```

### The behavioural suite

`scripts/test_suite.py` runs **110+ questions** in two halves that test opposite
failure modes:

| Half | Count | Asserts |
| --- | --- | --- |
| Conversational | 50 | greetings, capability, identity, chit-chat, thanks, farewell |
| Out of scope | 10 | capital of France, cricket, weather, recipes |
| Field lookups | ~24 | all 12 record fields, direct and possessive phrasing |
| Reverse lookups | ~15 | by date of birth, assets, constituency |
| Follow-up chains | 4 | pronoun resolution across turns |
| Ambiguity | 4 | shared first names, shared surnames |
| Absent | 9 | people and values not in the corpus |
| Aggregation | 8 | cross-record questions |
| ASR noise | 3 | lowercase, missing honorific, surname only |

**Expectations are parsed out of the PDF, not written by hand.** Hand-authoring
50 expected answers against a 56-record document guarantees two things:
transcription mistakes, and a suite that silently rots the moment the document
changes. `tests/question_bank.py` extracts every record's fields and *generates*
the questions and their ground truth together, so each question asserts against
the document rather than against someone's memory of it.

**The assertions are asymmetric on purpose.** Document questions assert a
positive (the right value is present, it is cited, no other record is named).
Conversational questions assert a *negative* — no candidate name, no rupee
figure, no date, no citation. That is what catches over-eager retrieval, which is
how "tell me what can you do for me" once returned a candidate's constituency
priorities.

**Aggregation questions are marked `NO_FABRICATION`, not "must answer".**
"Which candidate has the highest declared assets?" needs all 56 records and
`top_k` is 5. The correct behaviour is to decline rather than invent, so the pass
condition is that no figure appears without a citation. Asserting a correct answer
would be testing a feature the architecture deliberately does not have.

Value matching accepts digits *or* the spoken form, because the system prompt
verbalizes amounts for TTS — `43.1 lakh` and "forty-three lakh and ten thousand"
both satisfy the same assertion. An earlier digits-only check failed on perfectly
correct output.

### Known flake

An intermittent `exit -1073741819` (`0xC0000005`, access violation) can occur
while loading the cross-encoders under memory pressure — a hard native crash, not
a graceful OOM. It is an environment limit, not a logic fault: free memory and
re-run. `core/resources.py` refuses to load the precise reranker tier below 4 GB
free for this reason, and the suites pin `RERANK_MODE=fast`.

## Verify it works

```bash
cd backend
python scripts/smoke_test.py --backend local --fresh
```

Runs the real pipeline over the sample corpus with no server, no API keys and no
Milvus: ingests 5 documents, then walks nine probe queries covering exact-figure
lookup, keyword-heavy retrieval, district stickiness, follow-up resolution, table
rows, and an out-of-scope question. Prints every score the pipeline computed
(dense, BM25, RRF, rerank), per-stage timings, semantic-cache behaviour, and the
speculative-retrieval reuse rate.

Latency micro-benchmarks:

```bash
python scripts/bench_rerank.py     # model × sequence length × batch
python scripts/bench_cascade.py    # fast vs cascade vs single
```

---

## The API

Interactive docs at `http://localhost:8000/docs`.

### `POST /upload`
Multipart. `files` (repeatable) plus optional metadata overrides (`category`,
`district`, `state`, `topic`, `candidate`, `party`, or a `metadata` JSON blob).
Metadata is inferred automatically; anything you pass wins over inference.

```bash
curl -F "files=@manifesto.pdf" -F "district=Vijayawada" \
     -F "category=manifesto" http://localhost:8000/upload
```

### `POST /retrieve`
Retrieval only — no generation. This is the debugging endpoint: it returns every
score the pipeline computed and the filters it inferred.

```json
{
  "query": "I'm from Vijayawada, what about schools?",
  "top_k": 5,
  "similarity_threshold": 0.3,
  "filters": { "district": "Vijayawada", "category": "manifesto" },
  "session_id": "abc123"
}
```

Response includes, per hit: `score`, `dense_score`, `sparse_score`, `rrf_score`,
`rerank_score`, `retriever` (`dense` | `sparse` | `hybrid`), full metadata, the
child `chunk_text` and the expanded `text`, plus `timings_ms` per stage.

### `POST /query`
Grounded answer. `"stream": true` returns Server-Sent Events:

| Event | Payload |
| --- | --- |
| `retrieval` | sources + chunks + inferred filters — arrives **before** generation |
| `delta` | `{ "text": "…" }` incremental tokens |
| `final` | answer, verified citations, usage, per-stage timings |
| `error` | `{ "error": "…" }` |

### `GET /health`
Per-component status **and** rolling p50/p95/p99 for every pipeline stage. Also
`/health/live` (liveness), `/health/ready` (readiness, 503 until models load),
and `/metrics`.

### Voice

| Endpoint | Purpose |
| --- | --- |
| `POST /voice/stt` | base64 audio → text (auto-detects en-IN / te-IN / hi-IN) |
| `POST /voice/tts` | text → base64 WAV + viseme mouth cues |
| `POST /voice/turn` | one full turn: retrieve → generate → speak → lip-sync |
| `WS /ws/voice` | streaming turn with speculative retrieval on partials |
| `GET /voice/voices` | available voice presets |

### Documents & sessions
`GET /documents`, `DELETE /documents/{doc_id}`, `POST /ingest-path`,
`GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}`,
`POST /cache/invalidate`.

---

## Streaming retrieval for partial transcripts

Requirement #5 of the brief, and the most interesting part of the system.

A naive voice turn is strictly serial — the citizen stops talking, *then* the
pipeline starts:

```
speech ──► ASR final ──► retrieve ──► LLM ──► TTS ──► audio
                        │◄─ everything happens here: ~1.8 s of silence ─►│
```

But Azure emits `recognizing` events every ~150–300 ms **while the citizen is
still speaking**. By the time they finish we usually already know the question.
That window is free compute, so we use it:

```
"what is"                          → too short, ignored
"what is amma vodi"                → fire retrieval #1  ─┐
"what is amma vodi eligib"         → cancel #1, fire #2 ─┤ during speech
"what is amma vodi eligibility for my daughter"  FINAL   ┘
   → embed final, compare to held speculations
   → cosine 0.995 ≥ 0.94 → reuse #2's results, skip the pipeline entirely
```

Four guards keep it safe and cheap:

1. **Stability gate** — fire only above `PARTIAL_MIN_CHARS` / `PARTIAL_MIN_TOKENS`.
2. **Debounce + single-flight** — one speculation in flight; newer partials cancel
   older ones. Without this a 5-second utterance fires 20 retrievals.
3. **Embedding-similarity reuse, not string equality** — ASR revises words and adds
   punctuation, so the final is almost never byte-identical. Comparing embeddings
   is what makes the hit rate high.
4. **Verify before trust** — below threshold we throw the work away and retrieve
   fresh. A wrong reused context is worse than a slow correct one.

We keep a short *ring of recent speculations*, not just the latest, because ASR
revises mid-utterance (`"dia"` → `"dialysis"`) and the closest match to the final
is not always the most recent guess.

Measured on the sample corpus: reuse at cosine 0.995, **~2.6 s of pipeline work
removed from the post-speech critical path**. Because speculation runs during
speech, it also gets the *full* rerank cascade — a live turn without a usable
speculation drops to `fast` rerank instead, degrading latency rather than
correctness.

The second half is **sentence-level TTS**: the LLM's token stream is watched for
sentence boundaries, and each completed sentence is synthesized immediately while
the model is still writing the next. First audio therefore depends on the first
*sentence*, not the whole answer.

---

## Latency

Measured on 12 CPU cores, no GPU — the worst case the voice budget has to survive.

| Stage | p50 |
| --- | --- |
| Query understanding (rules) | 0.2 ms |
| Semantic cache probe | 0.05 ms |
| Query embedding (BGE-small, cached) | 3–8 ms |
| Dense HNSW + BM25 (concurrent) | 2–15 ms |
| RRF fusion + dedupe + boost | 0.3 ms |
| Rerank — `fast` (MiniLM ×16) | 250 ms |
| Rerank — `cascade` (MiniLM ×16 → BGE ×4) | 670 ms |
| Rerank — `single` (BGE ×16) | 1690 ms |
| **Retrieval total, cascade** | **~860 ms** |
| **Retrieval total, cache hit** | **~46 ms** |
| Claude first token (thinking off, effort low) | 400–700 ms |
| Azure TTS per sentence | 150–350 ms |

Where the wins came from, concretely:

- **Reranking is cascaded.** BGE-reranker-base is 278M params and costs ~105 ms
  per pair on CPU; MiniLM-L6 is 22M and costs ~16 ms. Running MiniLM over all 16
  candidates and BGE over only the top 4 gets BGE-quality ordering of the
  shortlist for ~40% of the cost. `RERANK_MODE` exposes the whole curve.
- **Rerank on children, not parent windows.** Cross-encoder cost scales with
  sequence length. Scoring 1800-char parent windows cost 304 ms/pair versus
  117 ms/pair for 700-char children, with no measured ranking gain — the extra
  text is mostly neighbouring topics.
- **`RERANKER_MAX_LENGTH` 512 → 256.** 512 paid full attention cost for tokens
  the model then truncated away. Halved per-pair time.
- **Branches run concurrently.** Dense and BM25 hit different indexes and neither
  depends on the other.
- **Thinking disabled, effort `low`.** A thinking block would add hundreds of ms
  of silence before the first audible word.
- **Prompt caching on a byte-stable system prefix.** Everything volatile lives
  after the cache breakpoint, so the ~1.3k-token prompt reads at ~0.1×.
- **Two semantic caches.** Retrieval results and full answers, keyed by embedding
  similarity and namespaced by filter signature. Exact-string caching is nearly
  useless for ASR output.
- **Torch threads pinned to physical cores.** Torch defaults to logical-core
  count, which oversubscribes and makes short cross-encoder batches slower.
- **Models warmed at startup**, not on the first citizen's question.

---

## Design decisions

### Why BGE-small-en-v1.5
384 dims, 33M params, MTEB-retrieval within a couple of points of models 10× its
size. It embeds a voice-length query in **3–8 ms on CPU** — an API embedding model
would add 80–250 ms of network round-trip per turn before we even reach the vector
DB. Three details that are easy to get wrong: the instruction prefix goes on
**queries only** (BGE is asymmetric); vectors are L2-normalized so IP ≡ cosine;
and the pooling is **CLS**, not mean.

### Why hybrid, not dense-only
Citizens ask about proper nouns — `Amma Vodi`, `Rythu Bharosa`, `Aarogyasri`,
`15,000`. Dense retrieval blurs rare tokens; BM25 nails exact matches. In the
smoke test `"rythu bharosa amount"` is carried by the BM25 branch (sparse 6.96)
while dense alone ranks it lower. Neither alone is sufficient, which is the whole
argument for fusion.

### Why RRF instead of weighted score fusion
The branches produce incomparable numbers — dense similarity is in [0,1], BM25 is
unbounded and corpus-dependent. Min-max normalizing before a weighted sum makes
fusion sensitive to each batch's *spread*: one outlier BM25 score squashes every
other keyword result toward zero. RRF only asks "how highly did each branch rank
this?", so it is stable across queries with no per-corpus tuning.

### Why small children with large parents
Retrieval precision peaks with chunks small enough to be about one thing (~700
chars); answer quality peaks with more surrounding context. So we embed the child
and hand the LLM the parent window around it. Best of both, no extra vectors.
`chunk_text` and `text` are kept separate on the response because a citation
snippet must show the child that matched — the parent window commonly opens in
the previous section, which reads as a mismatch.

### Why small talk never reaches retrieval

A citizen typed "hey", then "hello", and both times received a full factual answer
about a candidate's date of birth. Every utterance was being pushed through the
pipeline, so a greeting retrieved *something* — and conversation memory then
resolved it as a follow-up, turning "hello" into a restatement of the previous
question.

That is not a grounding bug, it is a missing conversational layer.
`retrieval/intent.py` classifies each utterance first and routes greetings,
farewells, thanks, acknowledgements, identity and capability questions **without
any retrieval at all**.

Rules, not an LLM classifier. This runs on every turn including every voice turn,
and the decision is between about eight lexically obvious categories: a regex pass
costs ~0.05 ms, an LLM would add 300–600 ms to the critical path of a spoken
conversation to answer a question a pattern already answers. Anything ambiguous
falls through to `FACTUAL`, which is the safe default — it just means we retrieve.
A greeting *prefix* stays factual, so "hello, what does the manifesto say about
roads?" is still a query.

It also removes small talk from the hallucination surface entirely: a greeting
never reaches the generator alongside a retrieved context it might quote.

### Why reverse lookups need an exact-value path

Asked *"who is born on 14 October 1985"*, the system answered with a candidate
born **7 September 1985**. Same misattribution class as the name case, arriving
through a different door — and neither retrieval branch could fix it:

* No name in the query, so the entity gate never engaged.
* **Dense cannot help.** All 56 profiles are one template; the query embedding is
  roughly equidistant from every one, and a date barely registers in a 384-dim
  sentence vector.
* **BM25 cannot either.** `14 October 1985` tokenizes to `{14, october, 1985}`,
  and many records share each token individually, so a two-of-three match ranks
  about as well as the exact one.

The correct record was often not even in the candidate set. `retrieval/literals.py`
extracts distinctive values from the query — dates, rupee amounts, decimals,
percentages — normalizes them (`14 October 1985` ≡ `14/10/1985`) and matches them
**verbatim** against record text, *before* the ANN search. An exact hit is
authoritative and replaces the candidate set.

The other half matters as much: when a distinctive value appears nowhere in the
corpus, retrieval returns **no context**. Returning near-misses is precisely what
produced the wrong date. A near-miss is more misleading than no answer.

Only *selective* literals qualify. A bare `100` appears in every "30 to 100 beds"
priority line, so gating on it would trade one wrong answer for another.

### Why chunking is record-atomic (the main anti-hallucination measure)

`data/RAG_Test_Candidate_Profiles.pdf` is 28 pages holding **56 candidate
profiles** that share one template and differ by a proper noun and a few numbers.
On a corpus like that the dangerous failure is not inventing facts — it is
**misattribution**: reporting candidate A's assets under candidate B's name. That
answer is fluent, cited, and wrong, and no amount of prompt engineering fixes it
if both records are in the context.

Four layers address it, and each one was added because a test caught the layer
above it failing:

**1. One record per chunk** (`ingestion/records.py`). Field labels that repeat
(`Born.` `Education.` `Assets declaration.`) are detected as a template; a label
repeating means a new record started. Each record becomes exactly one chunk,
whatever its size, and always leads with its own title. A size-based splitter
fails two ways here: it strands `Assets declaration.` in a chunk that names
nobody, and it merges the tail of one profile with the head of the next. Detection
is template-driven, so the scheme booklet (`Benefit.` / `Eligibility.`) and the
FAQ (`Q1.` / `A1.`) work with no extra code. Verified by
`scripts/test_records.py`: 56 records, 56 named, **0 merged, 0 anonymous**.

**2. Entity-aware retrieval.** A name in the query is matched against each
chunk's `record_name` using symmetric token F1 and boosted decisively. Cosine
similarity cannot do this job — two different profiles sit at ~0.95 — but a name
either matches a record or it doesn't.

**3. Entity gating** (`pipeline._gate_to_entity`). Boosting the right record to
rank 1 is not enough. The corpus contains deliberate near-namesakes — *Sarojini
Vasireddy* alongside *Padmavathi Vasireddy* and *Rajeswari Vasireddy* — so a
top-5 context for one of them carried three different asset declarations. For an
entity-scoped question, retrieval is made **precise instead of high-recall**: only
the best-matching record survives, plus any genuinely tied for best (which the
prompt then surfaces as "which one do you mean?"). Non-record chunks are kept,
since a manifesto paragraph may still be part of the answer.

Scoring had to be symmetric to make this work. An earlier coverage-plus-surname
score gave *Sarojini Vasireddy* vs *Padmavathi Vasireddy* **0.85** — a shared
surname was almost a full match — so all three Vasireddys survived. F1 penalises
a missing token in either direction and scores that pair 0.50.

**4. Entity-scoped caches.** The semantic cache was found serving one
candidate's results for another's question: *"assets of Anuradha Merugu"* and
*"assets of Anuradha Undela"* embed at **0.99 cosine**, above the 0.97 cache
threshold. Raising the threshold cannot fix it — the questions genuinely are
near-identical strings. The cache namespace therefore includes the named entity.

Result, from `scripts/test_grounding.py`: for a question naming one candidate the
assembled context contains **exactly one record (~1200 chars) and zero other
candidates' figures**. The wrong number is not merely discouraged — it is not in
the prompt.

The prompt adds a "never mix up records" section and labels every record passage
`THIS PASSAGE IS ONLY ABOUT: <name>`, as defence in depth for the ambiguous case
where two records legitimately remain.

### Why the contextual header
A chunk reading *"Rs. 15,000 per year for two children"* is unretrievable alone.
Prefixing `NTR | scheme | Amma Vodi › Eligibility` **at embedding time only**
makes it findable without polluting what the LLM reads.

### Why thresholding happens after reranking
The dense score of the correct chunk is often mediocre; its cross-encoder score is
not. Cutting on cosine first discards chunks the reranker would have promoted to
rank 1.

### Why filters relax on empty results
A district filter that returns nothing is usually over-constraint, not absence of
an answer. Telling a citizen from Vijayawada that we know nothing about roads
because no chunk was *primarily* tagged NTR is a worse failure than answering from
the state-wide manifesto. District matching is deliberately generous too — a chunk
qualifies if it is primarily about the district **or** merely mentions it.

### Why `category` is a boost, not a filter
`category_hint` and `topic_hint` come from keyword overlap and are guesses. As
hard filters they silently empty the result set — and a vague question is exactly
when you can least afford that. As a small additive boost, a strong lexical or
semantic match can still win.

### Why citations are verified, not assumed
The model is told to emit `[1]`, `[2]` markers; we then check which markers it
*actually used* and return only those. Returning all retrieved chunks as "sources"
is the common shortcut and it is dishonest — it implies the answer rests on five
documents when it rests on one. An answer that cites nothing is marked
`grounded: false` and the UI flags it.

### Why the vector store degrades instead of failing
Milvus Lite has no Windows build and Milvus standalone needs Docker. A missing
vector DB should make the system slower and less scalable, never non-functional —
someone cloning this repo gets a working demo, and the log says exactly what
happened and how to get the real thing.

---

## Layout

```
backend/
  api/            FastAPI app + routers (upload, retrieve, query, health, voice)
  core/           config · schemas · logging · latency instrumentation
  ingestion/      loader · chunker · metadata · ingest service
  embeddings/     BGE-small wrapper (torch or ONNX) + LRU cache
  vectorstore/    Milvus · local NumPy fallback · BM25 · factory
  retrieval/      pipeline · fusion (RRF) · cascade reranker · rewriter · caches
  llm/            Claude client · prompts · RAG service
  memory/         conversation memory + sticky slots
  voice/          Azure STT/TTS · lip-sync · speculative streaming
  scripts/        smoke_test · bench_rerank · bench_cascade
frontend/
  app/            Next.js 14 App Router
  components/     Avatar (GLB + visemes) · ChatWindow · MessageBubble ·
                  SourceCard · SidePanel
  lib/            api (SSE) · voice (mic, WS, audio queue)
  public/         Shayla_Changes(Visemes).glb · working.glb
data/sample_docs/ manifesto · district profile · schemes · FAQ · candidates
```

## Bonus requirements

| Feature | Where |
| --- | --- |
| Hybrid search | `vectorstore/milvus_store.py`, `vectorstore/bm25.py` |
| Query rewriting | `retrieval/query_rewriter.py` — rules → context → optional Haiku |
| Conversation memory | `memory/conversation.py` — history + sticky district slots |
| Multi-document retrieval | RRF across documents; dedupe by content in `fusion.py` |
| Citations | `llm/rag_service.py` marker verification, `SourceCard.tsx` |
| Multi-query expansion | synonym + district variants fused in `pipeline.py` |
| Semantic caching | `retrieval/cache.py` — retrieval + answer, filter-namespaced |
| Speculative retrieval | `voice/streaming.py` |

## Known limitations

- **Scanned PDFs** are rejected with a clear message rather than silently indexing
  nothing. OCR them first (`ocrmypdf`).
- **Sessions are in-process.** Fine for a single node; swap `SessionStore` for
  Redis to scale horizontally (4 methods).
- **The local NumPy store is brute-force.** Exact, and ~2 ms for 50k vectors, but
  switch to Milvus past ~200k chunks.
- **The district gazetteer covers Andhra Pradesh and major Telangana districts.**
  Other states need entries in `ingestion/metadata.py`.
- **Interim transcripts need Chrome or Edge.** Other browsers fall back to
  record-then-transcribe, which works but cannot overlap retrieval with speech.
