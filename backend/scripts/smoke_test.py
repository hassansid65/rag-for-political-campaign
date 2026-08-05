"""
End-to-end smoke test — no server, no API keys, no Milvus required.

Runs the real pipeline (load → chunk → metadata → embed → index → hybrid search
→ RRF → rerank) against the sample campaign documents and prints what each stage
produced. This is the fastest way to verify a change didn't break retrieval.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --backend local --no-rerank
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG pipeline smoke test")
    parser.add_argument("--backend", default="local", choices=["milvus", "milvus_lite", "local"])
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--docs", default=None, help="Directory of documents to ingest")
    parser.add_argument("--fresh", action="store_true", help="Rebuild the index from scratch")
    return parser.parse_args()


ARGS = parse_args()
os.environ["VECTOR_BACKEND"] = ARGS.backend
os.environ["ENABLE_RERANK"] = "false" if ARGS.no_rerank else "true"
os.environ["ENABLE_HYBRID"] = "false" if ARGS.no_hybrid else "true"
os.environ["ENABLE_LLM_QUERY_REWRITE"] = "false"      # keep the test offline
os.environ["LOG_LEVEL"] = "WARNING"
if ARGS.fresh:
    os.environ["COLLECTION_NAME"] = f"smoke_{int(time.time())}"
    os.environ["LOCAL_INDEX_FILE"] = str(BACKEND_DIR / "data" / "smoke_index.npz")

from core.config import settings  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402

setup_logging(settings.log_level)

QUESTIONS: list[tuple[str, str]] = [
    ("I'm from Vijayawada", "district statement — should set the sticky slot"),
    ("what about schools there?", "follow-up — must resolve 'there' to NTR district"),
    ("How much do I get under Amma Vodi?", "exact figure lookup (Rs. 15,000)"),
    ("am I eligible if I have three children", "follow-up on eligibility, elliptical"),
    ("rythu bharosa amount", "keyword-heavy — BM25 branch should carry this"),
    ("what is the pension for someone on dialysis", "table row lookup (Rs. 10,000)"),
    ("who is my candidate", "candidate profile, needs district context"),
    ("tell me about the metro rail project", "district-specific infrastructure"),
    ("what is the capital of France", "out of scope — must retrieve nothing useful"),
]


def rule(char: str = "─", width: int = 78) -> str:
    return char * width


async def main() -> int:
    from ingestion.service import get_ingest_service
    from memory.conversation import get_session_store
    from retrieval.pipeline import get_pipeline

    print(rule("="))
    print("  RAG PIPELINE SMOKE TEST")
    print(rule("="))
    print(f"  vector backend  : {settings.vector_backend}")
    print(f"  embedding model : {settings.embedding_model} ({settings.embedding_dim}d)")
    print(f"  reranker        : {settings.reranker_model if settings.enable_rerank else 'disabled'}")
    print(f"  hybrid search   : {settings.enable_hybrid}")
    print(f"  chunk / overlap : {settings.chunk_size} / {settings.chunk_overlap}")
    print(rule())

    # -------------------------------------------------------------- warm up
    t0 = time.perf_counter()
    pipeline = get_pipeline()
    pipeline.embedder.load()
    print(f"\n[1] Models loaded in {time.perf_counter() - t0:.2f}s")
    print(f"    store: {pipeline.store.stats()}")

    # --------------------------------------------------------------- ingest
    docs_dir = Path(ARGS.docs) if ARGS.docs else BACKEND_DIR.parent / "data" / "sample_docs"
    if not docs_dir.exists():
        print(f"\n!! Sample documents not found at {docs_dir}")
        return 1

    print(f"\n[2] Ingesting from {docs_dir}")
    service = get_ingest_service()
    t0 = time.perf_counter()
    outcomes = await service.ingest_directory(docs_dir)
    ingest_s = time.perf_counter() - t0

    total_chunks = 0
    for outcome in outcomes:
        if not outcome.ok:
            print(f"    FAIL  {outcome.error}")
            continue
        doc = outcome.document
        total_chunks += doc.chunks_indexed
        print(
            f"    ok    {doc.source:<42} {doc.chunks_indexed:>3} chunks  "
            f"category={doc.category:<16} districts={doc.districts[:2]}"
        )
    print(f"    → {total_chunks} chunks in {ingest_s:.2f}s "
          f"({total_chunks / max(ingest_s, 1e-9):.0f} chunks/s)")

    if total_chunks == 0:
        print("\n!! Nothing was indexed — aborting")
        return 1

    # ------------------------------------------------------- chunk inspection
    sample = outcomes[0].chunks[: min(2, len(outcomes[0].chunks))]
    print(f"\n[3] Chunk inspection ({outcomes[0].document.source})")
    for chunk in sample:
        meta = chunk.metadata
        print(f"    id={chunk.id}  chars={len(chunk.text)}  page={meta.page}")
        print(f"      section_path : {meta.section_path}")
        print(f"      district     : {meta.district}   topic: {meta.topic}")
        print(f"      schemes      : {meta.scheme_names[:3]}")
        print(f"      parent window: {len(chunk.parent_text or '')} chars")
        preview = " ".join(chunk.text.split())[:130]
        print(f"      text         : {preview}…")

    # ------------------------------------------------------------- retrieval
    print(f"\n[4] Retrieval + memory  (session-scoped, top_k={ARGS.top_k})")
    print(rule())
    store = get_session_store()
    session = store.get("smoke-session")
    latencies: list[float] = []
    failures = 0

    for question, intent in QUESTIONS:
        result = await pipeline.retrieve(
            question, session=session, top_k=ARGS.top_k
        )
        total_ms = result.timings_ms.get("total", 0.0)
        latencies.append(total_ms)

        print(f"\n  Q: {question}")
        print(f"     intent    : {intent}")
        if result.effective_query != question:
            print(f"     rewritten : {result.effective_query}")
        shown_filters = {
            k: v for k, v in result.applied_filters.items() if not k.startswith("_")
        }
        print(f"     filters   : {shown_filters or '(none)'}")
        print(
            f"     latency   : {total_ms:.1f}ms  "
            f"(candidates={result.total_candidates}, reranked={result.reranked}, "
            f"cache={result.cache_hit})"
        )
        stages = {
            k: v for k, v in result.timings_ms.items()
            if k not in {"total", "notes"} and not k.startswith("@")
        }
        print(f"     stages    : {stages}")

        if not result.results:
            print("     results   : (none)")
            if "capital of France" not in question:
                failures += 1
                print("     ^^ UNEXPECTED: expected at least one hit")
            continue

        for rank, hit in enumerate(result.results, start=1):
            meta = hit.metadata
            snippet = " ".join(hit.text.split())[:110]
            print(
                f"       {rank}. score={hit.score:.3f} "
                f"[dense={_fmt(hit.dense_score)} sparse={_fmt(hit.sparse_score)} "
                f"rrf={_fmt(hit.rrf_score, 5)} rerank={_fmt(hit.rerank_score)}] "
                f"via={hit.retriever}"
            )
            print(f"          {meta.source} › {meta.section or '-'} "
                  f"(district={meta.district}, category={meta.category})")
            print(f"          {snippet}…")

        # Simulate the assistant answering so follow-ups have history to resolve.
        session.add_turn("user", question)
        session.add_turn("assistant", f"[simulated answer about {result.effective_query}]")

    # ----------------------------------------------------------------- summary
    print(f"\n{rule('=')}")
    print("  SUMMARY")
    print(rule("="))
    ordered = sorted(latencies)
    print(f"  queries          : {len(latencies)}")
    print(f"  unexpected empty : {failures}")
    print(f"  latency p50      : {ordered[len(ordered) // 2]:.1f}ms")
    print(f"  latency p95      : {ordered[int(len(ordered) * 0.95)]:.1f}ms")
    print(f"  latency max      : {ordered[-1]:.1f}ms")
    print(f"  sticky district  : {session.district}")
    print(f"  embed cache      : {pipeline.embedder.cache_stats()}")
    print(f"  retrieval cache  : {pipeline.cache.stats()}")

    # ------------------------------------------------- cache + speculation
    print(f"\n[5] Semantic cache — asking a paraphrase of an earlier question")
    t0 = time.perf_counter()
    cached = await pipeline.retrieve("how much do i get under amma vodi", session=session)
    print(f"    cache_hit={cached.cache_hit} sim={cached.cache_similarity} "
          f"in {(time.perf_counter() - t0) * 1000:.1f}ms")

    print(f"\n[6] Speculative retrieval over partial transcripts")
    from voice.streaming import StreamingRetriever

    retriever = StreamingRetriever(session=store.get("smoke-voice"))
    # Azure emits whole words as it recognises, so partials grow word by word.
    partials = [
        "what is",                                          # too short → skipped
        "what is the pension",                              # fires
        "what is the pension for someone",                  # supersedes
        "what is the pension for someone on dialysis",      # ~identical to final
    ]
    for partial in partials:
        fired = await retriever.on_partial(partial)
        print(f"    partial {partial!r:<48} fired={fired}")
        await asyncio.sleep(0.30)   # let the debounce window pass + work finish

    t0 = time.perf_counter()
    final = await retriever.on_final(
        "What is the pension for someone on dialysis?"
    )
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"    final resolved in {elapsed:.1f}ms")
    print(f"    stats: {retriever.stats_dict()}")
    print(f"    notes: {final.notes}")
    if final.results:
        top = final.results[0]
        print(f"    top hit: {top.metadata.source} › {top.metadata.section}")
        print(f"      snippet (child): {' '.join((top.chunk_text or '').split())[:110]}…")
        print(f"      context (parent): {len(top.text)} chars")

    print(f"\n[7] Sticky district across a fresh session")
    fresh = store.get("smoke-sticky")
    for utterance in ("I'm from Vijayawada", "what about schools there?"):
        result = await pipeline.retrieve(utterance, session=fresh, top_k=2)
        shown = {k: v for k, v in result.applied_filters.items() if not k.startswith("_")}
        print(f"    {utterance!r:<32} slot={fresh.district!r:<8} filters={shown}")

    print(f"\n{rule('=')}")
    verdict = "PASS" if failures == 0 else f"FAIL ({failures} unexpected empty results)"
    print(f"  VERDICT: {verdict}")
    print(rule("="))
    return 0 if failures == 0 else 1


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
