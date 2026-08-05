"""
Live Milvus check: schema, server-side BM25, hybrid search, literal lookup.

The Milvus path had only ever been exercised against the local NumPy fallback, so
this verifies the things that are *specific* to the real server and cannot be
inferred from the fallback working:

  * collection creation with the BM25 Function and a SPARSE_FLOAT_VECTOR field
  * whether BM25 is actually computed server-side, or silently degraded
  * dense HNSW search with metadata pre-filtering via a boolean expression
  * `text LIKE '%…%'` scalar filtering, which backs the reverse-lookup gate
  * delete + re-upsert idempotency on a content-hashed doc_id
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

PDF = BACKEND_DIR.parent / "data" / "RAG_Test_Candidate_Profiles.pdf"

os.environ["VECTOR_BACKEND"] = "milvus"
os.environ.setdefault("MILVUS_URI", "http://localhost:19530")
os.environ["COLLECTION_NAME"] = "milvus_probe"
os.environ.setdefault("RERANK_MODE", "fast")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from core.logging_config import setup_logging  # noqa: E402

setup_logging("WARNING")


async def main() -> int:
    from ingestion.service import get_ingest_service
    from retrieval.literals import selective_literals
    from retrieval.pipeline import get_pipeline
    from vectorstore.factory import fallback_reason, get_store

    failures: list[str] = []
    print("=" * 78)
    print("  LIVE MILVUS CHECK")
    print("=" * 78)

    # ---- 1. connect --------------------------------------------------------
    store = get_store()
    if fallback_reason():
        print(f"  FELL BACK: {fallback_reason()}")
        return 1
    print(f"  backend        : {store.name}")

    pipeline = get_pipeline()
    pipeline.store = store
    pipeline.embedder.load()
    store.ensure_collection(pipeline.embedder.dim, recreate=True)

    stats = store.stats()
    print(f"  uri            : {stats['uri']}")
    print(f"  index / metric : {stats['index']} / {stats['metric']}")
    native = stats.get("native_bm25")
    print(f"  server BM25    : {native}")
    if not native:
        print("                   (degraded to the in-process BM25 index —")
        print("                    hybrid still works, but this is not the 2.5 path)")

    # ---- 2. ingest --------------------------------------------------------
    t0 = time.perf_counter()
    outcome = await get_ingest_service().ingest_file(PDF)
    if not outcome.ok:
        print(f"  ingest FAILED  : {outcome.error}")
        return 1
    ingest_s = time.perf_counter() - t0
    count = store.count()
    print(f"\n  ingested       : {count} entities in {ingest_s:.1f}s")
    if count != len(outcome.chunks):
        failures.append(f"count mismatch: store={count} chunks={len(outcome.chunks)}")

    # ---- 3. dense search --------------------------------------------------
    vector = pipeline.embedder.encode_query("declared assets of the Adoni candidate")
    t0 = time.perf_counter()
    dense = store.search_dense(vector, top_k=5)
    dense_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  dense search   : {len(dense)} hits in {dense_ms:.1f}ms")
    if not dense:
        failures.append("dense search returned nothing")
    else:
        print(f"                   top={dense[0].metadata.record_name!r} score={dense[0].score:.3f}")

    # ---- 4. sparse / BM25 -------------------------------------------------
    t0 = time.perf_counter()
    sparse = store.search_sparse("Adoni assembly constituency Kurnool", top_k=5)
    sparse_ms = (time.perf_counter() - t0) * 1000
    print(f"  sparse search  : {len(sparse)} hits in {sparse_ms:.1f}ms "
          f"({'server-side' if native else 'in-process'})")
    if not sparse:
        failures.append("sparse search returned nothing")
    else:
        print(f"                   top={sparse[0].metadata.record_name!r} score={sparse[0].score:.3f}")

    # ---- 5. metadata pre-filter (boolean expr during ANN traversal) --------
    # The district filter is deliberately generous: a chunk qualifies if it is
    # *primarily* about the district OR merely mentions it (see
    # build_filter_expr). So the assertion is "every hit references Kurnool
    # somewhere", not "every hit's primary district is Kurnool" — the strict
    # version contradicts the documented design and would make
    # "I'm from Vijayawada" miss the state-wide manifesto paragraph that lists it.
    filtered = store.search_dense(vector, top_k=10, filters={"district": "Kurnool"})
    primary = {h.metadata.district for h in filtered}
    print(f"\n  filtered dense : {len(filtered)} hits")
    print(f"                   primary districts : {sorted(d for d in primary if d)}")
    if not filtered:
        failures.append("metadata-filtered dense search returned nothing")
    else:
        off_target = [
            h.metadata.record_name
            for h in filtered
            if h.metadata.district != "Kurnool"
            and "Kurnool" not in (h.metadata.districts or [])
        ]
        mentions = sum(1 for h in filtered if "Kurnool" in (h.metadata.districts or []))
        print(f"                   mention Kurnool   : {mentions}/{len(filtered)}")
        if off_target:
            failures.append(f"filter admitted records with no Kurnool reference: {off_target[:3]}")

    # ---- 6. literal lookup (text LIKE) — backs the reverse-lookup gate ----
    literals = selective_literals("who is born on 14 October 1985")
    if not literals:
        failures.append("no literal extracted from a date query")
    else:
        t0 = time.perf_counter()
        hits = store.find_literal(literals[0].variants, limit=5)
        literal_ms = (time.perf_counter() - t0) * 1000
        names = [h.metadata.record_name for h in hits]
        print(f"  literal lookup : {len(hits)} hit(s) in {literal_ms:.1f}ms → {names}")
        if len(hits) != 1:
            failures.append(f"literal lookup returned {len(hits)} hits, expected exactly 1")
        elif "Kesineni" not in (names[0] or ""):
            failures.append(f"literal lookup found the wrong record: {names[0]}")

    # ---- 7. full pipeline through Milvus ----------------------------------
    print(f"\n  --- pipeline over Milvus ---")
    probes = [
        ("who is born on 14 October 1985", "Kesineni"),
        ("What are the declared assets of Smt. Sarojini Vasireddy?", "Vasireddy"),
        ("Who is the candidate for Adoni?", "Kesineni"),
    ]
    for question, expect in probes:
        result = await pipeline.retrieve(question, top_k=5)
        names = [r.metadata.record_name for r in result.results]
        ok = bool(names) and expect in (names[0] or "")
        total = result.timings_ms.get("total", 0)
        print(f"    [{'PASS' if ok else 'FAIL'}] {question[:46]:<48} → {names[:2]} ({total:.0f}ms)")
        if not ok:
            failures.append(f"pipeline {question!r} → {names[:2]}, expected {expect}")

    # ---- 8. delete + re-upsert idempotency --------------------------------
    doc_id = outcome.chunks[0].metadata.doc_id
    removed = store.delete_document(doc_id)
    after = store.count()
    print(f"\n  delete doc     : removed {removed}, {after} remain")
    if after != 0:
        failures.append(f"delete left {after} entities behind")

    reingest = await get_ingest_service().ingest_file(PDF)
    recount = store.count()
    print(f"  re-ingest      : {recount} entities (same doc_id={reingest.chunks[0].metadata.doc_id == doc_id})")
    if recount != count:
        failures.append(f"re-ingest count drifted: {recount} vs {count}")

    # ---- verdict ----------------------------------------------------------
    print(f"\n{'=' * 78}")
    if failures:
        print(f"  {len(failures)} PROBLEM(S):")
        for item in failures:
            print(f"    - {item}")
        print("  VERDICT: FAIL")
    else:
        print("  VERDICT: PASS — Milvus path fully exercised")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
