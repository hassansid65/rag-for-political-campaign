"""POST /retrieve — semantic + hybrid retrieval with Top-K and similarity threshold."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from core.latency import Trace
from core.schemas import RetrieveRequest, RetrieveResponse
from ingestion.metadata import all_districts, resolve_district
from memory.conversation import get_session_store
from retrieval.pipeline import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["retrieval"])


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """Retrieve relevant chunks without generating an answer.

    Exposed separately from `/query` because it is the endpoint you actually debug
    with: it returns every score the pipeline computed — dense similarity, BM25,
    the fused RRF score, and the cross-encoder score — plus the filters that were
    inferred from the utterance. If retrieval is wrong, this tells you which stage
    made it wrong.
    """
    pipeline = get_pipeline()
    trace = Trace(name="retrieve_api")

    session = None
    if request.session_id:
        session = get_session_store().get(request.session_id)

    filters = request.filters.model_dump(exclude_none=True) if request.filters else None
    if filters:
        # Accept "Vijayawada" and resolve to the canonical district the index uses.
        if filters.get("district"):
            filters["district"] = resolve_district(filters["district"]) or filters["district"]
        if filters.get("districts"):
            filters["districts"] = [
                resolve_district(d) or d for d in filters["districts"]
            ]
        filters = {k: v for k, v in filters.items() if v not in (None, "", [], {})} or None

    try:
        result = await pipeline.retrieve(
            request.query,
            session=session,
            top_k=request.top_k,
            filters=filters,
            similarity_threshold=request.similarity_threshold,
            use_rerank=request.rerank,
            use_hybrid=request.hybrid,
            rewrite=request.rewrite_query,
            trace=trace,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    results = result.results
    if not request.include_text:
        # Metadata-only mode for latency benchmarking / eval harnesses.
        results = [r.model_copy(update={"text": ""}) for r in results]

    return RetrieveResponse(
        query=result.query,
        effective_query=result.effective_query,
        rewrites=result.variants,
        inferred_filters={
            k: v for k, v in result.applied_filters.items() if not k.startswith("_")
        },
        results=results,
        total_candidates=result.total_candidates,
        reranked=result.reranked,
        cache_hit=result.cache_hit,
        timings_ms={**result.timings_ms, "notes": result.notes},
    )


@router.get("/districts")
async def districts() -> dict:
    """Canonical districts the retriever can filter on, for UI dropdowns."""
    return {"districts": all_districts(), "total": len(all_districts())}


@router.get("/resolve-district")
async def resolve(name: str) -> dict:
    """Resolve a free-text place name to a canonical district."""
    canonical = resolve_district(name)
    return {"input": name, "district": canonical, "resolved": canonical is not None}
