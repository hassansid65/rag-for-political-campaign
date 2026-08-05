"""GET /health, /health/live, /health/ready, /metrics."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Response

from core.config import settings
from core.latency import METRICS
from core.schemas import ComponentHealth, HealthResponse
from llm.rag_service import get_rag_service
from memory.conversation import get_session_store
from retrieval.pipeline import get_pipeline
from vectorstore.factory import fallback_reason
from voice.azure_speech import get_speech
from voice.lipsync import lipsync_health

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])

_STARTED_AT = time.time()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Aggregate health.

    Reports per-component status *and* rolling latency percentiles for every
    pipeline stage. The percentiles are the point: a system that is "up" but whose
    p95 rerank latency has tripled is broken for a voice caller, and a plain
    boolean health check would never show it.
    """
    components: list[ComponentHealth] = []
    pipeline = get_pipeline()

    # ---------------------------------------------------------- vector store
    store_start = time.perf_counter()
    store_health = pipeline.store.health()
    store_latency = (time.perf_counter() - store_start) * 1000
    store_status = "ok" if store_health.get("status") == "ok" else "down"
    detail = f"{store_health.get('backend')} · {store_health.get('entities', 0)} chunks"
    if fallback_reason():
        store_status = "degraded"
        detail += f" · {fallback_reason()}"
    components.append(
        ComponentHealth(
            name="vector_store",
            status=store_status,  # type: ignore[arg-type]
            detail=detail,
            latency_ms=round(store_latency, 2),
        )
    )

    # ------------------------------------------------------------- embedder
    embedder_health = pipeline.embedder.health()
    components.append(
        ComponentHealth(
            name="embedder",
            status="ok" if embedder_health["ready"] else "degraded",  # type: ignore[arg-type]
            detail=(
                f"{embedder_health['model']} · {embedder_health['dim']}d · "
                f"{embedder_health['backend']} · cache "
                f"{embedder_health['cache']['hit_rate']:.0%}"
            ),
        )
    )

    # ------------------------------------------------------------- reranker
    reranker_health = pipeline.reranker.health()
    if not settings.enable_rerank:
        rerank_status = "disabled"
    elif reranker_health["ready"]:
        rerank_status = "ok"
    elif reranker_health["available"]:
        rerank_status = "ok"      # lazy: loads on first query
    else:
        rerank_status = "degraded"
    components.append(
        ComponentHealth(
            name="reranker",
            status=rerank_status,  # type: ignore[arg-type]
            detail=f"{reranker_health['model']} · top-{settings.rerank_top_n}",
        )
    )

    # ------------------------------------------------------------------ LLM
    service = get_rag_service()
    llm_health = await service.llm.health()
    components.append(
        ComponentHealth(
            name="llm",
            status=llm_health.get("status", "down"),  # type: ignore[arg-type]
            detail=f"{llm_health.get('model')} · {llm_health.get('detail', '')}".strip(" ·"),
            latency_ms=llm_health.get("latency_ms"),
        )
    )

    # ---------------------------------------------------------------- speech
    speech_health = get_speech().health()
    components.append(
        ComponentHealth(
            name="azure_speech",
            status=speech_health.get("status", "down"),  # type: ignore[arg-type]
            detail=(
                f"region={speech_health.get('region', '-')} · "
                f"voice={speech_health.get('voice', '-')} · "
                f"ffmpeg={'yes' if speech_health.get('ffmpeg') else 'no'}"
                if speech_health.get("status") == "ok"
                else speech_health.get("detail", "")
            ),
        )
    )

    # --------------------------------------------------------------- lipsync
    lip = lipsync_health()
    components.append(
        ComponentHealth(
            name="lipsync",
            status="ok" if lip["enabled"] else "disabled",  # type: ignore[arg-type]
            detail=f"primary={lip['primary_source']} · rhubarb={lip['rhubarb']}",
        )
    )

    # A missing LLM key or Azure key is a *configuration* state, not a fault —
    # retrieval still works. Only a dead store or embedder is "down".
    critical = {c.name: c.status for c in components if c.name in {"vector_store", "embedder"}}
    if any(s == "down" for s in critical.values()):
        overall = "down"
    elif any(c.status in {"degraded", "down"} for c in components):
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,  # type: ignore[arg-type]
        version=settings.app_version,
        environment=settings.environment,
        uptime_s=round(time.time() - _STARTED_AT, 1),
        timestamp=datetime.now(timezone.utc),
        components=components,
        collection={
            **store_health,
            "documents": len(pipeline.store.list_documents()) if store_health.get("entities") else 0,
        },
        config={
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "reranker_model": settings.reranker_model,
            "llm_model": settings.llm_model,
            "llm_effort": settings.llm_effort,
            "llm_thinking": settings.llm_thinking,
            "prompt_cache": settings.llm_enable_prompt_cache,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "parent_expansion": settings.enable_parent_expansion,
            "top_k": settings.retrieval_top_k,
            "candidate_top_k": settings.candidate_top_k,
            "rerank_top_n": settings.rerank_top_n,
            "similarity_threshold": settings.similarity_threshold,
            "hybrid": settings.enable_hybrid,
            "rrf_k": settings.rrf_k,
            "semantic_cache": settings.enable_semantic_cache,
            "vector_backend": settings.vector_backend,
        },
        latency_ms=METRICS.snapshot(),
    )


@router.get("/health/live")
async def live() -> dict:
    """Liveness — is the process running? Never touches a dependency."""
    return {"status": "alive", "uptime_s": round(time.time() - _STARTED_AT, 1)}


@router.get("/health/ready")
async def ready(response: Response) -> dict:
    """Readiness — can we actually serve a query? Gates traffic in k8s."""
    pipeline = get_pipeline()
    store_ok = pipeline.store.health().get("status") == "ok"
    embedder_ok = pipeline.embedder.is_ready
    has_data = pipeline.store.count() > 0

    is_ready = store_ok and embedder_ok
    if not is_ready:
        response.status_code = 503
    return {
        "ready": is_ready,
        "store": store_ok,
        "embedder": embedder_ok,
        "indexed_chunks": pipeline.store.count(),
        # Not a readiness failure: an empty index is a valid pre-upload state.
        "has_documents": has_data,
    }


@router.get("/metrics")
async def metrics() -> dict:
    """Rolling latency percentiles + counters for every instrumented stage."""
    pipeline = get_pipeline()
    return {
        "latency_ms": METRICS.snapshot(),
        "counters": METRICS.counters(),
        "caches": {
            "embedding": pipeline.embedder.cache_stats(),
            "retrieval": pipeline.cache.stats(),
            "answer": get_rag_service().answer_cache.stats(),
        },
        "sessions": get_session_store().count(),
        "store": pipeline.store.stats(),
    }


@router.post("/metrics/reset")
async def reset_metrics() -> dict:
    METRICS.reset()
    return {"status": "reset"}


@router.post("/cache/invalidate")
async def invalidate_cache() -> dict:
    from retrieval.cache import invalidate_all_caches

    return {"status": "invalidated", "dropped": invalidate_all_caches()}
