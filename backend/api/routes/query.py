"""POST /query — grounded answer generation, streaming (SSE) or buffered."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from core.latency import Trace
from core.schemas import QueryRequest, QueryResponse
from ingestion.metadata import resolve_district
from llm.rag_service import get_rag_service
from memory.conversation import get_session_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, http_request: Request):
    """Answer a question from the indexed campaign documents.

    Set `stream: true` for Server-Sent Events. Streaming is the right default for
    a voice or chat UI: the `retrieval` event arrives first so source cards render
    while the model is still generating, and `delta` events let the client start
    TTS on the first complete sentence instead of waiting for the full answer.
    """
    service = get_rag_service()
    filters = _normalize_filters(request)

    if request.stream:
        return StreamingResponse(
            _sse(service, request, filters, http_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",   # nginx would otherwise buffer SSE
                "Connection": "keep-alive",
            },
        )

    trace = Trace(name="query_api")
    try:
        result = await service.answer(
            request.query,
            session_id=request.session_id,
            top_k=request.top_k,
            filters=filters,
            voice_mode=request.voice_mode,
            include_context=request.include_context,
            rewrite=request.rewrite_query,
            max_tokens=request.max_tokens,
            trace=trace,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    retrieval = result.retrieval
    return QueryResponse(
        answer=result.answer,
        session_id=result.session_id,
        query=result.query,
        effective_query=result.effective_query,
        grounded=result.grounded,
        citations=result.citations if request.include_citations else [],
        sources_used=len(result.citations),
        context=result.context or None,
        retrieved=retrieval.results if retrieval else [],
        inferred_filters=(
            {k: v for k, v in retrieval.applied_filters.items() if not k.startswith("_")}
            if retrieval
            else {}
        ),
        usage=result.usage,
        model=result.model,
        cache_hit=result.cache_hit,
        timings_ms={**result.timings_ms, "notes": result.notes},
    )


async def _sse(
    service,
    request: QueryRequest,
    filters,
    http_request: Request,
) -> AsyncIterator[str]:
    trace = Trace(name="query_sse")
    try:
        async for event in service.answer_stream(
            request.query,
            session_id=request.session_id,
            top_k=request.top_k,
            filters=filters,
            voice_mode=request.voice_mode,
            rewrite=request.rewrite_query,
            max_tokens=request.max_tokens,
            trace=trace,
        ):
            # Stop generating (and stop paying) the moment the client goes away.
            if await http_request.is_disconnected():
                logger.info("Client disconnected mid-stream; aborting generation")
                return
            yield _sse_frame(event["type"], event)
    except Exception as exc:  # noqa: BLE001
        logger.error("SSE stream failed: %s", exc, exc_info=True)
        yield _sse_frame("error", {"error": str(exc)})
    finally:
        yield "event: end\ndata: {}\n\n"


def _sse_frame(event: str, payload: dict) -> str:
    body = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {body}\n\n"


def _normalize_filters(request: QueryRequest):
    if not request.filters:
        return None
    filters = request.filters.model_dump(exclude_none=True)
    if filters.get("district"):
        filters["district"] = resolve_district(filters["district"]) or filters["district"]
    if filters.get("districts"):
        filters["districts"] = [resolve_district(d) or d for d in filters["districts"]]
    filters = {k: v for k, v in filters.items() if v not in (None, "", [], {})}
    return filters or None


# ------------------------------------------------------------------- sessions
@router.get("/sessions")
async def list_sessions() -> dict:
    store = get_session_store()
    return {"sessions": store.all(), "total": store.count()}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    store = get_session_store()
    session = store.get(session_id)
    return {
        **session.to_public(),
        "history": [
            {"role": t.role, "content": t.content, "grounded": t.grounded}
            for t in session.turns
        ],
    }


@router.delete("/sessions/{session_id}")
async def reset_session(session_id: str) -> dict:
    store = get_session_store()
    existed = store.reset(session_id)
    return {"status": "reset" if existed else "not_found", "session_id": session_id}
