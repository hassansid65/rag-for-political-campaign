"""
FastAPI application.

Startup does the expensive, one-time work up front — connect to the store, load
BGE-small, load the cross-encoder, warm both with a dummy forward pass. Lazy
loading would push a 3–5 second model init onto whichever unlucky citizen asks
the first question, which for a latency-graded voice system is the wrong place to
put it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from core.config import settings
from core.latency import METRICS
from core.logging_config import new_request_id, request_id_var, setup_logging

setup_logging(settings.log_level, as_json=settings.environment != "dev")
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    logger.info("Starting %s v%s (%s)", settings.app_name, settings.app_version, settings.environment)

    from embeddings.embedder import get_embedder
    from ingestion.service import get_ingest_service
    from llm.provider import get_llm
    from llm.rag_service import get_rag_service
    from retrieval.pipeline import get_pipeline
    from retrieval.query_rewriter import QueryRewriter
    from retrieval.reranker import get_reranker
    from vectorstore.factory import fallback_reason, get_store

    loop = asyncio.get_running_loop()

    async def load_store():
        store = await loop.run_in_executor(None, get_store)
        logger.info("Vector store ready: %s", store.stats())
        return store

    async def load_models():
        """Load the embedder and both cross-encoders, strictly one at a time.

        These must NOT load concurrently: `transformers` enters
        `accelerate.init_empty_weights()`, which swaps torch's *global* default
        device to `meta`, so a parallel load allocates its weights on the meta
        device and dies with "Cannot copy out of meta tensor". The models are
        also lock-guarded internally; this just keeps startup honest and ordered.
        """
        embedder = get_embedder()
        await loop.run_in_executor(None, embedder.load)
        logger.info("Embedder ready: %s", embedder.health())

        if settings.enable_rerank:
            reranker = get_reranker()
            await loop.run_in_executor(None, reranker.load)
            logger.info("Reranker ready: mode=%s", reranker.mode)
        else:
            logger.info("Reranker disabled by config")

    # The store is independent of the models, so those two can overlap.
    results = await asyncio.gather(load_store(), load_models(), return_exceptions=True)
    fatal = False
    for item in results:
        if isinstance(item, BaseException):
            logger.error("Component failed to initialise: %s", item, exc_info=item)
            fatal = True
    if fatal:
        # Serving with a half-initialised pipeline turns one clear startup error
        # into a 500 on every request. /health/ready will report 503.
        logger.error(
            "Startup completed with failures — /health/ready will report not-ready"
        )

    # Probe providers and pick one that actually answers, rather than trusting
    # that a key present in the environment is a key that works.
    from llm.provider import resolve_verified

    llm_client, llm_resolution = await resolve_verified()
    logger.info("LLM: %s", llm_resolution)

    # Wire the pipeline explicitly so the rewriter gets the LLM handle for its
    # optional rewrite pass (constructing it lazily would skip that).
    pipeline = get_pipeline()
    pipeline.rewriter = QueryRewriter(llm_client=llm_client)
    get_rag_service()
    get_ingest_service()

    if fallback_reason():
        logger.warning("VECTOR STORE FALLBACK ACTIVE — %s", fallback_reason())
    if not settings.llm_configured:
        logger.warning("ANTHROPIC_API_KEY is not set — /query returns retrieval-only fallbacks")
    if not settings.azure_speech_configured:
        logger.warning("AZURE_SPEECH_KEY is not set — voice endpoints return 503")

    logger.info(
        "Startup complete in %.2fs · %d chunks indexed",
        time.perf_counter() - started,
        pipeline.store.count(),
    )

    yield

    logger.info("Shutting down…")
    try:
        pipeline.store.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Store close failed: %s", exc)
    try:
        # Close the HTTP pool while the loop is still running, or httpx gets
        # finalised by the GC after loop close and raises "Event loop is closed".
        await get_llm().aclose()
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM client close failed: %s", exc)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Real-time Retrieval-Augmented Generation for a Voice AI political campaign "
        "assistant. BGE-small embeddings · Milvus hybrid search (dense + BM25) · "
        "RRF fusion · BGE cross-encoder reranking · Claude generation · "
        "Azure Speech STT/TTS with viseme lip-sync."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

class StreamAwareGZipMiddleware:
    """GZip, except for Server-Sent Events.

    Starlette's gzip responder feeds every response chunk through one zlib stream
    and only emits bytes when the compressor flushes. For a normal JSON body that
    is free compression; for SSE it destroys the entire point of streaming — dozens
    of ~200-byte `data:` frames sit in the compressor and are released in one
    burst, so the client receives a finished paragraph instead of a token stream.

    The stream was never broken. Gzip was silently re-assembling it.

    Deciding by response `content-type` would be more precise, but middleware sees
    headers only after the app has started sending — by which point the responder
    is already chosen. Excluding the streaming route is the simple correct fix; SSE
    frames are tiny and text-heavy, so we give up almost nothing.
    """

    def __init__(self, app: ASGIApp, minimum_size: int = 1024) -> None:
        self.app = app
        self.gzip = GZipMiddleware(app, minimum_size=minimum_size)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or _is_streaming_path(scope):
            await self.app(scope, receive, send)
            return
        await self.gzip(scope, receive, send)


# `/query` is the only SSE endpoint, and only when the caller asks for a stream —
# but the flag lives in the request *body*, which middleware must not consume.
# Excluding the path wholesale costs a few kB of uncompressed JSON on the
# non-streaming variant and cannot mis-buffer a stream.
_STREAMING_PATHS = ("/query",)


def _is_streaming_path(scope: Scope) -> bool:
    path = scope.get("path", "")
    return any(path.startswith(prefix) for prefix in _STREAMING_PATHS)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
)
# Retrieval responses carry full chunk text and scores; they compress ~5x.
# SSE is excluded — see StreamAwareGZipMiddleware.
app.add_middleware(StreamAwareGZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id and measure server-side latency for every call."""
    request_id = request.headers.get("X-Request-ID") or new_request_id()
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        METRICS.incr("http.error")
        request_id_var.reset(token)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000
    # Skip the health/metrics endpoints — polling them would dominate the
    # percentiles and hide real request latency.
    if not request.url.path.startswith(("/health", "/metrics")):
        METRICS.observe(f"http.{request.method}{request.url.path}", elapsed_ms)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    request_id_var.reset(token)
    return response


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": str(exc) if settings.environment == "dev" else "An internal error occurred",
            "request_id": request_id_var.get(),
        },
    )


# ------------------------------------------------------------------- routers
from api.routes import health as health_routes  # noqa: E402
from api.routes import query as query_routes  # noqa: E402
from api.routes import retrieve as retrieve_routes  # noqa: E402
from api.routes import upload as upload_routes  # noqa: E402
from api.routes import voice as voice_routes  # noqa: E402

app.include_router(health_routes.router)
app.include_router(upload_routes.router)
app.include_router(retrieve_routes.router)
app.include_router(query_routes.router)
app.include_router(voice_routes.router)


@app.get("/", tags=["ops"])
async def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "endpoints": {
            "ingest": ["POST /upload", "POST /ingest-path", "GET /documents", "DELETE /documents/{doc_id}"],
            "retrieval": ["POST /retrieve", "GET /districts", "GET /resolve-district"],
            "generation": ["POST /query (stream=true for SSE)"],
            "voice": [
                "POST /voice/stt", "POST /voice/tts", "POST /voice/turn",
                "GET /voice/voices", "WS /ws/voice",
            ],
            "ops": ["GET /health", "GET /health/live", "GET /health/ready", "GET /metrics"],
            "sessions": ["GET /sessions", "GET /sessions/{id}", "DELETE /sessions/{id}"],
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "dev",
        log_config=None,   # our setup_logging owns formatting
    )
