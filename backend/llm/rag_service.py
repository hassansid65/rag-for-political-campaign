"""
RAG answering service — retrieval + generation + citations + memory.

This is the layer the API endpoints and the voice loop both call, so the grounding
behaviour is identical whether a question arrives over HTTP or a WebSocket.

Citation handling is the part most worth reading. The LLM is told to emit `[1]`,
`[2]` markers; we then **verify** which markers it actually used and return only
those as citations. Returning all retrieved chunks as "sources" is the common
shortcut and it is dishonest — it implies the answer rests on five documents when
it rests on one. If the model cites nothing, we mark the answer `grounded=False`,
which the UI shows differently.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from core.config import settings
from core.latency import METRICS, Trace
from core.schemas import Citation, RetrievedChunk, TokenUsage
from llm.claude_client import GenerationResult
from llm.provider import get_llm
from llm.extractive import extractive_answer
from retrieval.intent import classify
from llm.prompts import (
    SYSTEM_PROMPT,
    build_context_block,
    build_user_turn,
    fallback_answer,
    no_context_answer,
)
from memory.conversation import Session, get_session_store
from retrieval.cache import get_answer_cache
from retrieval.pipeline import RetrievalPipeline, RetrievalResult, get_pipeline

logger = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class AnswerResult:
    answer: str
    session_id: str
    query: str
    effective_query: str
    grounded: bool = True
    citations: list[Citation] = field(default_factory=list)
    retrieval: Optional[RetrievalResult] = None
    context: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    cache_hit: bool = False
    timings_ms: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class RAGService:
    def __init__(
        self,
        pipeline: Optional[RetrievalPipeline] = None,
        # Any provider implementing the LLMClient protocol — see llm/provider.py.
        llm: Optional[Any] = None,
    ) -> None:
        self.pipeline = pipeline or get_pipeline()
        self.llm = llm or get_llm()
        self.sessions = get_session_store()
        self.answer_cache = get_answer_cache()

    # ==================================================================== ask
    async def answer(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        voice_mode: bool = False,
        include_context: bool = False,
        rewrite: bool = True,
        max_tokens: Optional[int] = None,
        trace: Optional[Trace] = None,
        use_cache: bool = True,
    ) -> AnswerResult:
        trace = trace or Trace(name="query")
        session = self.sessions.get(session_id or uuid.uuid4().hex[:12])

        # Small talk never reaches retrieval. "hey" was previously being embedded,
        # searched, resolved as a follow-up and answered with a candidate's date of
        # birth — a greeting is not a query, and routing it through the pipeline
        # both looks broken and needlessly exposes a generated answer to a
        # retrieved context it might quote.
        with trace.stage("intent"):
            intent = classify(query, has_history=bool(session.history_pairs()))
        if not intent.needs_retrieval and intent.reply:
            METRICS.incr(f"intent.{intent.intent.value}")
            session.add_turn("user", query)
            session.add_turn("assistant", intent.reply, grounded=True)
            return AnswerResult(
                answer=intent.reply,
                session_id=session.session_id,
                query=query,
                effective_query=query,
                # Conversational, not a factual claim — so no citation is owed and
                # the UI must not show an "uncited answer" warning.
                grounded=True,
                model=f"conversational ({intent.intent.value})",
                notes=[f"intent={intent.intent.value}; retrieval skipped"],
                timings_ms=trace.finish(),
            )

        session.update_slots_from_text(query)

        retrieval = await self.pipeline.retrieve(
            query,
            session=session,
            top_k=top_k,
            filters=filters,
            voice_mode=voice_mode,
            rewrite=rewrite,
            trace=trace,
            use_cache=use_cache,
        )

        # An empty context is a decision, not a dead end — see no_context_answer.
        if retrieval.is_empty():
            return self._no_context_result(
                query=query, session=session, retrieval=retrieval, trace=trace
            )

        # ---------------------------------------------- answer-level cache
        if use_cache and retrieval.query_vector is not None:
            with trace.stage("answer_cache"):
                # `cache_scope`, not `applied_filters` — it carries the named
                # entity, without which two questions about different people
                # collide in the cache at ~0.99 cosine. See pipeline.cache_scope.
                cached = self.answer_cache.get(
                    retrieval.query_vector, retrieval.cache_scope
                )
            if cached is not None:
                payload, similarity, _ = cached
                result = self._from_cached(payload, session, query, retrieval, similarity)
                result.timings_ms = trace.finish()
                session.add_turn("user", query)
                session.add_turn("assistant", result.answer, grounded=result.grounded)
                METRICS.incr("answer.cache_hit")
                return result

        with trace.stage("prompt"):
            context_block, citation_meta = build_context_block(
                retrieval.results, max_chars=settings.max_context_chars
            )
            user_turn = build_user_turn(
                question=retrieval.effective_query,
                context_block=context_block,
                district=session.district,
                history=session.transcript(limit=3),
                voice_mode=voice_mode,
            )

        # No LLM available: answer extractively rather than apologising. Retrieval
        # already found the text; only the paraphrasing step is missing, and
        # verbatim text cannot drift a figure. See llm/extractive.py.
        if not self.llm.is_configured:
            return self._extractive_result(
                query=query,
                session=session,
                retrieval=retrieval,
                citation_meta=citation_meta,
                context_block=context_block if include_context else "",
                trace=trace,
                reason="llm not configured",
            )

        with trace.stage("generate"):
            generation = await self._generate(
                user_turn,
                session=session,
                voice_mode=voice_mode,
                max_tokens=max_tokens,
            )
        trace.mark("answer_ready")

        # Generation failed (auth, rate limit, network). Same reasoning as above:
        # degrade to extractive rather than returning a canned non-answer.
        if generation.failed and retrieval.results:
            return self._extractive_result(
                query=query,
                session=session,
                retrieval=retrieval,
                citation_meta=citation_meta,
                context_block=context_block if include_context else "",
                trace=trace,
                reason=next(iter(generation.notes), "generation failed"),
            )

        answer_text = generation.text.strip() or fallback_answer(1)
        citations, grounded = self._resolve_citations(
            answer_text, citation_meta, retrieval.results
        )

        session.last_query = query
        session.last_effective_query = retrieval.effective_query
        session.last_chunk_ids = [c.id for c in retrieval.results]
        session.add_turn("user", query)
        session.add_turn(
            "assistant",
            answer_text,
            citations=[c.model_dump() for c in citations],
            grounded=grounded,
        )

        result = AnswerResult(
            answer=answer_text,
            session_id=session.session_id,
            query=query,
            effective_query=retrieval.effective_query,
            grounded=grounded,
            citations=citations,
            retrieval=retrieval,
            context=context_block if include_context else "",
            usage=generation.usage,
            model=generation.model,
            cache_hit=retrieval.cache_hit,
            notes=[*retrieval.notes, *generation.notes],
        )
        result.timings_ms = trace.finish()

        if use_cache and retrieval.query_vector is not None and grounded:
            self.answer_cache.put(
                retrieval.query_vector,
                retrieval.effective_query,
                {
                    "answer": answer_text,
                    "citations": [c.model_dump() for c in citations],
                    "grounded": grounded,
                    "model": generation.model,
                },
                retrieval.cache_scope,
            )

        return result

    # ============================================================== streaming
    async def answer_stream(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        voice_mode: bool = False,
        rewrite: bool = True,
        max_tokens: Optional[int] = None,
        trace: Optional[Trace] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a grounded answer.

        Event order is deliberate: `retrieval` first (so the UI can render source
        cards while the model is still generating), then `delta`s, then `final`.
        """
        trace = trace or Trace(name="query_stream")
        session = self.sessions.get(session_id or uuid.uuid4().hex[:12])

        # Same conversational short-circuit as the buffered path.
        intent = classify(query, has_history=bool(session.history_pairs()))
        if not intent.needs_retrieval and intent.reply:
            METRICS.incr(f"intent.{intent.intent.value}")
            session.add_turn("user", query)
            session.add_turn("assistant", intent.reply, grounded=True)
            yield {
                "type": "retrieval",
                "session_id": session.session_id,
                "effective_query": query,
                "inferred_filters": {},
                "sources": [],
                "chunks": [],
                "cache_hit": False,
                "timings_ms": {},
            }
            yield {"type": "delta", "text": intent.reply}
            yield {
                "type": "final",
                "answer": intent.reply,
                "grounded": True,
                "citations": [],
                "session_id": session.session_id,
                "usage": TokenUsage().model_dump(),
                "model": f"conversational ({intent.intent.value})",
                "timings_ms": trace.finish(),
                "notes": [f"intent={intent.intent.value}; retrieval skipped"],
            }
            return

        session.update_slots_from_text(query)

        retrieval = await self.pipeline.retrieve(
            query,
            session=session,
            top_k=top_k,
            filters=filters,
            voice_mode=voice_mode,
            rewrite=rewrite,
            trace=trace,
        )
        trace.mark("retrieval_done")

        context_block, citation_meta = build_context_block(
            retrieval.results, max_chars=settings.max_context_chars
        )

        yield {
            "type": "retrieval",
            "session_id": session.session_id,
            "effective_query": retrieval.effective_query,
            "inferred_filters": {
                k: v for k, v in retrieval.applied_filters.items() if not k.startswith("_")
            },
            "sources": citation_meta,
            "chunks": [c.model_dump() for c in retrieval.results],
            "cache_hit": retrieval.cache_hit,
            "timings_ms": dict(retrieval.timings_ms),
        }

        # Same deterministic refusal as the buffered path — an endpoint must not
        # be the reason grounding behaviour differs.
        if retrieval.is_empty():
            result = self._no_context_result(
                query=query, session=session, retrieval=retrieval, trace=trace
            )
            yield {"type": "delta", "text": result.answer}
            yield {
                "type": "final",
                "answer": result.answer,
                "grounded": False,
                "citations": [],
                "session_id": session.session_id,
                "usage": TokenUsage().model_dump(),
                "model": result.model,
                "timings_ms": result.timings_ms,
                "notes": result.notes,
            }
            return

        if not self.llm.is_configured:
            # Same degradation as the non-streaming path: extract verbatim rather
            # than emit a canned apology over text retrieval already found.
            result = self._extractive_result(
                query=query,
                session=session,
                retrieval=retrieval,
                citation_meta=citation_meta,
                context_block="",
                trace=trace,
                reason="llm not configured",
            )
            yield {"type": "delta", "text": result.answer}
            yield {
                "type": "final",
                "answer": result.answer,
                "grounded": result.grounded,
                "citations": [c.model_dump() for c in result.citations],
                "session_id": session.session_id,
                "usage": TokenUsage().model_dump(),
                "model": result.model,
                "timings_ms": result.timings_ms,
                "notes": result.notes,
            }
            return

        user_turn = build_user_turn(
            question=retrieval.effective_query,
            context_block=context_block,
            district=session.district,
            history=session.transcript(limit=3),
            voice_mode=voice_mode,
        )

        collected: list[str] = []
        generation: Optional[GenerationResult] = None
        first_delta = True

        async for event in self.llm.stream(
            SYSTEM_PROMPT,
            user_turn,
            max_tokens=max_tokens or (256 if voice_mode else settings.llm_max_tokens),
            effort="low" if voice_mode else None,
        ):
            if event["type"] == "text":
                if first_delta:
                    trace.mark("first_token")
                    first_delta = False
                collected.append(event["text"])
                yield {"type": "delta", "text": event["text"]}
            elif event["type"] == "error":
                # Mid-stream failure with nothing emitted yet: fall back to
                # extractive so the caller still gets a grounded answer instead of
                # an error banner. If tokens were already sent we cannot retract
                # them, so surface the error.
                if not collected and retrieval.results:
                    result = self._extractive_result(
                        query=query,
                        session=session,
                        retrieval=retrieval,
                        citation_meta=citation_meta,
                        context_block="",
                        trace=trace,
                        reason=f"stream failed: {event['error']}",
                    )
                    yield {"type": "delta", "text": result.answer}
                    yield {
                        "type": "final",
                        "answer": result.answer,
                        "grounded": result.grounded,
                        "citations": [c.model_dump() for c in result.citations],
                        "session_id": session.session_id,
                        "usage": TokenUsage().model_dump(),
                        "model": result.model,
                        "timings_ms": result.timings_ms,
                        "notes": result.notes,
                    }
                else:
                    yield {"type": "error", "error": event["error"]}
                return
            elif event["type"] == "done":
                generation = event["result"]

        answer_text = (generation.text if generation else "".join(collected)).strip()
        if not answer_text:
            answer_text = fallback_answer(1)

        citations, grounded = self._resolve_citations(
            answer_text, citation_meta, retrieval.results
        )

        session.last_query = query
        session.last_effective_query = retrieval.effective_query
        session.last_chunk_ids = [c.id for c in retrieval.results]
        session.add_turn("user", query)
        session.add_turn(
            "assistant",
            answer_text,
            citations=[c.model_dump() for c in citations],
            grounded=grounded,
        )

        yield {
            "type": "final",
            "answer": answer_text,
            "grounded": grounded,
            "citations": [c.model_dump() for c in citations],
            "session_id": session.session_id,
            "usage": (generation.usage if generation else TokenUsage()).model_dump(),
            "model": generation.model if generation else "",
            "timings_ms": trace.finish(),
            "notes": [*retrieval.notes, *(generation.notes if generation else [])],
        }

    # ================================================================ helpers
    async def _generate(
        self,
        user_turn: str,
        *,
        session: Session,
        voice_mode: bool,
        max_tokens: Optional[int],
    ) -> GenerationResult:
        try:
            return await self.llm.generate(
                SYSTEM_PROMPT,
                user_turn,
                max_tokens=max_tokens or (256 if voice_mode else settings.llm_max_tokens),
                effort="low" if voice_mode else None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Generation failed: %s", exc, exc_info=True)
            return GenerationResult(
                text="",
                notes=[f"generation failed: {type(exc).__name__}: {exc}"],
                failed=True,
            )

    @staticmethod
    def _resolve_citations(
        answer: str,
        citation_meta: list[dict[str, Any]],
        chunks: list[RetrievedChunk],
    ) -> tuple[list[Citation], bool]:
        """Return only the citations the answer actually references.

        `grounded` is True when at least one marker was used and resolved. An
        answer that cites nothing is either a refusal ("I don't have that") or an
        ungrounded claim; both should be visibly distinct from a cited answer.
        """
        if not citation_meta:
            return [], False

        by_marker = {meta["marker"]: meta for meta in citation_meta}
        used_indices: list[int] = []
        for match in _MARKER_RE.finditer(answer):
            index = int(match.group(1))
            if 1 <= index <= len(citation_meta) and index not in used_indices:
                used_indices.append(index)

        if not used_indices:
            return [], False

        citations: list[Citation] = []
        for index in used_indices:
            meta = by_marker.get(f"[{index}]")
            if meta:
                citations.append(Citation(**meta))
        return citations, bool(citations)

    def _no_context_result(
        self,
        *,
        query: str,
        session: Session,
        retrieval: RetrievalResult,
        trace: Trace,
    ) -> AnswerResult:
        """Decline, deterministically, when retrieval returned no context."""
        answer_text = no_context_answer(retrieval.absent_entity)

        session.last_query = query
        session.last_effective_query = retrieval.effective_query
        session.last_chunk_ids = []
        session.add_turn("user", query)
        # `grounded=False` is the honest flag: there is no source, and the UI must
        # not decorate a refusal with a citation it cannot show.
        session.add_turn("assistant", answer_text, grounded=False)
        METRICS.incr("answer.no_context")

        return AnswerResult(
            answer=answer_text,
            session_id=session.session_id,
            query=query,
            effective_query=retrieval.effective_query,
            grounded=False,
            citations=[],
            retrieval=retrieval,
            model="refusal (no context retrieved)",
            notes=["no context retrieved; declined without generating", *retrieval.notes],
            timings_ms=trace.finish(),
        )

    def _extractive_result(
        self,
        *,
        query: str,
        session: Session,
        retrieval: RetrievalResult,
        citation_meta: list[dict[str, Any]],
        context_block: str,
        trace: Trace,
        reason: str,
    ) -> AnswerResult:
        """Compose a grounded answer straight from the retrieved text."""
        extracted = extractive_answer(retrieval.effective_query, retrieval.results)

        if extracted is None:
            answer_text, citations, grounded = fallback_answer(), [], False
        else:
            answer_text, marker_index = extracted
            if marker_index == 0:
                # A deliberate refusal (no record for the person asked about).
                # It must NOT be marked grounded — there is no source, and the UI
                # showing a citation here would imply we found something.
                citations, grounded = [], False
            else:
                citations, grounded = self._resolve_citations(
                    answer_text, citation_meta, retrieval.results
                )
                # The extractor emits exactly one marker; if resolution somehow
                # found none, cite the chunk we actually copied from rather than
                # claiming the answer is unsourced.
                if not citations and citation_meta:
                    citations = [Citation(**citation_meta[marker_index - 1])]
                    grounded = True

        session.last_query = query
        session.last_chunk_ids = [c.id for c in retrieval.results]
        session.add_turn("user", query)
        session.add_turn(
            "assistant",
            answer_text,
            citations=[c.model_dump() for c in citations],
            grounded=grounded,
        )
        METRICS.incr("answer.extractive")

        return AnswerResult(
            answer=answer_text,
            session_id=session.session_id,
            query=query,
            effective_query=retrieval.effective_query,
            grounded=grounded,
            citations=citations,
            retrieval=retrieval,
            context=context_block,
            model="extractive (no generation)",
            notes=[f"extractive fallback: {reason}", *retrieval.notes],
            timings_ms=trace.finish(),
        )

    def _from_cached(
        self,
        payload: dict[str, Any],
        session: Session,
        query: str,
        retrieval: RetrievalResult,
        similarity: float,
    ) -> AnswerResult:
        return AnswerResult(
            answer=payload.get("answer", ""),
            session_id=session.session_id,
            query=query,
            effective_query=retrieval.effective_query,
            grounded=bool(payload.get("grounded", True)),
            citations=[Citation(**c) for c in payload.get("citations", [])],
            retrieval=retrieval,
            model=payload.get("model", ""),
            cache_hit=True,
            notes=[f"answer cache hit (sim={similarity:.3f})", *retrieval.notes],
        )

    # ================================================================= health
    async def health(self) -> dict[str, Any]:
        llm_health, = await asyncio.gather(self.llm.health())
        return {
            "llm": llm_health,
            "retrieval": self.pipeline.health(),
            "sessions": self.sessions.count(),
            "answer_cache": self.answer_cache.stats(),
        }


_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _service
    if _service is None:
        _service = RAGService()
    return _service


def set_rag_service(service: Optional[RAGService]) -> None:
    global _service
    _service = service
