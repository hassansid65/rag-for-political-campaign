"""
Retrieval orchestration.

    utterance
       ↓  rewrite + filter inference        (rules; ~0.2 ms)
       ↓  semantic cache probe              (~0.05 ms — may short-circuit here)
       ↓  embed query variants              (BGE-small; 3–8 ms)
       ↓  dense HNSW  ∥  BM25 sparse        (run concurrently; 2–15 ms)
       ↓  Reciprocal Rank Fusion            (~0.1 ms)
       ↓  dedupe + metadata boost           (~0.1 ms)
       ↓  BGE cross-encoder rerank top-N    (40–120 ms — the expensive stage)
       ↓  threshold + parent expansion
       ↓  top-k chunks

Design decisions worth defending:

* **The two branches run concurrently**, not sequentially. They hit different
  indexes and neither depends on the other, so serializing them would add the
  smaller branch's latency for nothing.
* **Threshold *after* rerank, not before.** The dense score of the correct chunk
  is often mediocre; its cross-encoder score is not. Cutting on cosine first
  throws away chunks the reranker would have promoted to #1.
* **Widen, then narrow.** `candidate_top_k` (30/branch) → fuse → rerank → `top_k`
  (5). Recall is only recoverable at the widening stage; precision is only
  recoverable at the narrowing stage. Doing both cheaply is the whole point.
* **Degradation is explicit.** If the reranker is missing, or one branch errors,
  the pipeline still returns ranked results and says so in `notes`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from core.config import settings
from core.latency import METRICS, Trace
from core.schemas import Chunk, RetrievedChunk
from embeddings.embedder import Embedder, get_embedder
from ingestion.chunker import DocumentChunker
from memory.conversation import Session
from retrieval.cache import get_retrieval_cache
from retrieval.fusion import FusedHit, deduplicate_by_content, fuse_multi, reciprocal_rank_fusion
from retrieval.literals import describe, selective_literals
from retrieval.query_rewriter import (
    QueryRewriter,
    RewriteResult,
    apply_filters,
    name_match_score,
)
from retrieval.reranker import Reranker, get_reranker
from vectorstore.base import SearchHit, VectorStore
from vectorstore.factory import get_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    query: str
    effective_query: str
    results: list[RetrievedChunk] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    inferred_filters: dict[str, Any] = field(default_factory=dict)
    applied_filters: dict[str, Any] = field(default_factory=dict)
    total_candidates: int = 0
    reranked: bool = False
    cache_hit: bool = False
    cache_similarity: float = 0.0
    query_vector: Optional[np.ndarray] = None
    notes: list[str] = field(default_factory=list)
    timings_ms: dict[str, Any] = field(default_factory=dict)
    # Cache key scope: the applied filters PLUS any entity the query named. Kept
    # separate from `applied_filters` because it must never reach the vector store
    # as a filter expression — it exists only to partition the caches.
    cache_scope: dict[str, Any] = field(default_factory=dict)
    # The person the query named when the corpus holds no record for them. Lets
    # the answer layer decline *by name* instead of generically, which is the
    # difference between "I don't have a profile for X" and a vague non-answer.
    absent_entity: Optional[str] = None

    def context_chunks(self) -> list[RetrievedChunk]:
        return self.results

    def is_empty(self) -> bool:
        return not self.results


class RetrievalPipeline:
    def __init__(
        self,
        store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
        rewriter: Optional[QueryRewriter] = None,
    ) -> None:
        self.store = store or get_store()
        self.embedder = embedder or get_embedder()
        self.reranker = reranker or get_reranker()
        self.rewriter = rewriter or QueryRewriter()
        self.cache = get_retrieval_cache()

    # ==================================================================== main
    async def retrieve(
        self,
        query: str,
        *,
        session: Optional[Session] = None,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
        use_rerank: Optional[bool] = None,
        use_hybrid: Optional[bool] = None,
        rerank_mode: Optional[str] = None,
        rewrite: bool = True,
        voice_mode: bool = False,
        speculative: bool = False,
        trace: Optional[Trace] = None,
        use_cache: bool = True,
    ) -> RetrievalResult:
        trace = trace or Trace(name="retrieve")
        top_k = top_k or settings.retrieval_top_k

        # A live voice turn cannot afford the full cascade unless the work is
        # hidden behind speech (speculative=True), in which case latency is free.
        if rerank_mode is None:
            # `self.reranker.mode` is already resolved from `auto` against the host;
            # reading settings directly here would pass "auto" down to scoring.
            rerank_mode = (
                "fast" if (voice_mode and not speculative) else self.reranker.mode
            )
        threshold = (
            settings.similarity_threshold if similarity_threshold is None else similarity_threshold
        )
        do_rerank = (settings.enable_rerank if use_rerank is None else use_rerank) and (
            self.reranker.is_available
        )
        do_hybrid = settings.enable_hybrid if use_hybrid is None else use_hybrid

        # ---------------------------------------------------- 1. understand
        with trace.stage("rewrite"):
            # Persist a stated district onto the session *before* rewriting, so
            # "I'm from Vijayawada" then "what about schools there?" resolves.
            # This has to live here rather than only in RAGService, otherwise the
            # /retrieve endpoint silently loses conversational state that /query
            # keeps — the same session would behave differently per endpoint.
            if session is not None and not speculative:
                session.update_slots_from_text(query)

            if rewrite:
                rw = self.rewriter.rewrite(query, session=session, voice_mode=voice_mode)
            else:
                rw = RewriteResult(original=query, effective=query)
            applied = apply_filters(rw.inferred, filters)

        outcome = RetrievalResult(
            query=query,
            effective_query=rw.effective,
            variants=[v for v in rw.all_queries() if v != rw.effective],
            inferred_filters=rw.inferred,
            applied_filters=applied,
            notes=list(rw.notes),
        )

        # Partition the caches by the named entity. Without this, two questions
        # about DIFFERENT people embed at ~0.99 cosine on a template corpus — above
        # the 0.97 cache threshold — and the cache confidently returns the wrong
        # candidate's record. Raising the threshold cannot fix it: the questions
        # genuinely are near-identical strings. The scope must include the entity.
        outcome.cache_scope = dict(applied)
        if rw.inferred.get("person_hint"):
            outcome.cache_scope["person"] = str(rw.inferred["person_hint"]).lower()

        # ------------------------------------------------------- 2. embed
        queries = rw.all_queries()[:3]        # primary + up to 2 variants
        with trace.stage("embed"):
            vectors = self._embed_queries(queries)
        outcome.query_vector = vectors[0] if len(vectors) else None

        # ---------------------------------------------- 3. cache probe
        if use_cache and outcome.query_vector is not None:
            with trace.stage("cache"):
                cached = self.cache.get(outcome.query_vector, outcome.cache_scope)
            if cached is not None:
                payload, similarity, cached_query = cached
                outcome.results = [RetrievedChunk(**r) for r in payload]
                outcome.cache_hit = True
                outcome.cache_similarity = round(similarity, 4)
                outcome.reranked = True
                outcome.total_candidates = len(outcome.results)
                outcome.notes.append(f"semantic cache hit (sim={similarity:.3f})")
                outcome.timings_ms = trace.finish()
                METRICS.incr("retrieve.cache_hit")
                return outcome

        # --------------------------------- 3b. exact literal lookup (reverse)
        # Runs BEFORE the ANN search because for a value query the right record
        # frequently does not rank at all: the date contributes almost nothing to
        # the embedding, and BM25 spreads {14, october, 1985} across many records.
        # An exact hit here is authoritative and replaces the candidate set.
        literals = selective_literals(query)
        if literals:
            with trace.stage("literal"):
                literal_hits = await self._literal_lookup(literals, applied)
            outcome.inferred_filters = {
                **outcome.inferred_filters,
                "literals": [lit.raw for lit in literals],
            }
            if literal_hits:
                outcome.notes.append(
                    f"exact literal match ({describe(literals)}) → "
                    f"{len(literal_hits)} record(s); ANN search skipped"
                )
                METRICS.incr("retrieve.literal_hit")
                outcome.results = self._select(
                    [FusedHit(hit=hit, rrf_score=1.0) for hit in literal_hits],
                    rerank_scores={},
                    top_k=top_k,
                    threshold=0.0,
                    reranked=False,
                    notes=outcome.notes,
                )
                outcome.total_candidates = len(literal_hits)
                outcome.timings_ms = trace.finish()
                return outcome

            # A distinctive value that appears nowhere in the corpus. Returning
            # near-miss records is what produced "born 14 October 1985?" being
            # answered with a candidate born 7 September 1985 — so return nothing
            # and let the generator say it has no match.
            outcome.notes.append(
                f"no record contains {describe(literals)}; returning no context"
            )
            METRICS.incr("retrieve.literal_miss")
            outcome.timings_ms = trace.finish()
            return outcome

        # ------------------------------------------------- 4. search branches
        with trace.stage("search"):
            fused = await self._search_and_fuse(
                queries=queries,
                vectors=vectors,
                filters=applied,
                do_hybrid=do_hybrid,
                notes=outcome.notes,
            )

        # A district filter that returns nothing is usually over-constraint, not
        # absence of an answer. Retry unfiltered rather than telling a voter from
        # Vijayawada that we know nothing about roads.
        if not fused and applied:
            outcome.notes.append("no hits with filters; retried unfiltered")
            METRICS.incr("retrieve.filter_relaxed")
            with trace.stage("search_relaxed"):
                fused = await self._search_and_fuse(
                    queries=queries,
                    vectors=vectors,
                    filters=None,
                    do_hybrid=do_hybrid,
                    notes=outcome.notes,
                )
            outcome.applied_filters = {}

        outcome.total_candidates = len(fused)
        if not fused:
            outcome.timings_ms = trace.finish()
            return outcome

        # ------------------------------------- 5. dedupe + metadata boosting
        with trace.stage("fuse"):
            fused = deduplicate_by_content(fused)
            fused = self._apply_boosts(fused, rw, applied)
            fused = self._gate_to_entity(fused, rw, outcome)

        # ---------------------------------------------------- 6. rerank
        candidates = fused[: settings.rerank_candidate_cap]
        rerank_scores: dict[str, float] = {}
        if do_rerank and candidates:
            with trace.stage("rerank"):
                rerank_scores = await self._rerank(rw.effective, candidates, rerank_mode)
            outcome.notes.append(f"rerank mode={rerank_mode}")
            if rerank_scores:
                outcome.reranked = True
                candidates.sort(
                    key=lambda f: rerank_scores.get(f.id, float("-inf")), reverse=True
                )
            else:
                outcome.notes.append("reranker unavailable; fusion order kept")

        # ---------------------------------------------- 7. threshold + select
        with trace.stage("select"):
            selected = self._select(
                candidates,
                rerank_scores=rerank_scores,
                top_k=top_k,
                threshold=threshold,
                reranked=outcome.reranked,
                notes=outcome.notes,
            )
            outcome.results = selected

        # ------------------------------------------------------- 8. cache put
        if use_cache and outcome.query_vector is not None and selected:
            self.cache.put(
                outcome.query_vector,
                rw.effective,
                [r.model_dump() for r in selected],
                outcome.cache_scope,
            )

        trace.set(
            candidates=outcome.total_candidates,
            returned=len(selected),
            reranked=outcome.reranked,
            filters=applied,
        )
        outcome.timings_ms = trace.finish()
        return outcome

    # ================================================================ stages
    def _embed_queries(self, queries: list[str]) -> np.ndarray:
        if len(queries) == 1:
            return self.embedder.encode_query(queries[0]).reshape(1, -1)
        # Batch the variants — one forward pass beats three.
        primary = self.embedder.encode_query(queries[0]).reshape(1, -1)
        rest = self.embedder.encode_queries(queries[1:])
        return np.vstack([primary, rest]) if len(rest) else primary

    async def _search_and_fuse(
        self,
        *,
        queries: list[str],
        vectors: np.ndarray,
        filters: Optional[dict[str, Any]],
        do_hybrid: bool,
        notes: list[str],
    ) -> list[FusedHit]:
        store_filters = {k: v for k, v in (filters or {}).items() if not k.startswith("_")}
        store_filters = store_filters or None
        k = settings.candidate_top_k
        loop = asyncio.get_running_loop()

        # Build the concurrent task set: dense per query variant + one sparse.
        tasks: list[asyncio.Future] = []
        labels: list[tuple[str, float]] = []

        for i, _query in enumerate(queries):
            vector = vectors[i] if i < len(vectors) else vectors[0]
            tasks.append(
                loop.run_in_executor(
                    None, self.store.search_dense, vector, k, store_filters
                )
            )
            # Variants contribute at reduced weight — they are guesses.
            labels.append(("dense", settings.dense_weight if i == 0 else settings.dense_weight * 0.5))

        if do_hybrid:
            tasks.append(
                loop.run_in_executor(
                    None, self.store.search_sparse, queries[0], k, store_filters
                )
            )
            labels.append(("sparse", settings.sparse_weight))

        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        branches: list[tuple[list[SearchHit], float]] = []
        for (kind, weight), result in zip(labels, gathered):
            if isinstance(result, BaseException):
                logger.warning("%s branch failed: %s", kind, result)
                notes.append(f"{kind} branch failed: {type(result).__name__}")
                METRICS.incr(f"retrieve.{kind}_error")
                continue
            branches.append((result, weight))

        if not branches:
            return []
        if len(branches) == 1:
            hits, _ = branches[0]
            return reciprocal_rank_fusion(
                hits if hits and hits[0].retriever == "dense" else [],
                hits if hits and hits[0].retriever == "sparse" else [],
            )
        return fuse_multi(branches)

    async def _literal_lookup(
        self,
        literals: list,
        filters: dict[str, Any],
    ) -> list[SearchHit]:
        """Records containing every extracted literal (conjunctive)."""
        store_filters = {
            k: v for k, v in (filters or {}).items() if not k.startswith("_")
        } or None
        loop = asyncio.get_running_loop()

        # Search on the most selective literal, then require the rest to be
        # present too — a query naming both a date and an amount must match the
        # single record carrying both, not either one.
        primary, *rest = literals
        hits: list[SearchHit] = await loop.run_in_executor(
            None,
            self.store.find_literal,
            primary.variants,
            max(20, settings.retrieval_top_k * 4),
            store_filters,
        )
        if not hits or not rest:
            return hits

        narrowed = [
            hit for hit in hits if all(lit.matches(hit.text) for lit in rest)
        ]
        # If nothing satisfies all of them, the primary match is still a real
        # exact hit and is better than falling through to fuzzy retrieval.
        return narrowed or hits

    async def _rerank(
        self,
        query: str,
        candidates: list[FusedHit],
        mode: Optional[str] = None,
    ) -> dict[str, float]:
        # Score the *child* chunk, not the parent window. The parent window is
        # ~1800 chars (~450 tokens) versus ~700 chars (~175 tokens) for the child,
        # and cross-encoder cost scales with sequence length — feeding parents cost
        # 304ms/pair instead of 117ms/pair for no measured ranking gain, because
        # the extra text is mostly neighbouring topics.
        passages = [c.hit.text for c in candidates]
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None, self.reranker.score, query, passages, mode
        )
        if not scores or len(scores) != len(candidates):
            return {}
        return {c.id: float(s) for c, s in zip(candidates, scores)}

    def _apply_boosts(
        self,
        fused: list[FusedHit],
        rw: RewriteResult,
        applied: dict[str, Any],
    ) -> list[FusedHit]:
        """Nudge — never filter — on soft signals.

        `category_hint` and `topic_hint` are guesses from keyword overlap. Using
        them as hard filters empties result sets; using them as a small additive
        boost lets a strong lexical/semantic match still win.
        """
        category_hint = rw.inferred.get("category_hint")
        topic_hint = rw.inferred.get("topic_hint")
        person_hint = rw.inferred.get("person_hint")
        district = applied.get("district")
        if not any([category_hint, topic_hint, district, person_hint]):
            return fused

        top_rrf = max((f.rrf_score for f in fused), default=1.0) or 1.0

        # An entity name in the query is treated as near-decisive rather than as a
        # nudge. On the candidate corpus the wrong record scores ~0.95 cosine
        # against the right one, so a 10% boost cannot separate them — but the
        # name either matches a record or it does not. Matching records are lifted
        # above the whole fused set; non-matching *records* are pushed down, while
        # non-record chunks (manifesto, FAQ) are left alone because they may still
        # legitimately answer a question that happens to mention a person.
        if person_hint:
            scored: list[tuple[FusedHit, float]] = [
                (item, name_match_score(person_hint, item.hit.metadata.record_name))
                for item in fused
            ]
            # 0.6 is above the 0.5 that a shared-surname-only match scores under
            # F1, so a different person with the same surname is not boosted.
            if any(score >= 0.6 for _, score in scored):
                for item, score in scored:
                    if score >= 0.6:
                        item.rrf_score += top_rrf * (2.0 + score)
                    elif item.hit.metadata.is_record:
                        item.rrf_score -= top_rrf * 0.75
                METRICS.incr("retrieve.person_matched")

        for item in fused:
            meta = item.hit.metadata
            boost = 0.0
            if category_hint and meta.category == category_hint:
                boost += 0.15
            if topic_hint and (meta.topic == topic_hint or topic_hint in (meta.topics or [])):
                boost += 0.10
            if district and meta.district == district:
                # Primary-district chunks beat chunks that merely mention it.
                boost += 0.12
            if boost:
                item.rrf_score += top_rrf * boost

        return sorted(fused, key=lambda f: f.rrf_score, reverse=True)

    def _gate_to_entity(
        self,
        fused: list[FusedHit],
        rw: RewriteResult,
        outcome: RetrievalResult,
    ) -> list[FusedHit]:
        """Drop other people's records when the query is about one named person.

        Boosting the right record to rank 1 is not enough. This corpus contains
        deliberate near-namesakes — "Smt. Sarojini Vasireddy" alongside
        "Smt. Padmavathi Vasireddy" and "Smt. Rajeswari Vasireddy" — so a top-5
        context for one of them contains three different asset declarations. No
        prompt reliably survives that: the wrong figures are right there, under a
        heading that looks correct.

        So for entity-scoped questions we make retrieval *precise* rather than
        high-recall: keep only the best-matching record (plus any tied for best,
        which is genuine ambiguity the model should surface). Non-record chunks
        stay — a manifesto paragraph can still be a legitimate part of the answer.

        Recall is not lost, only re-targeted: if the name matches nothing, the gate
        does not engage and normal top-k behaviour applies.
        """
        person = rw.inferred.get("person_hint")
        if not person:
            return fused

        scored = [
            (item, name_match_score(person, item.hit.metadata.record_name))
            for item in fused
        ]
        best = max((score for _, score in scored), default=0.0)
        if best < 0.6:
            # The query named a person we hold no record for. Drop the record
            # chunks entirely rather than passing five arbitrary profiles.
            #
            # Prompting alone is not enough here: told only "say you don't have
            # their details", GPT-4 complied and then helpfully added "what I can
            # tell you is that <different candidate> has declared…". Technically
            # obedient, and exactly the sentence that makes a listener attribute
            # those assets to the person they asked about. With no other profile
            # in context there is nothing to volunteer.
            #
            # Non-record chunks stay: "what does the manifesto say about Patel's
            # district" can still be answerable.
            non_records = [f for f in fused if not f.hit.metadata.is_record]
            dropped = len(fused) - len(non_records)
            if dropped:
                outcome.absent_entity = person
                outcome.notes.append(
                    f"entity gate: no record matches '{person}' "
                    f"(best={best:.2f}); dropped all {dropped} profile(s)"
                )
                METRICS.incr("retrieve.entity_absent")
            return non_records

        # Ties within a small margin are real ambiguity ("which Sarojini?").
        keep_threshold = best - 0.15
        kept: list[FusedHit] = []
        dropped_records = 0
        for item, score in scored:
            if score >= keep_threshold:
                kept.append(item)
            elif item.hit.metadata.is_record:
                dropped_records += 1
            else:
                kept.append(item)   # non-record context is still admissible

        matched = sorted(
            {
                item.hit.metadata.record_name
                for item, score in scored
                if score >= keep_threshold and item.hit.metadata.record_name
            }
        )
        outcome.notes.append(
            f"entity gate: '{person}' → {matched} "
            f"(dropped {dropped_records} other record(s))"
        )
        METRICS.incr("retrieve.entity_gated")
        if len(matched) > 1:
            METRICS.incr("retrieve.entity_ambiguous")
        return kept

    def _select(
        self,
        candidates: list[FusedHit],
        *,
        rerank_scores: dict[str, float],
        top_k: int,
        threshold: float,
        reranked: bool,
        notes: list[str],
    ) -> list[RetrievedChunk]:
        selected: list[RetrievedChunk] = []
        budget = settings.max_context_chars
        used_chars = 0
        dropped_low = 0

        for item in candidates:
            if len(selected) >= top_k:
                break

            rerank_score = rerank_scores.get(item.id)
            dense = item.dense_score

            # Relevance gate: prefer the cross-encoder's verdict when we have it.
            if reranked and rerank_score is not None:
                if rerank_score < settings.rerank_score_threshold:
                    dropped_low += 1
                    continue
                final_score = self.reranker.to_probability(rerank_score)
            else:
                if dense is not None and dense < threshold:
                    dropped_low += 1
                    continue
                final_score = dense if dense is not None else item.rrf_score

            child_text = item.hit.text
            text = child_text
            if settings.enable_parent_expansion and item.hit.parent_text:
                text = item.hit.parent_text

            # Context budget: better to return 3 complete chunks than 5 truncated.
            if used_chars + len(text) > budget:
                if not selected:
                    text = text[:budget]
                else:
                    continue
            used_chars += len(text)

            selected.append(
                RetrievedChunk(
                    id=item.id,
                    text=text,
                    chunk_text=child_text,
                    metadata=item.hit.metadata,
                    score=round(float(final_score), 4),
                    dense_score=round(dense, 4) if dense is not None else None,
                    sparse_score=round(item.sparse_score, 4) if item.sparse_score is not None else None,
                    rrf_score=round(item.rrf_score, 6),
                    rerank_score=round(rerank_score, 4) if rerank_score is not None else None,
                    retriever=item.retriever_label,
                )
            )

        if dropped_low:
            notes.append(f"dropped {dropped_low} below-threshold candidate(s)")

        # Nothing cleared the bar but candidates existed: return the single best
        # so the LLM can decide it's insufficient, rather than fabricating from
        # an empty context. The score is reported honestly and stays low.
        #
        # Except when the best candidate is not merely weak but *unrelated*. The
        # cross-encoder distinguishes those two cases by a wide margin (see
        # `rerank_abandon_margin`), and conflating them is how "how do I cook
        # biryani" came back with a candidate's constituency profile: nothing
        # cleared the threshold, so the pipeline dutifully handed over the least
        # irrelevant of 56 profiles and the extractor read it out. For an
        # off-topic question the honest context is no context.
        if not selected and candidates and reranked:
            best_score = max(
                (rerank_scores[c.id] for c in candidates if c.id in rerank_scores),
                default=None,
            )
            floor = settings.rerank_score_threshold - settings.rerank_abandon_margin
            if best_score is not None and best_score < floor:
                notes.append(
                    f"best candidate scored {best_score:.1f}, below the "
                    f"abandon floor {floor:.1f}; returning no context"
                )
                METRICS.incr("retrieve.off_topic")
                return []

        if not selected and candidates:
            best = candidates[0]
            notes.append("all candidates below threshold; returning best-effort top-1")
            METRICS.incr("retrieve.below_threshold_fallback")
            selected.append(
                RetrievedChunk(
                    id=best.id,
                    text=(best.hit.parent_text or best.hit.text)[:budget],
                    chunk_text=best.hit.text,
                    metadata=best.hit.metadata,
                    score=round(float(best.dense_score or best.rrf_score), 4),
                    dense_score=best.dense_score,
                    sparse_score=best.sparse_score,
                    rrf_score=round(best.rrf_score, 6),
                    rerank_score=rerank_scores.get(best.id),
                    retriever=best.retriever_label,
                )
            )

        return selected

    # ============================================================== ingestion
    async def index_chunks(self, chunks: list[Chunk], trace: Optional[Trace] = None) -> int:
        """Embed and upsert chunks. Runs off the event loop."""
        if not chunks:
            return 0
        trace = trace or Trace(name="index")
        loop = asyncio.get_running_loop()

        with trace.stage("embed_passages"):
            # Embed the contextual header + chunk text; see DocumentChunker.
            texts = [DocumentChunker.embedding_text(c) for c in chunks]
            vectors = await loop.run_in_executor(None, self.embedder.encode_passages, texts)

        with trace.stage("upsert"):
            self.store.ensure_collection(self.embedder.dim)
            count = await loop.run_in_executor(None, self.store.upsert, chunks, vectors)

        # New documents invalidate cached answers about the old corpus.
        self.cache.invalidate_all()
        return count

    # ================================================================= health
    def health(self) -> dict[str, Any]:
        return {
            "store": self.store.health(),
            "embedder": self.embedder.health(),
            "reranker": self.reranker.health(),
            "cache": self.cache.stats(),
        }


_pipeline: Optional[RetrievalPipeline] = None


def get_pipeline() -> RetrievalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RetrievalPipeline()
    return _pipeline


def set_pipeline(pipeline: Optional[RetrievalPipeline]) -> None:
    """Injection point for startup wiring and tests."""
    global _pipeline
    _pipeline = pipeline
