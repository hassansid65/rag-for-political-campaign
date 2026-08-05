"""
Cross-encoder reranking — cascaded, because a single BGE pass is too slow on CPU.

## Why reranking at all

The bi-encoder that powers retrieval compresses a chunk to 384 numbers *before* it
ever sees the query, so it can only measure rough topical overlap. A cross-encoder
reads query and chunk together and scores their actual relationship — it can tell
"eligibility for Amma Vodi" from "eligibility for Rythu Bharosa", which cosine
similarity on 384 dims often cannot. In this corpus it moves the correct chunk to
rank 1 with a score near 1.0 while pushing topical-but-wrong chunks below 0.05.

## Why it is cascaded

Measured on this machine (12 cores, CPU-only, `scripts/bench_rerank.py`):

    model                                 max_len  pairs   total    per pair
    BAAI/bge-reranker-base                    512     20   6082ms     304ms
    BAAI/bge-reranker-base                    256     20   2417ms     121ms
    BAAI/bge-reranker-base                    256     12   1400ms     117ms
    cross-encoder/ms-marco-MiniLM-L-6-v2      256     20    376ms      19ms
    cross-encoder/ms-marco-MiniLM-L-6-v2      256     12    207ms      17ms

BGE-reranker-base is a 278M-parameter model; MiniLM-L-6 is 22M. Six times cheaper
per pair, and on a corpus this size its ordering agrees with BGE's on the top few
almost always. So we cascade:

    20 fused candidates
      → tier 1 (MiniLM, all 20)          ~380ms, cheap and broad
      → keep top 6
      → tier 2 (BGE-reranker-base, 6)    ~630ms, expensive and precise
      → top 3

That buys BGE-quality ordering of the final shortlist at roughly half the cost of
running BGE over everything, and `RERANK_MODE` lets you pick a point on the curve:

    fast     tier 1 only              ~200–380ms   voice turns without speculation
    cascade  tier 1 → tier 2          ~1.0s        default; best quality per ms
    single   tier 2 only              ~1.4–2.4s    quality-first, latency ignored

## Why the slow default is still fine for voice

In the voice path the reranker runs *during* speculative retrieval — while the
citizen is still speaking (see `voice/streaming.py`). Its cost lands in a window
that was idle anyway, so it never shows up in the post-speech latency the caller
actually experiences. `voice_mode` without a usable speculation falls back to
`fast`, so a cold turn degrades in latency rather than in correctness.

Scores are raw logits (roughly -10..+10), kept raw for threshold comparisons;
`to_probability` sigmoids them for display.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from core.config import settings
from core.latency import METRICS
from core.model_loading import MODEL_LOAD_LOCK
from core.resources import probe, resolve_device, resolve_rerank_mode

logger = logging.getLogger(__name__)


def _pin_torch_threads() -> None:
    """Use physical cores, not hyperthreads.

    Torch defaults to logical-core count, which oversubscribes on
    hyperthreaded CPUs and makes short cross-encoder batches *slower*.
    """
    try:
        import torch

        cores = os.cpu_count() or 4
        torch.set_num_threads(max(1, cores // 2))
    except Exception:  # noqa: BLE001
        pass


@dataclass
class _Tier:
    """One cross-encoder stage in the cascade."""

    model_name: str
    max_length: int
    keep: Optional[int] = None      # candidates passed to the next tier
    device: str = "cpu"

    model: object | None = None
    failed: bool = False


class Reranker:
    """Cascaded cross-encoder reranker with per-request mode override."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        fast_model: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
        self.device = resolve_device(device or settings.reranker_device)
        self.max_length = max_length or settings.reranker_max_length

        # `auto` resolves against the actual host — see core/resources.py for why
        # a memory-tight box must not load the precise tier.
        self.mode, self.mode_reason = resolve_rerank_mode(mode or settings.rerank_mode)
        logger.info("Rerank mode: %s", self.mode_reason)

        # Tier 1: cheap and broad. Tier 2: expensive and precise.
        self.fast_tier = _Tier(
            model_name=fast_model or settings.reranker_fast_model,
            max_length=min(self.max_length, 256),
            keep=settings.rerank_cascade_keep,
            device=self.device,
        )
        self.precise_tier = _Tier(
            model_name=model_name or settings.reranker_model,
            max_length=self.max_length,
            keep=None,
            device=self.device,
        )

        self._lock = threading.Lock()
        self._threads_pinned = False

    # ------------------------------------------------------------------ load
    def _load_tier(self, tier: _Tier) -> None:
        if tier.model is not None or tier.failed:
            return
        with self._lock:
            if tier.model is not None or tier.failed:
                return
            if not self._threads_pinned:
                _pin_torch_threads()
                self._threads_pinned = True

            start = time.perf_counter()
            try:
                from sentence_transformers import CrossEncoder

                # Serialized against every other model construction in the
                # process — see core/model_loading.py.
                with MODEL_LOAD_LOCK:
                    tier.model = CrossEncoder(
                        tier.model_name, max_length=tier.max_length, device=tier.device
                    )
                tier.model.predict(  # type: ignore[union-attr]
                    [("warmup query", "warmup passage")], show_progress_bar=False
                )
                logger.info(
                    "Reranker tier ready: %s (max_len=%d) in %.2fs",
                    tier.model_name, tier.max_length, time.perf_counter() - start,
                )
            except Exception as exc:  # noqa: BLE001
                # A missing reranker must degrade ranking, never take the service
                # down — fusion order is a usable fallback.
                tier.failed = True
                logger.error(
                    "Reranker tier %s unavailable (%s); continuing without it",
                    tier.model_name, exc,
                )

    def load(self) -> None:
        """Preload whichever tiers the configured mode will actually use."""
        if self.mode in {"cascade", "fast"}:
            self._load_tier(self.fast_tier)
        if self.mode in {"cascade", "single"}:
            self._load_tier(self.precise_tier)

    @property
    def is_ready(self) -> bool:
        return any(t.model is not None for t in (self.fast_tier, self.precise_tier))

    @property
    def is_available(self) -> bool:
        return not (self.fast_tier.failed and self.precise_tier.failed)

    # --------------------------------------------------------------- scoring
    def score(
        self,
        query: str,
        passages: Sequence[str],
        mode: Optional[str] = None,
    ) -> list[float]:
        """Score every passage. Returns [] if no tier is usable.

        In cascade mode the returned list still has one score per input passage:
        candidates eliminated by tier 1 keep their (lower) tier-1 score, rescaled
        to sit strictly below every tier-2 score so the merged ordering is stable.
        """
        if not passages:
            return []

        effective_mode = (mode or self.mode).lower()
        start = time.perf_counter()

        if effective_mode == "single":
            scores = self._score_tier(self.precise_tier, query, passages)
            if not scores:
                scores = self._score_tier(self.fast_tier, query, passages)
        elif effective_mode == "fast":
            scores = self._score_tier(self.fast_tier, query, passages)
            if not scores:
                scores = self._score_tier(self.precise_tier, query, passages)
        else:
            scores = self._score_cascade(query, passages)

        if not scores:
            return []

        elapsed = (time.perf_counter() - start) * 1000
        METRICS.observe(f"rerank.{effective_mode}", elapsed)
        METRICS.observe("rerank.per_pair", elapsed / max(1, len(passages)))
        return scores

    def _score_cascade(self, query: str, passages: Sequence[str]) -> list[float]:
        coarse = self._score_tier(self.fast_tier, query, passages)
        if not coarse:
            # Tier 1 gone — fall back to tier 2 over everything.
            return self._score_tier(self.precise_tier, query, passages)

        keep = self.fast_tier.keep or settings.rerank_cascade_keep
        if keep >= len(passages) or self.precise_tier.failed:
            return coarse

        shortlist = sorted(range(len(passages)), key=lambda i: coarse[i], reverse=True)[:keep]
        precise = self._score_tier(
            self.precise_tier, query, [passages[i] for i in shortlist]
        )
        if not precise:
            return coarse

        # Merge: tier-2 scores win outright; everything else is compressed below
        # the tier-2 floor so a high tier-1 score can't outrank a rescored chunk.
        floor = min(precise)
        merged = list(coarse)
        coarse_max = max(coarse) or 1.0
        for i in range(len(merged)):
            if i not in set(shortlist):
                # Map tier-1 range into (-inf, floor) preserving relative order.
                merged[i] = floor - 1.0 - (coarse_max - merged[i])
        for position, index in enumerate(shortlist):
            merged[index] = precise[position]

        METRICS.incr("rerank.cascade_used")
        return merged

    def _score_tier(
        self, tier: _Tier, query: str, passages: Sequence[str]
    ) -> list[float]:
        self._load_tier(tier)
        if tier.model is None:
            return []

        start = time.perf_counter()
        try:
            pairs = [(query, self._truncate(p, tier.max_length)) for p in passages]
            raw = tier.model.predict(  # type: ignore[union-attr]
                pairs,
                batch_size=min(16, len(pairs)),
                show_progress_bar=False,
            )
            scores = [float(s) for s in raw]
        except Exception as exc:  # noqa: BLE001
            logger.error("Rerank tier %s failed (%s)", tier.model_name, exc)
            return []

        METRICS.observe(f"rerank.tier.{tier.model_name.split('/')[-1]}",
                        (time.perf_counter() - start) * 1000)
        return scores

    @staticmethod
    def _truncate(text: str, max_length: int) -> str:
        # ~4 chars/token; cheaper than tokenizing just to measure. Passing text
        # longer than the model's window costs full attention time for tokens
        # that get dropped, which is where the 512/2048 config lost 300ms/pair.
        budget = max_length * 4
        return text if len(text) <= budget else text[:budget]

    @staticmethod
    def to_probability(logit: float) -> float:
        """Sigmoid for display — a raw logit of 3.1 means nothing to a reviewer."""
        try:
            return 1.0 / (1.0 + math.exp(-logit))
        except OverflowError:
            return 0.0 if logit < 0 else 1.0

    # ------------------------------------------------------------------ health
    def health(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "mode_reason": self.mode_reason,
            "host": probe().as_dict(),
            "model": self.precise_tier.model_name,
            "fast_model": self.fast_tier.model_name,
            "cascade_keep": self.fast_tier.keep,
            "device": self.device,
            "max_length": self.max_length,
            "ready": self.is_ready,
            "available": self.is_available,
            "tiers": {
                "fast": {
                    "model": self.fast_tier.model_name,
                    "loaded": self.fast_tier.model is not None,
                    "failed": self.fast_tier.failed,
                },
                "precise": {
                    "model": self.precise_tier.model_name,
                    "loaded": self.precise_tier.model is not None,
                    "failed": self.precise_tier.failed,
                },
            },
        }


_reranker: Optional[Reranker] = None
_reranker_lock = threading.Lock()


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = Reranker()
    return _reranker
