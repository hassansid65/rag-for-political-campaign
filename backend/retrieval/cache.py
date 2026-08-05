"""
Semantic result cache.

Exact-string caching is nearly useless for voice: ASR returns "what is amma vodi",
"whats amma vodi", "what is Amma Vodi?" for the same question, and every one of
those is a cache miss. Embedding-similarity keying catches all three.

We already have the query embedding in hand (retrieval needs it anyway), so a
lookup is one matmul against at most `semantic_cache_size` rows — tens of
microseconds. On a hit we skip the vector search, the fusion, and the
cross-encoder entirely, which is 60–90% of retrieval latency.

Correctness guards:
  * **Filter-scoped keys.** A cached result for `district=NTR` must never serve a
    `district=Guntur` query, so the filter signature is part of the key namespace.
  * **TTL.** Uploading a document invalidates answers; we both expire by time and
    expose `invalidate_all()` for the ingest path to call.
  * **High threshold (0.97).** Cosine between two genuinely different questions
    on the same topic is often 0.90+. A loose threshold here answers the wrong
    question confidently, which is far worse than a cache miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

import numpy as np

from core.config import settings
from core.latency import METRICS

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    vector: np.ndarray
    query: str
    value: T
    created_at: float = field(default_factory=time.time)
    hits: int = 0


class SemanticCache(Generic[T]):
    def __init__(
        self,
        threshold: Optional[float] = None,
        max_size: Optional[int] = None,
        ttl_s: Optional[int] = None,
    ) -> None:
        self.threshold = settings.semantic_cache_threshold if threshold is None else threshold
        self.max_size = settings.semantic_cache_size if max_size is None else max_size
        self.ttl_s = settings.semantic_cache_ttl_s if ttl_s is None else ttl_s

        # namespace -> list of entries. Namespace = filter signature.
        self._buckets: dict[str, list[_Entry[T]]] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self._generation = 0

    # ------------------------------------------------------------------ keys
    @staticmethod
    def namespace(filters: Optional[dict[str, Any]]) -> str:
        if not filters:
            return "_"
        # Sorted JSON so key order can't produce two namespaces for one filter.
        normalized = {
            k: sorted(v) if isinstance(v, (list, set, tuple)) else v
            for k, v in sorted(filters.items())
            if not k.startswith("_") and v not in (None, "", [], {})
        }
        if not normalized:
            return "_"
        blob = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------ read
    def get(
        self,
        vector: np.ndarray,
        filters: Optional[dict[str, Any]] = None,
    ) -> Optional[tuple[T, float, str]]:
        """Return (value, similarity, cached_query) on a hit."""
        if not settings.enable_semantic_cache or self.max_size <= 0:
            return None

        start = time.perf_counter()
        namespace = self.namespace(filters)
        with self._lock:
            entries = self._buckets.get(namespace)
            if not entries:
                self.misses += 1
                return None

            cutoff = time.time() - self.ttl_s
            entries = [e for e in entries if e.created_at >= cutoff]
            self._buckets[namespace] = entries
            if not entries:
                self.misses += 1
                return None

            query_vec = np.asarray(vector, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(query_vec))
            if norm < 1e-12:
                self.misses += 1
                return None
            query_vec = query_vec / norm

            matrix = np.vstack([e.vector for e in entries])
            sims = matrix @ query_vec
            best = int(np.argmax(sims))
            best_sim = float(sims[best])

            METRICS.observe("cache.lookup", (time.perf_counter() - start) * 1000)

            if best_sim >= self.threshold:
                entry = entries[best]
                entry.hits += 1
                self.hits += 1
                METRICS.incr("cache.hit")
                logger.debug("Semantic cache hit (sim=%.4f) for %r", best_sim, entry.query)
                return entry.value, best_sim, entry.query

            self.misses += 1
            METRICS.incr("cache.miss")
            return None

    # ----------------------------------------------------------------- write
    def put(
        self,
        vector: np.ndarray,
        query: str,
        value: T,
        filters: Optional[dict[str, Any]] = None,
    ) -> None:
        if not settings.enable_semantic_cache or self.max_size <= 0:
            return

        namespace = self.namespace(filters)
        query_vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query_vec))
        if norm < 1e-12:
            return
        query_vec = query_vec / norm

        with self._lock:
            bucket = self._buckets.setdefault(namespace, [])
            # Replace a near-identical existing entry rather than growing.
            if bucket:
                matrix = np.vstack([e.vector for e in bucket])
                sims = matrix @ query_vec
                idx = int(np.argmax(sims))
                if float(sims[idx]) >= 0.995:
                    bucket[idx] = _Entry(vector=query_vec, query=query, value=value)
                    return

            bucket.append(_Entry(vector=query_vec, query=query, value=value))

            total = sum(len(b) for b in self._buckets.values())
            if total > self.max_size:
                self._evict_oldest(total - self.max_size)

    def _evict_oldest(self, count: int) -> None:
        flat: list[tuple[float, str, int]] = []
        for ns, entries in self._buckets.items():
            for i, entry in enumerate(entries):
                flat.append((entry.created_at, ns, i))
        flat.sort()
        # Delete high indices first so earlier removals don't shift targets.
        for _, ns, idx in sorted(flat[:count], key=lambda x: -x[2]):
            bucket = self._buckets.get(ns)
            if bucket and idx < len(bucket):
                bucket.pop(idx)
        self._buckets = {k: v for k, v in self._buckets.items() if v}

    # ------------------------------------------------------------ management
    def invalidate_all(self) -> int:
        """Called after ingest/delete — stale answers are worse than slow ones."""
        with self._lock:
            count = sum(len(b) for b in self._buckets.values())
            self._buckets.clear()
            self._generation += 1
            logger.info("Semantic cache invalidated (%d entries dropped)", count)
            return count

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = sum(len(b) for b in self._buckets.values())
            requests = self.hits + self.misses
            return {
                "enabled": settings.enable_semantic_cache,
                "entries": total,
                "namespaces": len(self._buckets),
                "max_size": self.max_size,
                "threshold": self.threshold,
                "ttl_s": self.ttl_s,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / requests, 4) if requests else 0.0,
                "generation": self._generation,
            }


_retrieval_cache: Optional[SemanticCache] = None
_answer_cache: Optional[SemanticCache] = None
_lock = threading.Lock()


def get_retrieval_cache() -> SemanticCache:
    global _retrieval_cache
    if _retrieval_cache is None:
        with _lock:
            if _retrieval_cache is None:
                _retrieval_cache = SemanticCache()
    return _retrieval_cache


def get_answer_cache() -> SemanticCache:
    """Separate, stricter cache for full generated answers."""
    global _answer_cache
    if _answer_cache is None:
        with _lock:
            if _answer_cache is None:
                _answer_cache = SemanticCache(
                    threshold=max(0.985, settings.semantic_cache_threshold),
                    max_size=max(64, settings.semantic_cache_size // 2),
                )
    return _answer_cache


def invalidate_all_caches() -> dict[str, int]:
    return {
        "retrieval": get_retrieval_cache().invalidate_all(),
        "answer": get_answer_cache().invalidate_all(),
    }
