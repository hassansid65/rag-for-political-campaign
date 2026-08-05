"""
In-process BM25 (Okapi) index.

Why hand-roll it instead of taking `rank_bm25`: we need (a) incremental adds
without rebuilding the whole index on every upload, (b) metadata pre-filtering
so BM25 and dense search see the *same* candidate universe, and (c) disk
persistence so a restart doesn't lose the keyword branch. `rank_bm25` gives none
of those and recomputes IDF over the full corpus per query.

BM25 earns its place in this system specifically: voters ask about proper nouns —
scheme names, district names, amounts, "Amma Vodi", "Rythu Bharosa". Dense
retrieval blurs rare tokens; BM25 nails exact matches. Neither alone is enough,
which is the whole argument for hybrid + RRF.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Keep digits and rupee amounts as tokens — "15000" and "Rs" are query-relevant.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[ఀ-౿]+|[ऀ-ॿ]+", re.IGNORECASE)

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "of", "in",
    "on", "at", "to", "for", "from", "by", "with", "about", "as", "is", "are",
    "was", "were", "be", "been", "being", "do", "does", "did", "have", "has",
    "had", "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those", "there", "here",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "s", "t", "am", "im", "ive", "dont", "please", "tell", "know", "want",
}

# Light suffix stripping. A full Porter stemmer over-stems proper nouns
# ("Konaseema" -> "Konaseem"), which is exactly what we must not break here.
_SUFFIXES = ("ies", "ing", "ed", "es", "s")


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if drop_stopwords and raw in _STOPWORDS:
            continue
        if len(raw) > 4:
            for suffix in _SUFFIXES:
                if raw.endswith(suffix) and len(raw) - len(suffix) >= 3:
                    raw = raw[: -len(suffix)]
                    break
        tokens.append(raw)
    return tokens


class BM25Index:
    """Incremental BM25 over chunk texts, with metadata-aware pre-filtering."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, path: Optional[Path] = None) -> None:
        self.k1 = k1
        self.b = b
        self.path = path
        self._lock = threading.RLock()

        # doc_key -> {term: freq}
        self._tf: dict[str, dict[str, int]] = {}
        self._length: dict[str, int] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._meta: dict[str, dict[str, Any]] = {}
        self._total_length = 0

    # ------------------------------------------------------------------ write
    def add(self, doc_key: str, text: str, meta: Optional[dict[str, Any]] = None) -> None:
        with self._lock:
            if doc_key in self._tf:
                self._remove_unlocked(doc_key)

            tokens = tokenize(text)
            if not tokens:
                # Still register the doc so filters and counts stay consistent.
                self._tf[doc_key] = {}
                self._length[doc_key] = 0
                self._meta[doc_key] = meta or {}
                return

            freqs: dict[str, int] = defaultdict(int)
            for token in tokens:
                freqs[token] += 1

            self._tf[doc_key] = dict(freqs)
            self._length[doc_key] = len(tokens)
            self._total_length += len(tokens)
            self._meta[doc_key] = meta or {}
            for term in freqs:
                self._postings[term].add(doc_key)

    def add_many(self, items: Iterable[tuple[str, str, dict[str, Any]]]) -> None:
        for doc_key, text, meta in items:
            self.add(doc_key, text, meta)

    def remove(self, doc_key: str) -> bool:
        with self._lock:
            return self._remove_unlocked(doc_key)

    def _remove_unlocked(self, doc_key: str) -> bool:
        if doc_key not in self._tf:
            return False
        for term in self._tf[doc_key]:
            bucket = self._postings.get(term)
            if bucket is not None:
                bucket.discard(doc_key)
                if not bucket:
                    del self._postings[term]
        self._total_length -= self._length.get(doc_key, 0)
        self._tf.pop(doc_key, None)
        self._length.pop(doc_key, None)
        self._meta.pop(doc_key, None)
        return True

    def remove_by(self, predicate) -> int:
        with self._lock:
            targets = [k for k, m in self._meta.items() if predicate(m)]
            for key in targets:
                self._remove_unlocked(key)
            return len(targets)

    def clear(self) -> None:
        with self._lock:
            self._tf.clear()
            self._length.clear()
            self._postings.clear()
            self._meta.clear()
            self._total_length = 0

    # ------------------------------------------------------------------- read
    @property
    def size(self) -> int:
        return len(self._tf)

    @property
    def avg_length(self) -> float:
        return (self._total_length / len(self._tf)) if self._tf else 0.0

    def search(
        self,
        query: str,
        top_k: int = 30,
        allowed: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """Top-k (doc_key, bm25_score). `allowed` restricts to a pre-filtered set."""
        with self._lock:
            terms = tokenize(query)
            if not terms or not self._tf:
                return []

            n_docs = len(self._tf)
            avgdl = self.avg_length or 1.0
            scores: dict[str, float] = defaultdict(float)

            # Query-side term frequency: repeating a term should count, but with
            # diminishing weight, hence the sqrt.
            qtf: dict[str, int] = defaultdict(int)
            for term in terms:
                qtf[term] += 1

            for term, q_count in qtf.items():
                postings = self._postings.get(term)
                if not postings:
                    continue
                df = len(postings)
                # Robertson-Sparck-Jones IDF with the +1 guard that keeps it
                # non-negative for terms present in > half the corpus.
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                q_weight = math.sqrt(q_count)

                candidates = postings if allowed is None else (postings & allowed)
                for doc_key in candidates:
                    tf = self._tf[doc_key].get(term, 0)
                    if not tf:
                        continue
                    dl = self._length[doc_key] or 1
                    denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                    scores[doc_key] += idf * q_weight * (tf * (self.k1 + 1.0)) / denom

            if not scores:
                return []
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            return ranked[:top_k]

    def meta(self, doc_key: str) -> dict[str, Any]:
        return self._meta.get(doc_key, {})

    def keys_matching(self, predicate) -> set[str]:
        with self._lock:
            return {k for k, m in self._meta.items() if predicate(m)}

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "documents": len(self._tf),
                "vocabulary": len(self._postings),
                "avg_doc_length": round(self.avg_length, 2),
                "k1": self.k1,
                "b": self.b,
            }

    # ------------------------------------------------------------ persistence
    def save(self, path: Optional[Path] = None) -> None:
        target = path or self.path
        if target is None:
            return
        with self._lock:
            payload = {
                "k1": self.k1,
                "b": self.b,
                "tf": self._tf,
                "length": self._length,
                "meta": self._meta,
                "total_length": self._total_length,
            }
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(target)   # atomic — a crash mid-write can't corrupt the index

    def load(self, path: Optional[Path] = None) -> bool:
        target = path or self.path
        if target is None or not target.exists():
            return False
        try:
            with target.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 index at %s is unreadable (%s); rebuilding", target, exc)
            return False

        with self._lock:
            self.k1 = payload.get("k1", self.k1)
            self.b = payload.get("b", self.b)
            self._tf = payload.get("tf", {})
            self._length = payload.get("length", {})
            self._meta = payload.get("meta", {})
            self._total_length = payload.get("total_length", sum(self._length.values()))
            self._postings = defaultdict(set)
            for doc_key, freqs in self._tf.items():
                for term in freqs:
                    self._postings[term].add(doc_key)
        logger.info("Loaded BM25 index: %d docs, %d terms", len(self._tf), len(self._postings))
        return True

    def to_sparse_vector(self, text: str, vocab_size: int = 65536) -> dict[int, float]:
        """Hashed sparse vector, for stores that accept a sparse field directly."""
        freqs: dict[int, float] = defaultdict(float)
        for term in tokenize(text):
            freqs[hash(term) % vocab_size] += 1.0
        return dict(freqs)

    def debug_dump(self, path: Path, limit: int = 50) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "stats": self.stats(),
                    "sample_keys": list(self._tf)[:limit],
                },
                fh,
                indent=2,
            )
