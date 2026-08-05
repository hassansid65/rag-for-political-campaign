"""
Reciprocal Rank Fusion.

RRF combines ranked lists using *rank*, not score:

    score(d) = Σ_lists  weight_i / (k + rank_i(d))

That property is the whole reason to use it here. Our two branches produce
incomparable numbers — dense similarity lives in [0, 1], BM25 is unbounded and
corpus-dependent. Min-max normalizing them before a weighted sum makes the fusion
sensitive to the *spread* of each batch: one outlier BM25 score squashes every
other keyword result toward zero. RRF only asks "how highly did each branch rank
this?", so it is stable across queries with no tuning per corpus.

`k` (default 60, from Cormack et al. 2009) controls how sharply top ranks are
favoured. Smaller k = more winner-take-all; larger k = flatter. 60 is a good
default and rarely worth tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from core.config import settings
from vectorstore.base import SearchHit


@dataclass
class FusedHit:
    hit: SearchHit
    rrf_score: float = 0.0
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    retrievers: set[str] = field(default_factory=set)

    @property
    def id(self) -> str:
        return self.hit.id

    @property
    def retriever_label(self) -> str:
        if len(self.retrievers) > 1:
            return "hybrid"
        return next(iter(self.retrievers), "dense")


def reciprocal_rank_fusion(
    dense: Sequence[SearchHit],
    sparse: Sequence[SearchHit],
    k: Optional[int] = None,
    dense_weight: Optional[float] = None,
    sparse_weight: Optional[float] = None,
) -> list[FusedHit]:
    """Fuse the dense and sparse branches into one ranked list."""
    k = settings.rrf_k if k is None else k
    w_dense = settings.dense_weight if dense_weight is None else dense_weight
    w_sparse = settings.sparse_weight if sparse_weight is None else sparse_weight

    fused: dict[str, FusedHit] = {}

    for rank, hit in enumerate(dense, start=1):
        entry = fused.setdefault(hit.id, FusedHit(hit=hit))
        entry.rrf_score += w_dense / (k + rank)
        entry.dense_score = hit.score
        entry.dense_rank = rank
        entry.retrievers.add("dense")

    for rank, hit in enumerate(sparse, start=1):
        entry = fused.get(hit.id)
        if entry is None:
            entry = FusedHit(hit=hit)
            fused[hit.id] = entry
        else:
            # Prefer whichever branch carried a parent window / richer text.
            if not entry.hit.parent_text and hit.parent_text:
                entry.hit.parent_text = hit.parent_text
        entry.rrf_score += w_sparse / (k + rank)
        entry.sparse_score = hit.score
        entry.sparse_rank = rank
        entry.retrievers.add("sparse")

    ordered = sorted(fused.values(), key=lambda f: f.rrf_score, reverse=True)
    return ordered


def fuse_multi(
    branches: Iterable[tuple[Sequence[SearchHit], float]],
    k: Optional[int] = None,
) -> list[FusedHit]:
    """RRF over N weighted branches — used for multi-query expansion."""
    k = settings.rrf_k if k is None else k
    fused: dict[str, FusedHit] = {}

    for hits, weight in branches:
        for rank, hit in enumerate(hits, start=1):
            entry = fused.setdefault(hit.id, FusedHit(hit=hit))
            entry.rrf_score += weight / (k + rank)
            entry.retrievers.add(hit.retriever)
            if hit.retriever == "dense":
                # Keep the best (highest) similarity seen for this chunk.
                if entry.dense_score is None or hit.score > entry.dense_score:
                    entry.dense_score = hit.score
                    entry.dense_rank = rank
            else:
                if entry.sparse_score is None or hit.score > entry.sparse_score:
                    entry.sparse_score = hit.score
                    entry.sparse_rank = rank

    return sorted(fused.values(), key=lambda f: f.rrf_score, reverse=True)


def deduplicate_by_content(
    hits: Sequence[FusedHit],
    similarity_chars: int = 160,
) -> list[FusedHit]:
    """Drop near-duplicate chunks produced by overlap between adjacent windows.

    Two chunks from the same document whose leading `similarity_chars` match are
    almost certainly the overlap region of neighbours. Feeding both to the LLM
    wastes context and biases the answer toward whatever was duplicated.
    """
    seen: set[tuple[str, str]] = set()
    out: list[FusedHit] = []
    for fused in hits:
        key = (
            fused.hit.metadata.doc_id,
            " ".join(fused.hit.text.lower().split())[:similarity_chars],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(fused)
    return out
