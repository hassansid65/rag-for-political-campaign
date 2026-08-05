"""Vector store interface — every backend implements exactly this surface."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from core.schemas import Chunk, ChunkMetadata


@dataclass
class SearchHit:
    id: str
    text: str
    metadata: ChunkMetadata
    score: float
    retriever: str = "dense"          # dense | sparse
    parent_text: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


class VectorStore(abc.ABC):
    """Abstract store.

    The interface is deliberately narrow: connect, upsert, two search branches,
    delete, stats. Keeping fusion and reranking *out* of the store means the
    retrieval pipeline behaves identically on Milvus, Milvus Lite, and the local
    fallback — the backend is a deployment choice, not a behaviour change.
    """

    name: str = "base"

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def ensure_collection(self, dim: int, recreate: bool = False) -> None: ...

    @abc.abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int: ...

    @abc.abstractmethod
    def search_dense(
        self,
        vector: np.ndarray,
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchHit]: ...

    @abc.abstractmethod
    def search_sparse(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchHit]: ...

    def find_literal(
        self,
        variants: Sequence[str],
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchHit]:
        """Chunks whose text contains any of `variants` verbatim.

        This is an *exact* lookup, deliberately outside the ANN path. Reverse
        queries ("who was born on 14 October 1985") need a value match, and both
        dense and BM25 retrieval blur values: the embedding barely encodes a date,
        and BM25 scores a record matching two of `{14, october, 1985}` almost as
        highly as the one matching all three. Scanning for the literal is exact
        and, at this corpus size, far cheaper than either.

        Default implementation returns nothing; backends that can scan override it.
        """
        return []

    @abc.abstractmethod
    def delete_document(self, doc_id: str) -> int: ...

    @abc.abstractmethod
    def list_documents(self) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    def count(self) -> int: ...

    @abc.abstractmethod
    def stats(self) -> dict[str, Any]: ...

    def health(self) -> dict[str, Any]:
        try:
            return {"status": "ok", "backend": self.name, **self.stats()}
        except Exception as exc:  # noqa: BLE001
            return {"status": "down", "backend": self.name, "error": str(exc)}

    def close(self) -> None:  # optional override
        return None
