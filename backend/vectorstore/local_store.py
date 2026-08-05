"""
Zero-dependency fallback store (NumPy flat index + the same BM25Index).

This exists for one reason: **Milvus must not be a prerequisite for running the
project.** Milvus Lite is Linux/macOS-only and Milvus standalone needs Docker, so
on a bare Windows machine there is otherwise no way to even smoke-test the
pipeline. Set `VECTOR_BACKEND=local` and everything above this layer — hybrid
search, RRF, reranking, filtering, citations — behaves identically.

It is honestly not a production store: brute-force cosine over a dense matrix.
That said, a flat scan is *exact* (recall@k = 1.0 by construction) and NumPy's
BLAS-backed matmul does ~50k × 384 dims in about 2 ms, so for the assignment's
document volume it is genuinely fast. Past ~200k chunks, switch to Milvus.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from core.config import settings
from core.latency import METRICS
from core.schemas import Chunk, ChunkMetadata
from vectorstore.base import SearchHit, VectorStore
from vectorstore.bm25 import BM25Index
from vectorstore.milvus_store import build_meta_predicate

logger = logging.getLogger(__name__)


class LocalStore(VectorStore):
    name = "local"

    def __init__(self, path: Optional[Path] = None, collection: Optional[str] = None) -> None:
        self.collection = collection or settings.collection_name
        self.path = Path(path or settings.local_index_file)
        self._lock = threading.RLock()

        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._matrix: Optional[np.ndarray] = None      # (n, dim) float32, L2-normalized
        self._texts: list[str] = []
        self._parents: list[str] = []
        self._metas: list[dict[str, Any]] = []
        self._dim = settings.embedding_dim
        self._dirty = False

        self._bm25 = BM25Index(path=self.path.with_name(f"{self.path.stem}_bm25.pkl"))

    # ---------------------------------------------------------------- connect
    def connect(self) -> None:
        self._load()

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        with self._lock:
            self._dim = dim
            if recreate:
                self._reset()
                self._persist()
                return
            self._load()
            if self._matrix is not None and self._matrix.shape[1] != dim:
                logger.warning(
                    "Local index dim=%d but embedder dim=%d — resetting index.",
                    self._matrix.shape[1], dim,
                )
                self._reset()

    def _reset(self) -> None:
        self._ids, self._index = [], {}
        self._matrix = None
        self._texts, self._parents, self._metas = [], [], []
        self._bm25.clear()
        self._dirty = True

    # ----------------------------------------------------------------- upsert
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks/vectors mismatch: {len(chunks)} vs {len(vectors)}")

        start = time.perf_counter()
        with self._lock:
            vectors = np.asarray(vectors, dtype=np.float32)
            # Re-normalize defensively; the search path assumes unit vectors.
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)

            new_rows: list[np.ndarray] = []
            for chunk, vector in zip(chunks, vectors):
                meta_dict = json.loads(chunk.metadata.model_dump_json())
                existing = self._index.get(chunk.id)
                if existing is not None:
                    # In-place update keeps ids stable across re-uploads.
                    self._texts[existing] = chunk.text
                    self._parents[existing] = chunk.parent_text or ""
                    self._metas[existing] = meta_dict
                    if self._matrix is not None:
                        self._matrix[existing] = vector
                else:
                    self._index[chunk.id] = len(self._ids)
                    self._ids.append(chunk.id)
                    self._texts.append(chunk.text)
                    self._parents.append(chunk.parent_text or "")
                    self._metas.append(meta_dict)
                    new_rows.append(vector)

                self._bm25.add(chunk.id, chunk.text, _bm25_meta(meta_dict))

            if new_rows:
                block = np.vstack(new_rows)
                self._matrix = block if self._matrix is None else np.vstack([self._matrix, block])

            self._dim = self._matrix.shape[1] if self._matrix is not None else self._dim
            self._dirty = True
            self._persist()

        METRICS.observe("store.upsert", (time.perf_counter() - start) * 1000)
        logger.info("Local store now holds %d chunks", len(self._ids))
        return len(chunks)

    # ----------------------------------------------------------------- search
    def search_dense(
        self,
        vector: np.ndarray,
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchHit]:
        with self._lock:
            if self._matrix is None or not len(self._ids):
                return []

            start = time.perf_counter()
            query = np.asarray(vector, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(query)
            if norm > 1e-12:
                query = query / norm

            candidate_rows = self._filter_rows(filters)
            if candidate_rows is not None and not len(candidate_rows):
                return []

            if candidate_rows is None:
                scores = self._matrix @ query
                order = np.argsort(-scores)[:top_k]
                rows = order
            else:
                subset = self._matrix[candidate_rows]
                scores_subset = subset @ query
                order = np.argsort(-scores_subset)[:top_k]
                rows = candidate_rows[order]
                scores = np.zeros(len(self._ids), dtype=np.float32)
                scores[candidate_rows] = scores_subset

            METRICS.observe("store.search_dense", (time.perf_counter() - start) * 1000)
            return [
                SearchHit(
                    id=self._ids[i],
                    text=self._texts[i],
                    metadata=_meta_model(self._metas[i]),
                    # Unit vectors => dot product is cosine; shift to [0, 1].
                    score=float(max(0.0, min(1.0, (scores[i] + 1.0) / 2.0)))
                    if scores[i] < 0
                    else float(min(1.0, scores[i])),
                    retriever="dense",
                    parent_text=self._parents[i] or None,
                )
                for i in rows
            ]

    def search_sparse(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchHit]:
        with self._lock:
            if not self._ids:
                return []
            allowed: Optional[set[str]] = None
            if filters:
                allowed = self._bm25.keys_matching(build_meta_predicate(filters))
                if not allowed:
                    return []

            start = time.perf_counter()
            ranked = self._bm25.search(query, top_k=top_k, allowed=allowed)
            METRICS.observe("store.search_sparse", (time.perf_counter() - start) * 1000)

            hits: list[SearchHit] = []
            for doc_id, score in ranked:
                idx = self._index.get(doc_id)
                if idx is None:
                    continue
                hits.append(
                    SearchHit(
                        id=doc_id,
                        text=self._texts[idx],
                        metadata=_meta_model(self._metas[idx]),
                        score=float(score),
                        retriever="sparse",
                        parent_text=self._parents[idx] or None,
                    )
                )
            return hits

    def _filter_rows(self, filters: Optional[dict[str, Any]]) -> Optional[np.ndarray]:
        if not filters:
            return None
        predicate = build_meta_predicate(filters)
        rows = [
            i
            for i, meta in enumerate(self._metas)
            if predicate(_bm25_meta(meta))
        ]
        return np.asarray(rows, dtype=np.int64)

    def find_literal(
        self,
        variants: Sequence[str],
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchHit]:
        """Exact substring scan over chunk texts. See VectorStore.find_literal."""
        if not variants:
            return []
        with self._lock:
            if not self._ids:
                return []
            needles = [v.lower() for v in variants if v]
            predicate = build_meta_predicate(filters) if filters else None

            hits: list[SearchHit] = []
            for index, text in enumerate(self._texts):
                lowered = text.lower()
                if not any(needle in lowered for needle in needles):
                    continue
                meta = self._metas[index]
                if predicate and not predicate(_bm25_meta(meta)):
                    continue
                hits.append(
                    SearchHit(
                        id=self._ids[index],
                        text=text,
                        metadata=_meta_model(meta),
                        # Exact match: a literal either is or is not present, so
                        # there is no meaningful gradation to report.
                        score=1.0,
                        retriever="literal",
                        parent_text=self._parents[index] or None,
                    )
                )
                if len(hits) >= limit:
                    break
            return hits

    def fetch(self, ids: Sequence[str]) -> list[SearchHit]:
        with self._lock:
            out: list[SearchHit] = []
            for doc_id in ids:
                idx = self._index.get(doc_id)
                if idx is None:
                    continue
                out.append(
                    SearchHit(
                        id=doc_id,
                        text=self._texts[idx],
                        metadata=_meta_model(self._metas[idx]),
                        score=0.0,
                        parent_text=self._parents[idx] or None,
                    )
                )
            return out

    # ------------------------------------------------------------ management
    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            keep = [i for i, m in enumerate(self._metas) if m.get("doc_id") != doc_id]
            removed = len(self._ids) - len(keep)
            if not removed:
                return 0
            self._ids = [self._ids[i] for i in keep]
            self._texts = [self._texts[i] for i in keep]
            self._parents = [self._parents[i] for i in keep]
            self._metas = [self._metas[i] for i in keep]
            self._matrix = self._matrix[keep] if self._matrix is not None else None
            self._index = {cid: i for i, cid in enumerate(self._ids)}
            self._bm25.remove_by(lambda m: m.get("doc_id") == doc_id)
            self._dirty = True
            self._persist()
            return removed

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            docs: dict[str, dict[str, Any]] = {}
            for meta in self._metas:
                doc_id = meta.get("doc_id", "")
                if not doc_id:
                    continue
                entry = docs.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "source": meta.get("source", ""),
                        "category": meta.get("category", "other"),
                        "districts": set(),
                        "topics": set(),
                        "chunks": 0,
                        "ingested_at": meta.get("ingested_at"),
                    },
                )
                entry["chunks"] += 1
                entry["districts"].update(meta.get("districts") or [])
                entry["topics"].update(meta.get("topics") or [])
            return [
                {**d, "districts": sorted(d["districts"]), "topics": sorted(d["topics"])}
                for d in docs.values()
            ]

    def count(self) -> int:
        return len(self._ids)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "collection": self.collection,
            "entities": len(self._ids),
            "dim": self._dim,
            "index": "FLAT (exact)",
            "metric": "COSINE",
            "native_bm25": False,
            "path": str(self.path),
            "bm25": self._bm25.stats(),
        }

    # ------------------------------------------------------------ persistence
    def _persist(self) -> None:
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp,
                ids=np.asarray(self._ids, dtype=object),
                matrix=self._matrix if self._matrix is not None else np.zeros((0, self._dim), np.float32),
                texts=np.asarray(self._texts, dtype=object),
                parents=np.asarray(self._parents, dtype=object),
                metas=np.asarray([json.dumps(m) for m in self._metas], dtype=object),
            )
            tmp.replace(self.path)
            self._bm25.save()
            self._dirty = False
        except Exception as exc:  # noqa: BLE001 — never fail a request over a cache write
            logger.warning("Failed to persist local index: %s", exc)

    def _load(self) -> None:
        if self._ids or not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=True) as data:
                self._ids = [str(x) for x in data["ids"].tolist()]
                matrix = data["matrix"]
                self._matrix = matrix.astype(np.float32) if matrix.size else None
                self._texts = [str(x) for x in data["texts"].tolist()]
                self._parents = [str(x) for x in data["parents"].tolist()]
                self._metas = [json.loads(x) for x in data["metas"].tolist()]
            self._index = {cid: i for i, cid in enumerate(self._ids)}
            if self._matrix is not None:
                self._dim = int(self._matrix.shape[1])
            if not self._bm25.load():
                for cid, text, meta in zip(self._ids, self._texts, self._metas):
                    self._bm25.add(cid, text, _bm25_meta(meta))
                self._bm25.save()
            logger.info("Loaded local index: %d chunks from %s", len(self._ids), self.path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load local index (%s); starting empty", exc)
            self._reset()

    def close(self) -> None:
        with self._lock:
            self._persist()


def _bm25_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": meta.get("doc_id", ""),
        "district": meta.get("district") or "",
        "districts": meta.get("districts") or [],
        "category": meta.get("category", "other"),
        "topic": meta.get("topic") or "",
        "topics": meta.get("topics") or [],
        "source": meta.get("source", ""),
        "language": meta.get("language", "en"),
    }


def _meta_model(meta: dict[str, Any]) -> ChunkMetadata:
    try:
        return ChunkMetadata(**meta)
    except Exception:  # noqa: BLE001
        return ChunkMetadata(
            doc_id=meta.get("doc_id", "unknown"),
            source=meta.get("source", "unknown"),
        )
