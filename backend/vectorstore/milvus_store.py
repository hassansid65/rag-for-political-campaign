"""
Milvus vector store — dense HNSW + BM25 sparse, with metadata filtering.

Deployment shapes, all one code path:
  * `VECTOR_BACKEND=milvus`      + MILVUS_URI=http://host:19530   (standalone/cluster)
  * `VECTOR_BACKEND=milvus`      + MILVUS_URI=https://…zilliz…    (Zilliz Cloud, MILVUS_TOKEN)
  * `VECTOR_BACKEND=milvus_lite` + MILVUS_LITE_FILE=…db           (embedded, no server)

Two things worth calling out:

**Sparse retrieval degrades, it doesn't disappear.** Milvus 2.5+ can compute BM25
server-side from a `text` field via a built-in Function, which is the right
production answer. Milvus Lite and older servers can't. Rather than silently
dropping to dense-only (which is where hybrid systems quietly lose their
proper-noun recall), we detect the capability at collection-create time and fall
back to the in-process `BM25Index`. Hybrid search behaves the same either way.

**Filtering happens before ANN, not after.** Milvus applies the boolean
expression during graph traversal, so `district == "NTR"` narrows the search
space rather than discarding results after the fact. Post-filtering a top-30 is
how you end up returning 3 chunks for a district that has 200.
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

logger = logging.getLogger(__name__)

_MAX_TEXT = 8192
_MAX_PARENT = 8192
_TEXT_SPARSE_FIELD = "sparse"
_DENSE_FIELD = "vector"


class MilvusStore(VectorStore):
    name = "milvus"

    def __init__(
        self,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        collection: Optional[str] = None,
        lite: bool = False,
    ) -> None:
        self.lite = lite
        self.uri = str(settings.milvus_lite_file) if lite else (uri or settings.milvus_uri)
        self.token = token if token is not None else settings.milvus_token
        self.collection = collection or settings.collection_name
        self.name = "milvus_lite" if lite else "milvus"

        self._client = None
        self._dim = settings.embedding_dim
        self._native_bm25 = False
        self._lock = threading.RLock()

        # Client-side keyword branch — used when the server has no BM25 Function,
        # and always kept warm so we can switch without a re-index.
        bm25_path = Path(settings.milvus_lite_file).parent / f"{self.collection}_bm25.pkl"
        self._bm25 = BM25Index(path=bm25_path)
        self._bm25_loaded = False

    # ---------------------------------------------------------------- connect
    def connect(self) -> None:
        if self._client is not None:
            return
        with self._lock:
            if self._client is not None:
                return
            from pymilvus import MilvusClient

            start = time.perf_counter()
            kwargs: dict[str, Any] = {"uri": self.uri}
            if not self.lite:
                if self.token:
                    kwargs["token"] = self.token
                if settings.milvus_db_name and settings.milvus_db_name != "default":
                    kwargs["db_name"] = settings.milvus_db_name

            self._client = MilvusClient(**kwargs)
            logger.info(
                "Connected to %s at %s in %.0fms",
                self.name, self.uri, (time.perf_counter() - start) * 1000,
            )

    def _c(self):
        if self._client is None:
            self.connect()
        return self._client

    # ------------------------------------------------------------- collection
    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        from pymilvus import DataType

        self._dim = dim
        client = self._c()

        if recreate and client.has_collection(self.collection):
            logger.warning("Dropping collection %s (recreate=True)", self.collection)
            client.drop_collection(self.collection)
            self._bm25.clear()
            self._bm25.save()

        if client.has_collection(self.collection):
            self._native_bm25 = self._detect_native_bm25()
            self._ensure_loaded()
            self._load_bm25_if_needed()
            logger.info(
                "Collection %s ready (entities=%s, native_bm25=%s)",
                self.collection, self._safe_count(), self._native_bm25,
            )
            return

        # ---- schema -------------------------------------------------------
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        # enable_analyzer on `text` is what lets Milvus build BM25 server-side.
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(_DENSE_FIELD, DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("text", DataType.VARCHAR, max_length=_MAX_TEXT, enable_analyzer=True)
        schema.add_field("parent_text", DataType.VARCHAR, max_length=_MAX_PARENT)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        schema.add_field("district", DataType.VARCHAR, max_length=128)
        schema.add_field("state", DataType.VARCHAR, max_length=64)
        schema.add_field("topic", DataType.VARCHAR, max_length=64)
        schema.add_field("section", DataType.VARCHAR, max_length=512)
        schema.add_field("language", DataType.VARCHAR, max_length=16)
        schema.add_field("party", DataType.VARCHAR, max_length=64)
        schema.add_field("candidate", DataType.VARCHAR, max_length=256)
        schema.add_field("page", DataType.INT64)
        schema.add_field("chunk_index", DataType.INT64)
        # ARRAY fields let us filter "chunk mentions Vijayawada" without a JSON scan.
        schema.add_field(
            "districts", DataType.ARRAY, element_type=DataType.VARCHAR,
            max_capacity=16, max_length=128,
        )
        schema.add_field(
            "topics", DataType.ARRAY, element_type=DataType.VARCHAR,
            max_capacity=8, max_length=64,
        )
        schema.add_field("meta", DataType.JSON)

        native_bm25 = self._try_add_bm25_function(schema)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=_DENSE_FIELD,
            index_type=settings.milvus_index_type,
            metric_type=settings.milvus_metric_type,
            params={"M": settings.hnsw_m, "efConstruction": settings.hnsw_ef_construction},
        )
        if native_bm25:
            index_params.add_index(
                field_name=_TEXT_SPARSE_FIELD,
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
            )
        # Scalar indexes turn metadata filters from a scan into a lookup.
        for field in ("doc_id", "district", "category", "topic", "source", "language"):
            try:
                index_params.add_index(field_name=field, index_type="INVERTED")
            except Exception:  # pragma: no cover - older pymilvus
                pass

        try:
            client.create_collection(
                collection_name=self.collection,
                schema=schema,
                index_params=index_params,
                consistency_level="Bounded",  # Strong doubles search latency for no benefit here
            )
            self._native_bm25 = native_bm25
        except Exception as exc:  # noqa: BLE001
            if not native_bm25:
                raise
            logger.warning(
                "Collection create with server-side BM25 failed (%s). "
                "Retrying dense-only; keyword search will use the in-process BM25 index.",
                exc,
            )
            self._create_dense_only(dim)
            self._native_bm25 = False

        self._ensure_loaded()
        logger.info(
            "Created collection %s (dim=%d, index=%s, native_bm25=%s)",
            self.collection, dim, settings.milvus_index_type, self._native_bm25,
        )

    def _try_add_bm25_function(self, schema) -> bool:
        """Add the sparse field + BM25 Function if this pymilvus/Milvus supports it."""
        if self.lite:
            # Milvus Lite has no full-text-search Function support; don't pretend.
            return False
        try:
            from pymilvus import DataType, Function, FunctionType

            schema.add_field(_TEXT_SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(
                Function(
                    name="text_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=[_TEXT_SPARSE_FIELD],
                )
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.info("Server-side BM25 unavailable (%s); using in-process BM25", exc)
            return False

    def _create_dense_only(self, dim: int) -> None:
        from pymilvus import DataType

        client = self._c()
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field(_DENSE_FIELD, DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("text", DataType.VARCHAR, max_length=_MAX_TEXT)
        schema.add_field("parent_text", DataType.VARCHAR, max_length=_MAX_PARENT)
        for field, length in (
            ("doc_id", 128), ("source", 512), ("category", 64), ("district", 128),
            ("state", 64), ("topic", 64), ("section", 512), ("language", 16),
            ("party", 64), ("candidate", 256),
        ):
            schema.add_field(field, DataType.VARCHAR, max_length=length)
        schema.add_field("page", DataType.INT64)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field(
            "districts", DataType.ARRAY, element_type=DataType.VARCHAR,
            max_capacity=16, max_length=128,
        )
        schema.add_field(
            "topics", DataType.ARRAY, element_type=DataType.VARCHAR,
            max_capacity=8, max_length=64,
        )
        schema.add_field("meta", DataType.JSON)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=_DENSE_FIELD,
            index_type=settings.milvus_index_type,
            metric_type=settings.milvus_metric_type,
            params={"M": settings.hnsw_m, "efConstruction": settings.hnsw_ef_construction},
        )
        client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
            consistency_level="Bounded",
        )

    def _detect_native_bm25(self) -> bool:
        try:
            desc = self._c().describe_collection(self.collection)
            fields = {f.get("name") for f in desc.get("fields", [])}
            return _TEXT_SPARSE_FIELD in fields
        except Exception:  # noqa: BLE001
            return False

    def _ensure_loaded(self) -> None:
        """Load the collection into memory — Milvus refuses searches otherwise."""
        try:
            state = self._c().get_load_state(self.collection)
            value = str(state.get("state", state)) if isinstance(state, dict) else str(state)
            if "Loaded" not in value:
                self._c().load_collection(self.collection)
        except Exception as exc:  # noqa: BLE001
            logger.debug("load_collection: %s", exc)

    # -------------------------------------------------------------- BM25 sync
    def _load_bm25_if_needed(self) -> None:
        if self._native_bm25 or self._bm25_loaded:
            return
        self._bm25_loaded = True
        if self._bm25.load():
            return
        # No persisted index but the collection has data -> rebuild from Milvus.
        try:
            if self._safe_count() > 0:
                logger.info("Rebuilding in-process BM25 index from collection…")
                self._rebuild_bm25_from_collection()
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 rebuild failed: %s", exc)

    def _rebuild_bm25_from_collection(self, batch: int = 1000) -> int:
        client = self._c()
        self._bm25.clear()
        total = 0
        offset = 0
        while True:
            rows = client.query(
                collection_name=self.collection,
                filter="",
                output_fields=["id", "text", "doc_id", "district", "category", "topic",
                               "source", "language", "districts", "topics"],
                limit=batch,
                offset=offset,
            )
            if not rows:
                break
            for row in rows:
                self._bm25.add(row["id"], row.get("text", ""), self._bm25_meta(row))
            total += len(rows)
            offset += len(rows)
            if len(rows) < batch:
                break
        self._bm25.save()
        logger.info("BM25 index rebuilt from %d chunks", total)
        return total

    @staticmethod
    def _bm25_meta(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "doc_id": row.get("doc_id", ""),
            "district": row.get("district", ""),
            "districts": row.get("districts") or [],
            "category": row.get("category", ""),
            "topic": row.get("topic", ""),
            "topics": row.get("topics") or [],
            "source": row.get("source", ""),
            "language": row.get("language", "en"),
        }

    # ----------------------------------------------------------------- upsert
    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks/vectors mismatch: {len(chunks)} vs {len(vectors)}")

        client = self._c()
        rows: list[dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors):
            meta = chunk.metadata
            rows.append(
                {
                    "id": chunk.id,
                    _DENSE_FIELD: vector.tolist(),
                    "text": chunk.text[:_MAX_TEXT],
                    "parent_text": (chunk.parent_text or "")[:_MAX_PARENT],
                    "doc_id": meta.doc_id,
                    "source": meta.source[:512],
                    "category": meta.category,
                    "district": meta.district or "",
                    "state": meta.state or "",
                    "topic": meta.topic or "",
                    "section": (meta.section or "")[:512],
                    "language": meta.language or "en",
                    "party": meta.party or "",
                    "candidate": (meta.candidate or "")[:256],
                    "page": int(meta.page or 0),
                    "chunk_index": int(meta.chunk_index),
                    "districts": (meta.districts or [])[:16],
                    "topics": (meta.topics or [])[:8],
                    "meta": json.loads(meta.model_dump_json()),
                }
            )

        start = time.perf_counter()
        inserted = 0
        # Batch to keep single gRPC messages well under the 64 MB limit.
        for i in range(0, len(rows), 256):
            batch = rows[i : i + 256]
            client.upsert(collection_name=self.collection, data=batch)
            inserted += len(batch)

        # Seal the growing segments. Without this, freshly upserted rows are not
        # visible to search under `Bounded` consistency and `get_collection_stats`
        # reports row_count=0 — which looks exactly like "the upsert silently
        # failed". Ingest is not the latency-critical path, so paying for a flush
        # here to get read-your-writes is the right trade; the alternative is
        # Strong consistency on every search, which doubles query latency forever
        # to fix a problem that only exists for a few seconds after an upload.
        try:
            client.flush(collection_name=self.collection)
        except Exception as exc:  # noqa: BLE001 — data is inserted either way
            logger.warning("Flush after upsert failed (%s); rows may lag", exc)

        if not self._native_bm25:
            self._load_bm25_if_needed()
            for chunk in chunks:
                self._bm25.add(chunk.id, chunk.text, self._bm25_meta_from_chunk(chunk))
            self._bm25.save()

        METRICS.observe("store.upsert", (time.perf_counter() - start) * 1000)
        logger.info("Upserted %d chunks into %s", inserted, self.collection)
        return inserted

    @staticmethod
    def _bm25_meta_from_chunk(chunk: Chunk) -> dict[str, Any]:
        meta = chunk.metadata
        return {
            "doc_id": meta.doc_id,
            "district": meta.district or "",
            "districts": meta.districts or [],
            "category": meta.category,
            "topic": meta.topic or "",
            "topics": meta.topics or [],
            "source": meta.source,
            "language": meta.language or "en",
        }

    # ----------------------------------------------------------------- search
    def search_dense(
        self,
        vector: np.ndarray,
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchHit]:
        client = self._c()
        expr = build_filter_expr(filters)
        start = time.perf_counter()
        try:
            results = client.search(
                collection_name=self.collection,
                data=[vector.tolist()],
                anns_field=_DENSE_FIELD,
                limit=top_k,
                filter=expr or "",
                output_fields=["id", "text", "parent_text", "meta"],
                search_params={
                    "metric_type": settings.milvus_metric_type,
                    "params": {"ef": max(settings.hnsw_ef_search, top_k)},
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Dense search failed (expr=%r): %s", expr, exc)
            return []
        METRICS.observe("store.search_dense", (time.perf_counter() - start) * 1000)

        hits: list[SearchHit] = []
        for row in (results[0] if results else []):
            entity = row.get("entity", row)
            hits.append(
                SearchHit(
                    id=str(entity.get("id") or row.get("id")),
                    text=entity.get("text", ""),
                    metadata=_meta_from_row(entity),
                    score=_normalize_dense(float(row.get("distance", 0.0))),
                    retriever="dense",
                    parent_text=entity.get("parent_text") or None,
                )
            )
        return hits

    def search_sparse(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchHit]:
        if self._native_bm25:
            return self._search_sparse_native(query, top_k, filters)
        return self._search_sparse_local(query, top_k, filters)

    def _search_sparse_native(
        self, query: str, top_k: int, filters: Optional[dict[str, Any]]
    ) -> list[SearchHit]:
        client = self._c()
        expr = build_filter_expr(filters)
        start = time.perf_counter()
        try:
            results = client.search(
                collection_name=self.collection,
                data=[query],                      # raw text: Milvus applies BM25
                anns_field=_TEXT_SPARSE_FIELD,
                limit=top_k,
                filter=expr or "",
                output_fields=["id", "text", "parent_text", "meta"],
                search_params={"metric_type": "BM25", "params": {"drop_ratio_search": 0.2}},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Native BM25 search failed (%s); falling back to local BM25", exc)
            self._native_bm25 = False
            self._load_bm25_if_needed()
            if self._bm25.size == 0:
                try:
                    self._rebuild_bm25_from_collection()
                except Exception:  # noqa: BLE001
                    return []
            return self._search_sparse_local(query, top_k, filters)
        METRICS.observe("store.search_sparse", (time.perf_counter() - start) * 1000)

        hits: list[SearchHit] = []
        for row in (results[0] if results else []):
            entity = row.get("entity", row)
            hits.append(
                SearchHit(
                    id=str(entity.get("id") or row.get("id")),
                    text=entity.get("text", ""),
                    metadata=_meta_from_row(entity),
                    score=float(row.get("distance", 0.0)),
                    retriever="sparse",
                    parent_text=entity.get("parent_text") or None,
                )
            )
        return hits

    def _search_sparse_local(
        self, query: str, top_k: int, filters: Optional[dict[str, Any]]
    ) -> list[SearchHit]:
        self._load_bm25_if_needed()
        if self._bm25.size == 0:
            return []

        allowed: Optional[set[str]] = None
        if filters:
            predicate = build_meta_predicate(filters)
            allowed = self._bm25.keys_matching(predicate)
            if not allowed:
                return []

        start = time.perf_counter()
        ranked = self._bm25.search(query, top_k=top_k, allowed=allowed)
        if not ranked:
            METRICS.observe("store.search_sparse", (time.perf_counter() - start) * 1000)
            return []

        ids = [doc_id for doc_id, _ in ranked]
        rows = self._fetch_by_ids(ids)
        by_id = {row["id"]: row for row in rows}
        METRICS.observe("store.search_sparse", (time.perf_counter() - start) * 1000)

        hits: list[SearchHit] = []
        for doc_id, score in ranked:
            row = by_id.get(doc_id)
            if not row:
                continue
            hits.append(
                SearchHit(
                    id=doc_id,
                    text=row.get("text", ""),
                    metadata=_meta_from_row(row),
                    score=float(score),
                    retriever="sparse",
                    parent_text=row.get("parent_text") or None,
                )
            )
        return hits

    def _fetch_by_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        try:
            return self._c().get(
                collection_name=self.collection,
                ids=list(ids),
                output_fields=["id", "text", "parent_text", "meta"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus get() failed: %s", exc)
            return []

    def find_literal(
        self,
        variants: Sequence[str],
        limit: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchHit]:
        """Exact literal lookup via Milvus scalar filtering.

        Uses `LIKE '%value%'` on the `text` field, which Milvus evaluates without
        touching the vector index. Falls back to the in-process BM25 index's
        stored texts if the server rejects the expression (older versions have
        patchier LIKE support), so the capability never silently disappears.
        """
        if not variants:
            return []

        clauses = [f'text LIKE "%{_esc(v)}%"' for v in variants if v]
        if not clauses:
            return []
        expr = "(" + " or ".join(clauses) + ")"
        base = build_filter_expr(filters)
        if base:
            expr = f"{base} and {expr}"

        try:
            rows = self._c().query(
                collection_name=self.collection,
                filter=expr,
                output_fields=["id", "text", "parent_text", "meta"],
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Literal query failed (%s); scanning the local BM25 texts", exc)
            return self._find_literal_local(variants, limit, filters)

        return [
            SearchHit(
                id=row["id"],
                text=row.get("text", ""),
                metadata=_meta_from_row(row),
                score=1.0,
                retriever="literal",
                parent_text=row.get("parent_text") or None,
            )
            for row in rows
        ]

    def _find_literal_local(
        self,
        variants: Sequence[str],
        limit: int,
        filters: Optional[dict[str, Any]],
    ) -> list[SearchHit]:
        self._load_bm25_if_needed()
        if self._bm25.size == 0:
            return []
        predicate = build_meta_predicate(filters) if filters else None
        needles = [v.lower() for v in variants if v]

        candidate_ids = [
            key
            for key in self._bm25.keys_matching(lambda m: not predicate or predicate(m))
        ]
        rows = self._fetch_by_ids(candidate_ids[:2000])
        hits: list[SearchHit] = []
        for row in rows:
            lowered = (row.get("text") or "").lower()
            if any(needle in lowered for needle in needles):
                hits.append(
                    SearchHit(
                        id=row["id"],
                        text=row.get("text", ""),
                        metadata=_meta_from_row(row),
                        score=1.0,
                        retriever="literal",
                        parent_text=row.get("parent_text") or None,
                    )
                )
            if len(hits) >= limit:
                break
        return hits

    def fetch(self, ids: Sequence[str]) -> list[SearchHit]:
        rows = self._fetch_by_ids(ids)
        return [
            SearchHit(
                id=row["id"],
                text=row.get("text", ""),
                metadata=_meta_from_row(row),
                score=0.0,
                parent_text=row.get("parent_text") or None,
            )
            for row in rows
        ]

    # ------------------------------------------------------------ management
    def delete_document(self, doc_id: str) -> int:
        client = self._c()
        try:
            before = self._safe_count()
            client.delete(collection_name=self.collection, filter=f'doc_id == "{_esc(doc_id)}"')
            # Same visibility rule as upsert: without a flush the deleted rows keep
            # answering searches, so a re-upload appears to duplicate the document.
            try:
                client.flush(collection_name=self.collection)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Flush after delete failed: %s", exc)
            if not self._native_bm25:
                self._load_bm25_if_needed()
                self._bm25.remove_by(lambda m: m.get("doc_id") == doc_id)
                self._bm25.save()
            after = self._safe_count()
            return max(0, before - after)
        except Exception as exc:  # noqa: BLE001
            logger.error("delete_document(%s) failed: %s", doc_id, exc)
            return 0

    def list_documents(self) -> list[dict[str, Any]]:
        client = self._c()
        docs: dict[str, dict[str, Any]] = {}
        offset = 0
        batch = 1000
        while True:
            try:
                rows = client.query(
                    collection_name=self.collection,
                    filter="",
                    output_fields=["doc_id", "source", "category", "districts", "topics", "meta"],
                    limit=batch,
                    offset=offset,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("list_documents query failed: %s", exc)
                break
            if not rows:
                break
            for row in rows:
                doc_id = row.get("doc_id", "")
                if not doc_id:
                    continue
                entry = docs.setdefault(
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "source": row.get("source", ""),
                        "category": row.get("category", "other"),
                        "districts": set(),
                        "topics": set(),
                        "chunks": 0,
                        "ingested_at": (row.get("meta") or {}).get("ingested_at"),
                    },
                )
                entry["chunks"] += 1
                entry["districts"].update(row.get("districts") or [])
                entry["topics"].update(row.get("topics") or [])
            offset += len(rows)
            if len(rows) < batch:
                break

        return [
            {**d, "districts": sorted(d["districts"]), "topics": sorted(d["topics"])}
            for d in docs.values()
        ]

    def count(self) -> int:
        return self._safe_count()

    def _safe_count(self) -> int:
        """Entity count.

        `count(*)` is queried rather than read from `get_collection_stats`, whose
        `row_count` only reflects *sealed* segments — so it reports 0 for data that
        was just inserted and is perfectly searchable. A count that disagrees with
        search results is worse than a slightly slower count.
        """
        client = self._c()
        try:
            rows = client.query(
                collection_name=self.collection,
                filter="",
                output_fields=["count(*)"],
            )
            if rows:
                first = rows[0]
                for key in ("count(*)", "count", "COUNT(*)"):
                    if key in first:
                        return int(first[key])
        except Exception as exc:  # noqa: BLE001
            logger.debug("count(*) query failed (%s); falling back to stats", exc)

        try:
            stats = client.get_collection_stats(self.collection)
            return int(stats.get("row_count", 0))
        except Exception:  # noqa: BLE001
            return 0

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "uri": self.uri if self.lite else _redact(self.uri),
            "collection": self.collection,
            "entities": self._safe_count(),
            "dim": self._dim,
            "index": settings.milvus_index_type,
            "metric": settings.milvus_metric_type,
            "native_bm25": self._native_bm25,
            "bm25": self._bm25.stats() if not self._native_bm25 else {"mode": "server-side"},
        }

    def close(self) -> None:
        if not self._native_bm25:
            try:
                self._bm25.save()
            except Exception:  # noqa: BLE001
                pass
        try:
            if self._client is not None:
                self._client.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- helpers
def _esc(value: str) -> str:
    """Escape a value for inclusion in a Milvus filter string literal."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _normalize_dense(distance: float) -> float:
    """Map the raw metric to a 0..1 similarity so thresholds mean one thing."""
    metric = settings.milvus_metric_type.upper()
    if metric in {"COSINE", "IP"}:
        # Cosine/IP on unit vectors is in [-1, 1]; shift to [0, 1].
        return max(0.0, min(1.0, (distance + 1.0) / 2.0)) if distance < 0 else min(1.0, distance)
    if metric == "L2":
        return 1.0 / (1.0 + max(0.0, distance))
    return distance


def build_filter_expr(filters: Optional[dict[str, Any]]) -> str:
    """Compile a filter dict into a Milvus boolean expression."""
    if not filters:
        return ""
    clauses: list[str] = []

    doc_id = filters.get("doc_id")
    if doc_id:
        clauses.append(f'doc_id == "{_esc(doc_id)}"')

    source = filters.get("source")
    if source:
        clauses.append(f'source == "{_esc(source)}"')

    language = filters.get("language")
    if language:
        clauses.append(f'language == "{_esc(language)}"')

    # District matching is deliberately generous: a chunk qualifies if it is
    # *primarily* about the district OR merely mentions it. Requiring the primary
    # field only makes "I'm from Vijayawada" miss the state-wide manifesto
    # paragraph that names Vijayawada in a list.
    districts = _as_list(filters.get("districts")) or _as_list(filters.get("district"))
    if districts:
        parts = []
        for d in districts:
            esc = _esc(d)
            parts.append(f'district == "{esc}"')
            parts.append(f'ARRAY_CONTAINS(districts, "{esc}")')
        clauses.append("(" + " or ".join(parts) + ")")

    categories = _as_list(filters.get("categories")) or _as_list(filters.get("category"))
    if categories:
        joined = ", ".join(f'"{_esc(c)}"' for c in categories)
        clauses.append(f"category in [{joined}]")

    topics = _as_list(filters.get("topics")) or _as_list(filters.get("topic"))
    if topics:
        parts = []
        for t in topics:
            esc = _esc(t)
            parts.append(f'topic == "{esc}"')
            parts.append(f'ARRAY_CONTAINS(topics, "{esc}")')
        clauses.append("(" + " or ".join(parts) + ")")

    return " and ".join(clauses)


def build_meta_predicate(filters: Optional[dict[str, Any]]):
    """The same filter semantics as `build_filter_expr`, for the local BM25 index."""
    filters = filters or {}
    districts = {d.lower() for d in _as_list(filters.get("districts")) or _as_list(filters.get("district"))}
    categories = {c.lower() for c in _as_list(filters.get("categories")) or _as_list(filters.get("category"))}
    topics = {t.lower() for t in _as_list(filters.get("topics")) or _as_list(filters.get("topic"))}
    doc_id = filters.get("doc_id")
    source = filters.get("source")
    language = filters.get("language")

    def predicate(meta: dict[str, Any]) -> bool:
        if doc_id and meta.get("doc_id") != doc_id:
            return False
        if source and meta.get("source") != source:
            return False
        if language and meta.get("language") != language:
            return False
        if districts:
            owned = {str(meta.get("district") or "").lower()}
            owned.update(str(x).lower() for x in (meta.get("districts") or []))
            if not (owned & districts):
                return False
        if categories and str(meta.get("category") or "").lower() not in categories:
            return False
        if topics:
            owned_t = {str(meta.get("topic") or "").lower()}
            owned_t.update(str(x).lower() for x in (meta.get("topics") or []))
            if not (owned_t & topics):
                return False
        return True

    return predicate


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v]
    return [str(value)]


def _meta_from_row(row: dict[str, Any]) -> ChunkMetadata:
    raw = row.get("meta")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            raw = None
    if isinstance(raw, dict) and raw.get("doc_id"):
        try:
            return ChunkMetadata(**raw)
        except Exception:  # noqa: BLE001 — schema drift shouldn't break search
            pass
    return ChunkMetadata(
        doc_id=row.get("doc_id", "unknown"),
        source=row.get("source", "unknown"),
        category=row.get("category", "other"),
        district=row.get("district") or None,
        districts=row.get("districts") or [],
        topic=row.get("topic") or None,
        topics=row.get("topics") or [],
        section=row.get("section") or None,
        page=row.get("page") or None,
        language=row.get("language", "en"),
        chunk_index=int(row.get("chunk_index", 0) or 0),
    )


def _redact(uri: str) -> str:
    if "@" in uri:
        scheme, _, rest = uri.partition("://")
        return f"{scheme}://***@{rest.split('@')[-1]}"
    return uri
