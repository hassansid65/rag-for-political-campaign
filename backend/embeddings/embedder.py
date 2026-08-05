"""
BGE-small-en-v1.5 embedding layer.

Why this model: 384 dims, 33M params, ~0.5 GB RAM, and MTEB-retrieval scores
within a couple of points of models 10x its size. On CPU it embeds a voice-length
query in **3–8 ms**, which is what makes a sub-second voice turn possible at all.
A 1536-dim API embedding would add 80–250 ms of network round-trip per turn
before we even reach the vector DB.

Three correctness details that are easy to get wrong and cost real recall:

1. **Asymmetric prefixing.** BGE is trained with an instruction prefix on the
   *query* side only. Prefixing passages too (or neither) measurably degrades
   retrieval. `encode_query` adds it; `encode_passages` never does.
2. **L2 normalization.** Required for cosine/IP equivalence. We normalize here
   so the store can use IP (fastest metric) and still mean cosine.
3. **Warmup.** The first forward pass pays lazy-init cost (~1–3 s). We burn it
   at startup, not on the user's first question.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Iterable, Optional, Sequence

import numpy as np

from core.config import settings
from core.latency import METRICS
from core.model_loading import MODEL_LOAD_LOCK
from core.resources import resolve_device

logger = logging.getLogger(__name__)


class _LRUCache:
    """Tiny thread-safe LRU for query vectors.

    Voice traffic is extremely repetitive — partial transcripts converge on the
    same string, and 'what about my district' gets asked in every session. A hit
    here skips the whole forward pass.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, capacity)
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        if not self.capacity:
            return None
        with self._lock:
            vector = self._data.get(key)
            if vector is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return vector

    def put(self, key: str, value: np.ndarray) -> None:
        if not self.capacity:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "size": len(self._data),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


class Embedder:
    """Thin, hot-path-optimized wrapper around BGE-small-en-v1.5."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        query_prefix: Optional[str] = None,
        cache_size: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        # "auto" → CUDA when present with enough VRAM, else CPU.
        self.device = resolve_device(device or settings.embedding_device)
        self.query_prefix = (
            settings.embedding_query_prefix if query_prefix is None else query_prefix
        )
        self.dim = settings.embedding_dim
        self.batch_size = settings.embedding_batch_size
        self._cache = _LRUCache(
            settings.embedding_cache_size if cache_size is None else cache_size
        )
        self._model = None
        self._backend = "uninitialized"
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- load
    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            start = time.perf_counter()

            if settings.embedding_use_onnx:
                try:
                    self._load_onnx()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "ONNX embedding backend unavailable (%s); using sentence-transformers",
                        exc,
                    )
                    self._load_sentence_transformers()
            else:
                self._load_sentence_transformers()

            logger.info(
                "Embedder ready: %s (backend=%s, dim=%d, device=%s) in %.2fs",
                self.model_name,
                self._backend,
                self.dim,
                self.device,
                time.perf_counter() - start,
            )
            self._warmup()

    def _load_sentence_transformers(self) -> None:
        from sentence_transformers import SentenceTransformer

        # Serialized against every other model construction in the process —
        # see core/model_loading.py for why concurrent loads corrupt weights.
        with MODEL_LOAD_LOCK:
            model = SentenceTransformer(self.model_name, device=self.device)
        model.max_seq_length = settings.embedding_max_seq_length
        self._model = model
        self._backend = "sentence-transformers"
        actual = model.get_sentence_embedding_dimension()
        if actual and actual != self.dim:
            logger.warning(
                "EMBEDDING_DIM=%d but %s produces %d dims; using %d. "
                "Re-index if the collection was built at the old dimension.",
                self.dim, self.model_name, actual, actual,
            )
            self.dim = actual

    def _load_onnx(self) -> None:
        """Optional ONNX Runtime path — ~2x faster on CPU, no torch at runtime."""
        import onnxruntime as ort  # noqa: F401
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        with MODEL_LOAD_LOCK:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = ORTModelForFeatureExtraction.from_pretrained(self.model_name, export=True)
        self._model = (tokenizer, model)
        self._backend = "onnxruntime"

    def _warmup(self) -> None:
        try:
            self.encode_query("warmup")
            self.encode_passages(["warmup passage"])
        except Exception as exc:  # noqa: BLE001 — warmup must never block startup
            logger.warning("Embedder warmup failed: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def backend(self) -> str:
        return self._backend

    # --------------------------------------------------------------- encode
    def encode_query(self, query: str, use_cache: bool = True) -> np.ndarray:
        """Encode a single search query (with the BGE instruction prefix)."""
        key = query.strip().lower()
        if use_cache:
            cached = self._cache.get(key)
            if cached is not None:
                METRICS.incr("embed.query.cache_hit")
                return cached

        start = time.perf_counter()
        vector = self._encode([f"{self.query_prefix}{query}"])[0]
        METRICS.observe("embed.query", (time.perf_counter() - start) * 1000)
        METRICS.incr("embed.query.cache_miss")

        if use_cache:
            self._cache.put(key, vector)
        return vector

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Batch-encode queries — used for multi-query expansion."""
        if not queries:
            return np.zeros((0, self.dim), dtype=np.float32)
        start = time.perf_counter()
        vectors = self._encode([f"{self.query_prefix}{q}" for q in queries])
        METRICS.observe("embed.queries", (time.perf_counter() - start) * 1000)
        return vectors

    def encode_passages(self, passages: Sequence[str]) -> np.ndarray:
        """Encode documents/chunks — no prefix (BGE is asymmetric)."""
        if not passages:
            return np.zeros((0, self.dim), dtype=np.float32)
        start = time.perf_counter()
        vectors = self._encode(list(passages))
        METRICS.observe("embed.passages", (time.perf_counter() - start) * 1000)
        return vectors

    # ------------------------------------------------------------- internals
    def _encode(self, texts: list[str]) -> np.ndarray:
        self.load()
        if self._backend == "onnxruntime":
            return self._encode_onnx(texts)

        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # unit vectors => IP == cosine
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _encode_onnx(self, texts: list[str]) -> np.ndarray:
        tokenizer, model = self._model  # type: ignore[misc]
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=settings.embedding_max_seq_length,
                return_tensors="np",
            )
            hidden = model(**encoded).last_hidden_state
            # BGE uses the CLS token, not mean pooling. Mean pooling here silently
            # costs a few points of nDCG.
            cls = hidden[:, 0]
            norms = np.linalg.norm(cls, axis=1, keepdims=True)
            out.append((cls / np.maximum(norms, 1e-12)).astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    # ------------------------------------------------------------------ misc
    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for already-normalized vectors (falls back safely)."""
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def cache_stats(self) -> dict[str, float]:
        return self._cache.stats()

    def health(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "backend": self._backend,
            "dim": self.dim,
            "device": self.device,
            "ready": self.is_ready,
            "cache": self.cache_stats(),
        }


_embedder: Optional[Embedder] = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    """Process-wide singleton — loading the model twice doubles RAM for nothing."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder
