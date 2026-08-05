"""
Store selection with graceful degradation.

Order of preference is the configured backend, then Milvus Lite, then the local
NumPy store. The fallback chain is deliberate: a missing vector DB should make
the system slower and less scalable, never non-functional — a reviewer cloning
this repo on Windows without Docker still gets a working demo, and the log line
tells them exactly what happened and how to get the real thing.
"""

from __future__ import annotations

import logging
import platform
import threading
from typing import Optional

from core.config import settings
from vectorstore.base import VectorStore

logger = logging.getLogger(__name__)

_store: Optional[VectorStore] = None
_lock = threading.Lock()
_fallback_reason: str = ""


def _try_milvus(lite: bool) -> Optional[VectorStore]:
    from vectorstore.milvus_store import MilvusStore

    store = MilvusStore(lite=lite)
    store.connect()
    store.ensure_collection(settings.embedding_dim)
    return store


def build_store(force_backend: Optional[str] = None) -> VectorStore:
    global _fallback_reason
    backend = (force_backend or settings.vector_backend).lower()
    attempts: list[str] = []

    if backend == "milvus":
        attempts = ["milvus", "milvus_lite", "local"]
    elif backend == "milvus_lite":
        attempts = ["milvus_lite", "local"]
    else:
        attempts = ["local"]

    errors: list[str] = []
    for candidate in attempts:
        try:
            if candidate == "local":
                from vectorstore.local_store import LocalStore

                store: VectorStore = LocalStore()
                store.connect()
                store.ensure_collection(settings.embedding_dim)
            else:
                if candidate == "milvus_lite" and platform.system() == "Windows":
                    raise RuntimeError(
                        "milvus-lite has no Windows build — use Milvus standalone "
                        "(docker compose up -d milvus) or VECTOR_BACKEND=local"
                    )
                store = _try_milvus(lite=candidate == "milvus_lite")  # type: ignore[assignment]

            if candidate != backend:
                _fallback_reason = (
                    f"requested backend '{backend}' unavailable "
                    f"({errors[0] if errors else 'unknown'}); using '{candidate}'"
                )
                logger.warning("Vector store fallback: %s", _fallback_reason)
            else:
                _fallback_reason = ""
                logger.info("Vector store: %s", candidate)
            return store

        except Exception as exc:  # noqa: BLE001
            message = f"{candidate}: {exc}"
            errors.append(message)
            logger.warning("Vector store '%s' unavailable — %s", candidate, exc)

    raise RuntimeError("No vector store backend could be initialised: " + "; ".join(errors))


def get_store() -> VectorStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = build_store()
    return _store


def reset_store() -> None:
    """Drop the singleton (tests, or after a backend config change)."""
    global _store
    with _lock:
        if _store is not None:
            try:
                _store.close()
            except Exception:  # noqa: BLE001
                pass
        _store = None


def fallback_reason() -> str:
    return _fallback_reason
