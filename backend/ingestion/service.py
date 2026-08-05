"""Ingestion orchestration: load → extract metadata → chunk → embed → index."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.config import settings
from core.latency import Trace
from core.schemas import Chunk, MetadataOverride, UploadedDocument
from ingestion.chunker import DocumentChunker
from ingestion.loader import SUPPORTED_EXTENSIONS, UnsupportedFileType, load_document
from ingestion.metadata import STATE_BY_DISTRICT, extract_metadata, resolve_district
from retrieval.pipeline import RetrievalPipeline, get_pipeline

logger = logging.getLogger(__name__)


@dataclass
class IngestOutcome:
    document: Optional[UploadedDocument] = None
    chunks: list[Chunk] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.document is not None


class IngestService:
    def __init__(
        self,
        pipeline: Optional[RetrievalPipeline] = None,
        chunker: Optional[DocumentChunker] = None,
    ) -> None:
        self.pipeline = pipeline or get_pipeline()
        self.chunker = chunker or DocumentChunker()

    # ------------------------------------------------------------------ files
    def save_upload(self, filename: str, data: bytes) -> Path:
        """Persist an upload. The stored copy is what /documents re-indexes from."""
        safe = _safe_filename(filename)
        target = settings.upload_dir / safe
        # Never silently overwrite a different document with the same name.
        if target.exists() and target.read_bytes() != data:
            stem, suffix = target.stem, target.suffix
            counter = 2
            while target.exists():
                target = settings.upload_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    # -------------------------------------------------------------- ingestion
    async def ingest_file(
        self,
        path: Path,
        *,
        override: Optional[MetadataOverride] = None,
        trace: Optional[Trace] = None,
        index: bool = True,
    ) -> IngestOutcome:
        trace = trace or Trace(name="ingest")

        try:
            with trace.stage("load"):
                document = load_document(path)
        except UnsupportedFileType as exc:
            return IngestOutcome(error=str(exc))
        except FileNotFoundError:
            return IngestOutcome(error=f"{path.name}: file not found")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load %s: %s", path.name, exc, exc_info=True)
            return IngestOutcome(error=f"{path.name}: could not be parsed ({exc})")

        if not document.blocks:
            return IngestOutcome(
                error=(
                    f"{path.name}: no extractable text. "
                    "If this is a scanned PDF, OCR it first (e.g. ocrmypdf)."
                )
            )

        with trace.stage("metadata"):
            doc_meta = extract_metadata(document.raw_text, filename=path.name)
            _apply_override(doc_meta, override)

        with trace.stage("chunk"):
            chunks = self.chunker.chunk(
                document, doc_meta, source_path=str(path)
            )

        if not chunks:
            return IngestOutcome(error=f"{path.name}: produced no chunks")

        indexed = 0
        if index:
            indexed = await self.pipeline.index_chunks(chunks, trace=trace)

        summary = UploadedDocument(
            doc_id=chunks[0].metadata.doc_id,
            source=document.source,
            category=doc_meta.category,  # type: ignore[arg-type]
            districts=doc_meta.districts[:10],
            topics=doc_meta.topics[:6],
            pages=document.pages,
            chars=document.char_count,
            chunks_indexed=indexed if index else len(chunks),
            detected_language=document.detected_language,
            warnings=list(document.warnings),
        )
        logger.info(
            "Ingested %s → %d chunks (category=%s, districts=%s)",
            document.source, len(chunks), doc_meta.category, doc_meta.districts[:3],
        )
        return IngestOutcome(document=summary, chunks=chunks)

    async def ingest_directory(
        self,
        directory: Path,
        *,
        override: Optional[MetadataOverride] = None,
        recursive: bool = True,
    ) -> list[IngestOutcome]:
        pattern = "**/*" if recursive else "*"
        outcomes: list[IngestOutcome] = []
        for path in sorted(directory.glob(pattern)):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                outcomes.append(await self.ingest_file(path, override=override))
        return outcomes

    # ------------------------------------------------------------- management
    def delete_document(self, doc_id: str) -> int:
        removed = self.pipeline.store.delete_document(doc_id)
        if removed:
            self.pipeline.cache.invalidate_all()
        return removed

    def list_documents(self) -> list[dict[str, Any]]:
        return self.pipeline.store.list_documents()

    def clear_uploads(self) -> int:
        count = 0
        for path in settings.upload_dir.glob("*"):
            if path.is_file():
                path.unlink(missing_ok=True)
                count += 1
        return count


def _apply_override(doc_meta, override: Optional[MetadataOverride]) -> None:
    """Explicit metadata from the uploader beats anything we inferred."""
    if override is None:
        return
    if override.category:
        doc_meta.category = override.category
    if override.district:
        resolved = resolve_district(override.district) or override.district
        doc_meta.district = resolved
        if resolved not in doc_meta.districts:
            doc_meta.districts.insert(0, resolved)
        doc_meta.state = STATE_BY_DISTRICT.get(resolved, doc_meta.state)
    if override.state:
        doc_meta.state = override.state
    if override.topic:
        doc_meta.topic = override.topic
        if override.topic not in doc_meta.topics:
            doc_meta.topics.insert(0, override.topic)
    if override.candidate:
        doc_meta.candidate = override.candidate
    if override.party:
        doc_meta.party = override.party


_INVALID = '<>:"/\\|?*'


def _safe_filename(name: str) -> str:
    base = Path(name).name or "upload.bin"
    cleaned = "".join("_" if ch in _INVALID else ch for ch in base).strip(". ")
    return cleaned[:180] or "upload.bin"


_service: Optional[IngestService] = None


def get_ingest_service() -> IngestService:
    global _service
    if _service is None:
        _service = IngestService()
    return _service


def set_ingest_service(service: Optional[IngestService]) -> None:
    global _service
    _service = service
