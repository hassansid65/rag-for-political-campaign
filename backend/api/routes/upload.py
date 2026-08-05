"""POST /upload — ingest campaign documents (PDF / DOCX / TXT / MD / HTML / CSV)."""

from __future__ import annotations

import json
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from core.config import settings
from core.latency import Trace
from core.schemas import (
    DocumentListResponse,
    DocumentSummary,
    MetadataOverride,
    UploadResponse,
)
from ingestion.loader import SUPPORTED_EXTENSIONS
from ingestion.service import get_ingest_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingestion"])


@router.post("/upload", response_model=UploadResponse)
async def upload(
    files: Annotated[list[UploadFile], File(description="One or more documents")],
    category: Annotated[Optional[str], Form()] = None,
    district: Annotated[Optional[str], Form()] = None,
    state: Annotated[Optional[str], Form()] = None,
    topic: Annotated[Optional[str], Form()] = None,
    candidate: Annotated[Optional[str], Form()] = None,
    party: Annotated[Optional[str], Form()] = None,
    metadata: Annotated[Optional[str], Form(description="JSON metadata override")] = None,
) -> UploadResponse:
    """Ingest documents: parse → chunk → embed → index.

    Metadata is auto-extracted (district, category, topic, schemes, people) and any
    field supplied here overrides the inference. Multi-file upload is supported so a
    whole campaign folder can be indexed in one call; one bad file does not fail the
    batch — it comes back in `documents`-adjacent warnings and the rest still index.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files were provided")

    override = _build_override(
        metadata=metadata,
        category=category,
        district=district,
        state=state,
        topic=topic,
        candidate=candidate,
        party=party,
    )

    service = get_ingest_service()
    trace = Trace(name="upload")
    documents = []
    warnings: list[str] = []
    total_chunks = 0

    for upload_file in files:
        filename = upload_file.filename or "unnamed"
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in SUPPORTED_EXTENSIONS:
            warnings.append(
                f"{filename}: unsupported type '{suffix or 'none'}' "
                f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
            )
            continue

        try:
            data = await upload_file.read()
        finally:
            await upload_file.close()

        if not data:
            warnings.append(f"{filename}: file is empty")
            continue
        size_mb = len(data) / (1024 * 1024)
        if size_mb > settings.max_upload_mb:
            warnings.append(
                f"{filename}: {size_mb:.1f} MB exceeds the "
                f"{settings.max_upload_mb} MB limit"
            )
            continue

        with trace.stage("save"):
            path = service.save_upload(filename, data)

        outcome = await service.ingest_file(path, override=override, trace=trace)
        if not outcome.ok:
            warnings.append(outcome.error or f"{filename}: ingestion failed")
            continue

        documents.append(outcome.document)
        total_chunks += outcome.document.chunks_indexed  # type: ignore[union-attr]

    if not documents:
        raise HTTPException(
            status_code=422,
            detail={"message": "No document could be ingested", "warnings": warnings},
        )

    status = "success" if not warnings else "partial"
    message = f"Indexed {total_chunks} chunks from {len(documents)} document(s)"
    if warnings:
        message += f"; {len(warnings)} file(s) skipped"

    return UploadResponse(
        status=status,  # type: ignore[arg-type]
        documents=documents,  # type: ignore[arg-type]
        total_chunks_indexed=total_chunks,
        collection=settings.collection_name,
        timings_ms=trace.finish(),
        message=message,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List indexed documents with their inferred metadata and chunk counts."""
    service = get_ingest_service()
    raw = service.list_documents()
    documents = [
        DocumentSummary(
            doc_id=item["doc_id"],
            source=item.get("source", ""),
            category=item.get("category", "other"),
            districts=item.get("districts", []),
            topics=item.get("topics", []),
            chunks=item.get("chunks", 0),
            ingested_at=item.get("ingested_at"),
        )
        for item in sorted(raw, key=lambda d: d.get("source", ""))
    ]
    return DocumentListResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=sum(d.chunks for d in documents),
    )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    """Remove a document and all of its chunks from the index."""
    service = get_ingest_service()
    removed = service.delete_document(doc_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No chunks found for doc_id={doc_id}")
    return {"status": "deleted", "doc_id": doc_id, "chunks_removed": removed}


@router.post("/ingest-path")
async def ingest_path(
    path: Annotated[str, Form()],
    recursive: Annotated[bool, Form()] = True,
) -> dict:
    """Ingest a server-side file or directory — the bulk path for seeding a corpus."""
    from pathlib import Path

    target = Path(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    service = get_ingest_service()
    if target.is_dir():
        outcomes = await service.ingest_directory(target, recursive=recursive)
    else:
        outcomes = [await service.ingest_file(target)]

    ok = [o for o in outcomes if o.ok]
    return {
        "status": "success" if ok else "error",
        "documents": [o.document for o in ok],
        "total_chunks_indexed": sum(o.document.chunks_indexed for o in ok),  # type: ignore[union-attr]
        "errors": [o.error for o in outcomes if o.error],
    }


def _build_override(
    *,
    metadata: Optional[str],
    category: Optional[str],
    district: Optional[str],
    state: Optional[str],
    topic: Optional[str],
    candidate: Optional[str],
    party: Optional[str],
) -> Optional[MetadataOverride]:
    payload: dict = {}

    if metadata:
        try:
            parsed = json.loads(metadata)
            if isinstance(parsed, dict):
                payload.update(parsed)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail=f"`metadata` is not valid JSON: {exc}"
            ) from exc

    # Discrete form fields win over the JSON blob — they are more explicit.
    for key, value in (
        ("category", category),
        ("district", district),
        ("state", state),
        ("topic", topic),
        ("candidate", candidate),
        ("party", party),
    ):
        if value:
            payload[key] = value

    if not payload:
        return None
    try:
        return MetadataOverride(**payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid metadata: {exc}") from exc
