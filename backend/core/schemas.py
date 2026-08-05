"""Shared Pydantic models — the wire contract for every endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Category = Literal[
    "manifesto",
    "district_info",
    "candidate_profile",
    "scheme",
    "faq",
    "press_release",
    "speech",
    "other",
]


# --------------------------------------------------------------------- chunks
class ChunkMetadata(BaseModel):
    """The metadata contract required by the assignment, plus retrieval aids."""

    doc_id: str
    source: str                      # original filename
    source_path: Optional[str] = None
    category: Category = "other"
    district: Optional[str] = None   # e.g. "Vijayawada"
    districts: list[str] = Field(default_factory=list)  # all districts mentioned
    state: Optional[str] = None
    topic: Optional[str] = None
    topics: list[str] = Field(default_factory=list)
    section: Optional[str] = None    # nearest heading
    section_path: list[str] = Field(default_factory=list)  # heading breadcrumb
    page: Optional[int] = None
    language: str = "en"
    candidate: Optional[str] = None
    party: Optional[str] = None
    scheme_names: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 0
    char_start: int = 0
    char_end: int = 0
    ingested_at: Optional[str] = None

    # ---- record-atomic chunking (see ingestion/records.py) ----------------
    # Set when this chunk is exactly one record (a candidate, a scheme, a Q&A).
    # `record_name` is the entity the record is about and is what makes
    # per-entity retrieval — and per-entity answer verification — possible on a
    # corpus of near-identical profiles.
    is_record: bool = False
    record_name: Optional[str] = None
    record_title: Optional[str] = None
    record_labels: list[str] = Field(default_factory=list)
    constituency: Optional[str] = None

    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata
    parent_text: Optional[str] = None


class RetrievedChunk(BaseModel):
    id: str
    # `text` is what goes to the LLM — the parent window when expansion is on.
    # `chunk_text` is the child chunk that actually matched, which is what a
    # citation snippet and any UI highlight must show. Conflating the two makes
    # citations point at a neighbouring section's opening line.
    text: str
    chunk_text: Optional[str] = None
    metadata: ChunkMetadata
    score: float                          # final score used for ordering
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retriever: str = "hybrid"             # dense | sparse | hybrid


class Citation(BaseModel):
    marker: str            # "[1]"
    source: str
    category: Category
    district: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    chunk_id: str
    score: float
    snippet: str


# --------------------------------------------------------------------- upload
class MetadataOverride(BaseModel):
    category: Optional[Category] = None
    district: Optional[str] = None
    state: Optional[str] = None
    topic: Optional[str] = None
    candidate: Optional[str] = None
    party: Optional[str] = None
    language: Optional[str] = None


class UploadedDocument(BaseModel):
    doc_id: str
    source: str
    category: Category
    districts: list[str]
    topics: list[str]
    pages: Optional[int] = None
    chars: int
    chunks_indexed: int
    detected_language: str
    warnings: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    status: Literal["success", "partial", "error"]
    documents: list[UploadedDocument]
    total_chunks_indexed: int
    collection: str
    timings_ms: dict[str, Any]
    message: str = ""


# ------------------------------------------------------------------- retrieve
class RetrieveFilters(BaseModel):
    district: Optional[str] = None
    districts: list[str] = Field(default_factory=list)
    category: Optional[Category] = None
    categories: list[Category] = Field(default_factory=list)
    topic: Optional[str] = None
    source: Optional[str] = None
    doc_id: Optional[str] = None
    language: Optional[str] = None

    def is_empty(self) -> bool:
        return not any(
            [
                self.district,
                self.districts,
                self.category,
                self.categories,
                self.topic,
                self.source,
                self.doc_id,
                self.language,
            ]
        )


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: Optional[int] = Field(None, ge=1, le=50)
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    filters: Optional[RetrieveFilters] = None
    session_id: Optional[str] = None
    rerank: Optional[bool] = None
    hybrid: Optional[bool] = None
    rewrite_query: bool = True
    include_text: bool = True

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class RetrieveResponse(BaseModel):
    query: str
    effective_query: str
    rewrites: list[str] = Field(default_factory=list)
    inferred_filters: dict[str, Any] = Field(default_factory=dict)
    results: list[RetrievedChunk]
    total_candidates: int
    reranked: bool
    cache_hit: bool = False
    timings_ms: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------- query
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    top_k: Optional[int] = Field(None, ge=1, le=50)
    filters: Optional[RetrieveFilters] = None
    stream: bool = False
    include_citations: bool = True
    include_context: bool = False
    rewrite_query: bool = True
    voice_mode: bool = False          # terser, TTS-friendly phrasing
    max_tokens: Optional[int] = Field(None, ge=32, le=8192)

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    query: str
    effective_query: str
    grounded: bool
    citations: list[Citation] = Field(default_factory=list)
    sources_used: int = 0
    context: Optional[str] = None
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    inferred_filters: dict[str, Any] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    cache_hit: bool = False
    timings_ms: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------- health
class ComponentHealth(BaseModel):
    name: str
    status: Literal["ok", "degraded", "down", "disabled"]
    detail: str = ""
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    environment: str
    uptime_s: float
    timestamp: datetime
    components: list[ComponentHealth]
    collection: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    latency_ms: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------- voice
class STTResponse(BaseModel):
    text: str
    success: bool
    language: Optional[str] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: Optional[str] = None
    rate: Optional[str] = None
    session_id: Optional[str] = None


class MouthCue(BaseModel):
    start: float
    end: float
    value: str


class LipSyncPayload(BaseModel):
    mouthCues: list[MouthCue]
    metadata: dict[str, Any] = Field(default_factory=dict)


class VoiceTurnRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = "default"
    voice: Optional[str] = None
    filters: Optional[RetrieveFilters] = None
    include_citations: bool = True


class VoiceTurnResponse(BaseModel):
    text: str
    spoken_text: str
    session_id: str
    audio: Optional[str] = None          # base64 wav
    audio_format: str = "wav"
    lipsync: Optional[LipSyncPayload] = None
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
    facialExpression: str = "default"
    animation: str = "Idle"
    timings_ms: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ documents
class DocumentSummary(BaseModel):
    doc_id: str
    source: str
    category: Category
    districts: list[str]
    topics: list[str]
    chunks: int
    ingested_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total_documents: int
    total_chunks: int
