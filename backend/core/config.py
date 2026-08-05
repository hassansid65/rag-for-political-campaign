"""
Central configuration.

Every knob is env-overridable so the same image runs on a laptop (Milvus Lite /
local fallback) and in production (Milvus standalone or Zilliz Cloud) without a
code change.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent

# Load .env from backend/ first, then project root (project root wins nothing —
# first loader wins, which keeps backend/.env authoritative for local dev).
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(PROJECT_DIR / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    # ------------------------------------------------------------------ app
    app_name: str = "Real-Time RAG — Voice AI Campaign Assistant"
    app_version: str = "1.0.0"
    environment: Literal["dev", "staging", "prod"] = Field("dev", alias="ENVIRONMENT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    host: str = Field("0.0.0.0", alias="HOST")
    port: int = Field(8000, alias="PORT")
    cors_origins: str = Field(
        "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000",
        alias="CORS_ORIGINS",
    )

    # ------------------------------------------------------------- storage
    upload_dir: Path = Field(BACKEND_DIR / "data" / "uploads", alias="UPLOAD_DIR")
    tts_audio_dir: Path = Field(BACKEND_DIR / "data" / "tts_audio", alias="TTS_AUDIO_DIR")
    max_upload_mb: int = Field(50, alias="MAX_UPLOAD_MB")

    # ------------------------------------------------------------- vectors
    # backend: "milvus" (server / Zilliz), "milvus_lite" (embedded file), "local" (numpy fallback)
    vector_backend: Literal["milvus", "milvus_lite", "local"] = Field(
        "milvus", alias="VECTOR_BACKEND"
    )
    milvus_uri: str = Field("http://localhost:19530", alias="MILVUS_URI")
    milvus_token: str = Field("", alias="MILVUS_TOKEN")
    milvus_db_name: str = Field("default", alias="MILVUS_DB_NAME")
    milvus_lite_file: Path = Field(
        BACKEND_DIR / "data" / "milvus_campaign.db", alias="MILVUS_LITE_FILE"
    )
    local_index_file: Path = Field(
        BACKEND_DIR / "data" / "local_index.npz", alias="LOCAL_INDEX_FILE"
    )
    collection_name: str = Field("campaign_chunks", alias="COLLECTION_NAME")
    # HNSW is the latency/recall sweet spot for < 10M vectors on CPU.
    milvus_index_type: str = Field("HNSW", alias="MILVUS_INDEX_TYPE")
    milvus_metric_type: str = Field("COSINE", alias="MILVUS_METRIC_TYPE")
    hnsw_m: int = Field(24, alias="HNSW_M")
    hnsw_ef_construction: int = Field(200, alias="HNSW_EF_CONSTRUCTION")
    hnsw_ef_search: int = Field(96, alias="HNSW_EF_SEARCH")

    # ---------------------------------------------------------- embeddings
    embedding_model: str = Field("BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(384, alias="EMBEDDING_DIM")
    # CPU by default — deliberately. A laptop GPU is faster but adds a CUDA
    # runtime, VRAM limits and driver variance to a system whose CPU numbers
    # already fit the latency budget. Set "cuda" (or "auto" to probe) to opt in.
    embedding_device: str = Field("cpu", alias="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(64, alias="EMBEDDING_BATCH_SIZE")
    embedding_max_seq_length: int = Field(512, alias="EMBEDDING_MAX_SEQ_LENGTH")
    # BGE asymmetric retrieval: queries get an instruction prefix, passages don't.
    embedding_query_prefix: str = Field(
        "Represent this sentence for searching relevant passages: ",
        alias="EMBEDDING_QUERY_PREFIX",
    )
    embedding_cache_size: int = Field(2048, alias="EMBEDDING_CACHE_SIZE")
    embedding_use_onnx: bool = Field(False, alias="EMBEDDING_USE_ONNX")

    # ------------------------------------------------------------ chunking
    # auto     → use record-atomic chunking when a record template is detected,
    #            otherwise structural. This is the right default: it costs nothing
    #            on prose and eliminates cross-record misattribution on record
    #            corpora (see ingestion/records.py).
    # record   → force record detection
    # structural → force heading/size-based chunking
    chunk_strategy: Literal["auto", "record", "structural"] = Field(
        "auto", alias="CHUNK_STRATEGY"
    )
    chunk_size: int = Field(700, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(120, alias="CHUNK_OVERLAP")
    min_chunk_chars: int = Field(80, alias="MIN_CHUNK_CHARS")
    # A record is kept whole even when it exceeds chunk_size — splitting a
    # candidate profile is what produces "whose assets are these?" answers.
    # Only past this ceiling do we fall back to splitting (title still prepended).
    max_record_chars: int = Field(4000, alias="MAX_RECORD_CHARS")
    # A field label must repeat this often to count as part of a record template.
    record_min_label_repeats: int = Field(3, alias="RECORD_MIN_LABEL_REPEATS")
    # Small chunks retrieve precisely but read poorly; we retrieve on the child
    # and expand to the parent window before handing text to the LLM.
    parent_window_chars: int = Field(1800, alias="PARENT_WINDOW_CHARS")
    enable_parent_expansion: bool = Field(True, alias="ENABLE_PARENT_EXPANSION")

    # ----------------------------------------------------------- retrieval
    retrieval_top_k: int = Field(5, alias="RETRIEVAL_TOP_K")
    candidate_top_k: int = Field(30, alias="CANDIDATE_TOP_K")  # per branch, pre-fusion
    rerank_top_n: int = Field(3, alias="RERANK_TOP_N")
    similarity_threshold: float = Field(0.30, alias="SIMILARITY_THRESHOLD")
    rerank_score_threshold: float = Field(-2.0, alias="RERANK_SCORE_THRESHOLD")
    # How far below `rerank_score_threshold` the best candidate may sit before we
    # stop returning it as a best-effort answer and return nothing at all.
    # Measured on this corpus with the MiniLM tier (scripts/test_suite.py):
    #   on-topic, gated to one record        +2.9 … +7.0
    #   on-topic but unanswerable by top-k   -3.1 … -4.6   (aggregation, missing seat)
    #   off-topic entirely                  -10.1 … -11.2   ("how do I cook biryani")
    # The gap is wide and stable, so 6.0 below the bar cleanly separates "this
    # passage is a poor answer" from "this passage has nothing to do with the
    # question". Handing the second kind to the generator is what produced a
    # candidate's profile in reply to "what is the weather today".
    rerank_abandon_margin: float = Field(6.0, alias="RERANK_ABANDON_MARGIN")
    rrf_k: int = Field(60, alias="RRF_K")
    dense_weight: float = Field(1.0, alias="DENSE_WEIGHT")
    sparse_weight: float = Field(0.7, alias="SPARSE_WEIGHT")
    enable_hybrid: bool = Field(True, alias="ENABLE_HYBRID")
    enable_rerank: bool = Field(True, alias="ENABLE_RERANK")
    # Reranking is cascaded because BGE-reranker-base costs ~105ms/pair on CPU
    # at 512-char passages, versus ~16ms/pair for MiniLM-L6. Measured on 12 cores
    # over 16 candidates (scripts/bench_cascade.py):
    #   fast    → MiniLM only                    ~250ms
    #   cascade → MiniLM(16) then BGE(top 4)     ~670ms   (default)
    #   single  → BGE over all 16                ~1690ms
    # A CUDA device collapses all three to tens of milliseconds; these figures are
    # the CPU-only worst case, which is what the voice budget has to survive.
    # `auto` probes available RAM at startup and picks `cascade` when
    # BGE-reranker-base fits (or a GPU is present), else `fast`. On a memory-tight
    # host, loading the precise tier costs ~5x through paging — see core/resources.py.
    rerank_mode: Literal["auto", "fast", "cascade", "single"] = Field(
        "auto", alias="RERANK_MODE"
    )
    reranker_model: str = Field("BAAI/bge-reranker-base", alias="RERANKER_MODEL")
    reranker_fast_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_FAST_MODEL"
    )
    # We return top 3, so rescoring 4 with the precise model is sufficient —
    # going to 6 added ~180ms and never changed the top 3 on this corpus.
    rerank_cascade_keep: int = Field(4, alias="RERANK_CASCADE_KEEP")
    reranker_device: str = Field("cpu", alias="RERANKER_DEVICE")
    # 256 tokens covers a 700-char chunk. 512 doubled per-pair cost for tokens
    # the model then truncated away.
    reranker_max_length: int = Field(256, alias="RERANKER_MAX_LENGTH")
    # Voice turns are latency-bound: cap how many candidates hit the cross-encoder.
    rerank_candidate_cap: int = Field(16, alias="RERANK_CANDIDATE_CAP")
    max_context_chars: int = Field(6000, alias="MAX_CONTEXT_CHARS")

    # ---------------------------------------------------------------- LLM
    # auto → use whichever provider has usable credentials (Anthropic preferred).
    llm_provider: Literal["auto", "anthropic", "azure_openai"] = Field(
        "auto", alias="LLM_PROVIDER"
    )

    # ---- Azure OpenAI ----------------------------------------------------
    # Full chat-completions URLs including api-version, exactly as Azure issues
    # them. Posting to the given URL avoids reassembling it from parts.
    azure_openai_api_key: str = Field("", alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field("", alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_deployment: str = Field("gpt-4-0613", alias="AZURE_OPENAI_DEPLOYMENT")
    # Smaller/faster deployment, used for voice turns and query rewriting where
    # time-to-first-token matters more than reasoning depth.
    azure_openai_fast_endpoint: str = Field("", alias="AZURE_OPENAI_FAST_ENDPOINT")
    azure_openai_fast_deployment: str = Field(
        "gpt-35-turbo-16k-0613", alias="AZURE_OPENAI_FAST_DEPLOYMENT"
    )

    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    # Accepted alias — the console labels the same secret "Claude API key", and
    # having a key silently ignored because of the env-var name is a bad first
    # five minutes with the project.
    claude_api_key: str = Field("", alias="CLAUDE_API_KEY")
    llm_model: str = Field("claude-opus-5", alias="LLM_MODEL")
    llm_max_tokens: int = Field(1024, alias="LLM_MAX_TOKENS")
    # Voice answers must start fast. Thinking off + low effort is the right
    # trade-off for grounded extractive answering over a small context.
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        "low", alias="LLM_EFFORT"
    )
    llm_thinking: Literal["off", "adaptive"] = Field("off", alias="LLM_THINKING")
    llm_enable_prompt_cache: bool = Field(True, alias="LLM_ENABLE_PROMPT_CACHE")
    llm_timeout_s: float = Field(60.0, alias="LLM_TIMEOUT_S")
    llm_max_retries: int = Field(2, alias="LLM_MAX_RETRIES")
    # Rewrite model: a cheap, fast pass for pronoun/ellipsis resolution.
    rewrite_model: str = Field("claude-haiku-4-5", alias="REWRITE_MODEL")
    enable_llm_query_rewrite: bool = Field(True, alias="ENABLE_LLM_QUERY_REWRITE")

    # ------------------------------------------------------------- memory
    memory_max_turns: int = Field(8, alias="MEMORY_MAX_TURNS")
    session_ttl_s: int = Field(3600, alias="SESSION_TTL_S")

    # -------------------------------------------------------------- cache
    enable_semantic_cache: bool = Field(True, alias="ENABLE_SEMANTIC_CACHE")
    semantic_cache_threshold: float = Field(0.97, alias="SEMANTIC_CACHE_THRESHOLD")
    semantic_cache_size: int = Field(512, alias="SEMANTIC_CACHE_SIZE")
    semantic_cache_ttl_s: int = Field(900, alias="SEMANTIC_CACHE_TTL_S")

    # ---------------------------------------------------- streaming voice
    # Only fire speculative retrieval once the partial transcript looks like it
    # carries signal — below this it is mostly filler ("um", "so I").
    partial_min_chars: int = Field(12, alias="PARTIAL_MIN_CHARS")
    partial_min_tokens: int = Field(3, alias="PARTIAL_MIN_TOKENS")
    partial_debounce_ms: int = Field(180, alias="PARTIAL_DEBOUNCE_MS")
    # If the final transcript embeds this close to the partial we already
    # retrieved for, reuse those results instead of re-running the pipeline.
    partial_reuse_threshold: float = Field(0.94, alias="PARTIAL_REUSE_THRESHOLD")

    # -------------------------------------------------------- azure speech
    azure_speech_key: str = Field("", alias="AZURE_SPEECH_KEY")
    azure_speech_region: str = Field("centralindia", alias="AZURE_SPEECH_REGION")
    azure_speech_endpoint: str = Field("", alias="AZURE_SPEECH_ENDPOINT")
    azure_tts_voice: str = Field("en-IN-NeerjaNeural", alias="AZURE_TTS_VOICE")
    azure_stt_language: str = Field("en-IN", alias="AZURE_STT_LANGUAGE")
    azure_stt_languages: str = Field("en-IN,te-IN,hi-IN", alias="AZURE_STT_LANGUAGES")
    azure_tts_rate: str = Field("+8%", alias="AZURE_TTS_RATE")

    # ------------------------------------------------------------- lipsync
    rhubarb_path: str = Field("", alias="RHUBARB_PATH")
    ffmpeg_path: str = Field("", alias="FFMPEG_PATH")
    enable_lipsync: bool = Field(True, alias="ENABLE_LIPSYNC")

    # ----------------------------------------------------------- helpers
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def stt_language_list(self) -> list[str]:
        return [x.strip() for x in self.azure_stt_languages.split(",") if x.strip()]

    @property
    def azure_speech_configured(self) -> bool:
        return bool(self.azure_speech_key and self.azure_speech_region)

    @property
    def resolved_anthropic_key(self) -> str:
        return self.anthropic_api_key or self.claude_api_key

    @property
    def llm_configured(self) -> bool:
        return bool(self.resolved_anthropic_key or os.getenv("ANTHROPIC_AUTH_TOKEN"))

    def ensure_dirs(self) -> None:
        for path in (self.upload_dir, self.tts_audio_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.milvus_lite_file.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()  # type: ignore[call-arg]
    settings.ensure_dirs()
    return settings


settings = get_settings()
