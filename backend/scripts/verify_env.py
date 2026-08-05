"""
Environment self-check: is this interpreter able to run the project?

Run this first when anything looks wrong. It verifies the Python version, that
every dependency imports, that the isolated venv is actually in use, and which
optional integrations are configured — before you spend time debugging a symptom
that is really a missing package or a global-site-packages conflict.

    python scripts/verify_env.py
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# (module, pip name, required)
DEPENDENCIES: list[tuple[str, str, bool]] = [
    ("fastapi", "fastapi", True),
    ("uvicorn", "uvicorn", True),
    ("pydantic", "pydantic", True),
    ("pydantic_settings", "pydantic-settings", True),
    ("dotenv", "python-dotenv", True),
    ("multipart", "python-multipart", True),
    ("websockets", "websockets", True),
    ("fitz", "pymupdf", True),
    ("docx", "python-docx", True),
    ("numpy", "numpy", True),
    ("torch", "torch", True),
    ("sentence_transformers", "sentence-transformers", True),
    ("transformers", "transformers", True),
    ("pkg_resources", "setuptools<81", True),
    ("pymilvus", "pymilvus", True),
    ("anthropic", "anthropic", True),
    ("httpx", "httpx", True),
    ("azure.cognitiveservices.speech", "azure-cognitiveservices-speech", False),
    ("pytest", "pytest", False),
    ("unstructured", "unstructured (optional parser)", False),
    ("onnxruntime", "onnxruntime (optional)", False),
]

PROJECT_MODULES = [
    "core.config",
    "core.resources",
    "ingestion.loader",
    "ingestion.records",
    "ingestion.chunker",
    "embeddings.embedder",
    "vectorstore.factory",
    "retrieval.pipeline",
    "retrieval.reranker",
    "llm.claude_client",
    "llm.rag_service",
    "voice.azure_speech",
    "voice.streaming",
    "api.main",
]


def rule(char: str = "-", width: int = 74) -> None:
    print(char * width)


def main() -> int:
    problems: list[str] = []
    warnings: list[str] = []

    rule("=")
    print("  ENVIRONMENT VERIFICATION")
    rule("=")

    # ------------------------------------------------------------- interpreter
    version = sys.version_info
    print(f"  python            : {sys.version.split()[0]}")
    print(f"  executable        : {sys.executable}")
    if version[:2] != (3, 10):
        warnings.append(
            f"Python {version.major}.{version.minor} — the project is pinned to 3.10; "
            "other versions may resolve different wheels"
        )

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print(f"  virtualenv        : {'yes → ' + sys.prefix if in_venv else 'NO'}")
    if not in_venv:
        warnings.append(
            "Not running inside a virtualenv. Global site-packages can hold "
            "conflicting pins (torch/numpy/transformers). Use backend\\.venv."
        )

    # ------------------------------------------------------------ dependencies
    print()
    rule()
    print("  DEPENDENCIES")
    rule()
    for module, pip_name, required in DEPENDENCIES:
        try:
            loaded = importlib.import_module(module)
            installed = getattr(loaded, "__version__", "")
            print(f"    ok       {pip_name:<40} {installed}")
        except Exception as exc:
            if required:
                problems.append(f"{pip_name} missing ({type(exc).__name__})")
                print(f"    MISSING  {pip_name:<40} {type(exc).__name__}")
            else:
                print(f"    absent   {pip_name:<40} (optional)")

    # --------------------------------------------------------- project modules
    print()
    rule()
    print("  PROJECT MODULES")
    rule()
    for module in PROJECT_MODULES:
        try:
            importlib.import_module(module)
            print(f"    ok       {module}")
        except Exception as exc:
            problems.append(f"{module}: {type(exc).__name__}: {exc}")
            print(f"    FAIL     {module}: {type(exc).__name__}: {exc}")

    # ----------------------------------------------------------- configuration
    print()
    rule()
    print("  CONFIGURATION")
    rule()
    try:
        from core.config import settings
        from core.resources import probe, resolve_device, resolve_rerank_mode

        env_path = BACKEND_DIR / ".env"
        print(f"    .env file       : {'found' if env_path.exists() else 'MISSING (copy .env.example)'}")
        print(f"    vector backend  : {settings.vector_backend}")
        print(f"    chunk strategy  : {settings.chunk_strategy}")
        print(f"    embedding model : {settings.embedding_model} ({settings.embedding_dim}d)")
        print(f"    embed device    : {settings.embedding_device} → {resolve_device(settings.embedding_device)}")
        mode, reason = resolve_rerank_mode(settings.rerank_mode)
        print(f"    rerank mode     : {reason}")
        print(f"    host resources  : {probe().as_dict()}")

        if settings.llm_configured:
            key = settings.resolved_anthropic_key
            print(f"    Anthropic key   : set ({key[:14]}…{key[-4:]}, {len(key)} chars)")
        else:
            print("    Anthropic key   : NOT SET")
            warnings.append(
                "ANTHROPIC_API_KEY is not set — /query returns retrieval-only fallbacks"
            )

        if settings.azure_speech_configured:
            print(f"    Azure Speech    : set (region={settings.azure_speech_region})")
        else:
            print("    Azure Speech    : NOT SET")
            warnings.append("AZURE_SPEECH_KEY is not set — voice endpoints return 503")

        if not env_path.exists():
            problems.append("backend/.env is missing")
    except Exception as exc:
        problems.append(f"config load failed: {exc}")
        print(f"    FAIL: {exc}")

    # ------------------------------------------------------------------ assets
    print()
    rule()
    print("  ASSETS")
    rule()
    checks = [
        (BACKEND_DIR.parent / "data" / "RAG_Test_Candidate_Profiles.pdf", "candidate PDF", False),
        (BACKEND_DIR.parent / "data" / "sample_docs", "sample docs", False),
        (BACKEND_DIR / "bin" / "rhubarb.exe", "rhubarb (lip-sync fallback)", False),
        (BACKEND_DIR.parent / "frontend" / "public" / "Shayla_Changes(Visemes).glb", "avatar GLB", False),
    ]
    for path, label, required in checks:
        exists = path.exists()
        size = f"{path.stat().st_size / 1024:.0f} KB" if exists and path.is_file() else ""
        print(f"    {'ok      ' if exists else 'absent  '} {label:<34} {size}")
        if required and not exists:
            problems.append(f"{label} missing at {path}")

    # ------------------------------------------------------------------ verdict
    print()
    rule("=")
    if problems:
        print(f"  {len(problems)} PROBLEM(S) — the app will not run correctly:")
        for item in problems:
            print(f"    ✗ {item}")
    if warnings:
        print(f"  {len(warnings)} WARNING(S) — the app runs with reduced capability:")
        for item in warnings:
            print(f"    ! {item}")
    if not problems and not warnings:
        print("  ALL CHECKS PASSED")
    elif not problems:
        print("  READY (with warnings above)")
    rule("=")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
