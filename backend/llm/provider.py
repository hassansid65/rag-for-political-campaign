"""
LLM provider selection.

Both clients expose the same surface, so everything above this layer —
`RAGService`, the voice loop, the query rewriter — is provider-agnostic. Switching
is one env var.

`LLM_PROVIDER=auto` (the default) picks whichever provider actually has usable
credentials, preferring Anthropic when both are present. That is not laziness: an
invalid key is indistinguishable from a missing one until you call the API, and a
system that hard-fails on a dead key when a working alternative is configured is
strictly worse than one that logs the substitution and answers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol, runtime_checkable

from core.config import settings

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """The contract every provider implements."""

    provider: str

    @property
    def is_configured(self) -> bool: ...

    async def generate(self, system: str, user_message: str, **kwargs: Any): ...

    def stream(self, system: str, user_message: str, **kwargs: Any): ...

    def complete_short(self, prompt: str, **kwargs: Any) -> Optional[str]: ...

    async def health(self) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


_client: Optional[Any] = None
_resolution: str = ""
_verified: bool = False


async def resolve_verified(force: bool = False) -> tuple[Any, str]:
    """Pick a provider by *probing* it, not by checking a key exists.

    A configured-but-invalid key is indistinguishable from a valid one until you
    call the API. The first version of `auto` selected Anthropic because a key was
    present in the environment; that key returned 401 on every request, so the
    service degraded to extractive answers while a working Azure deployment sat
    unused right next to it.

    So at startup we health-check candidates in preference order and take the
    first that actually answers. One extra round-trip per boot removes a whole
    class of "why is it not using my LLM?" confusion.
    """
    global _client, _resolution, _verified
    if _verified and not force and _client is not None:
        return _client, _resolution

    from llm.azure_openai_client import AzureOpenAIClient
    from llm.claude_client import ClaudeClient

    provider = (settings.llm_provider or "auto").lower()

    # Explicit choice is respected without probing — the operator asked for it.
    if provider != "auto":
        _client, _resolution = _build()
        _verified = True
        logger.info("LLM provider: %s", _resolution)
        return _client, _resolution

    candidates: list[tuple[str, Any]] = []
    if settings.llm_configured:
        candidates.append(("anthropic", ClaudeClient()))
    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        candidates.append(("azure_openai", AzureOpenAIClient()))

    rejected: list[str] = []
    for name, candidate in candidates:
        try:
            health = await candidate.health()
        except Exception as exc:  # noqa: BLE001
            rejected.append(f"{name}: {type(exc).__name__}")
            continue

        if health.get("status") == "ok":
            suffix = f" (skipped {', '.join(rejected)})" if rejected else ""
            _client = candidate
            _resolution = f"auto → {name} verified live{suffix}"
            _verified = True
            logger.info("LLM provider: %s", _resolution)
            return _client, _resolution

        rejected.append(f"{name}: {str(health.get('detail', health.get('status')))[:70]}")
        await _safe_close(candidate)

    _client = candidates[0][1] if candidates else ClaudeClient()
    _resolution = (
        "auto → no provider passed a health check "
        f"({'; '.join(rejected) or 'none configured'}); extractive answers only"
    )
    _verified = True
    logger.warning("LLM provider: %s", _resolution)
    return _client, _resolution


async def _safe_close(client: Any) -> None:
    try:
        await client.aclose()
    except Exception:  # noqa: BLE001
        pass


def _build() -> tuple[Any, str]:
    provider = (settings.llm_provider or "auto").lower()

    from llm.azure_openai_client import AzureOpenAIClient
    from llm.claude_client import ClaudeClient

    anthropic_ready = settings.llm_configured
    azure_ready = bool(settings.azure_openai_api_key and settings.azure_openai_endpoint)

    if provider == "anthropic":
        return ClaudeClient(), "anthropic (explicit)"
    if provider in {"azure_openai", "azure", "openai"}:
        return AzureOpenAIClient(), "azure_openai (explicit)"

    # auto
    if anthropic_ready and azure_ready:
        return ClaudeClient(), "auto → anthropic (both configured; Anthropic preferred)"
    if anthropic_ready:
        return ClaudeClient(), "auto → anthropic (only provider configured)"
    if azure_ready:
        return AzureOpenAIClient(), "auto → azure_openai (only provider configured)"

    # Nothing configured: return the Anthropic client so `is_configured` is False
    # and callers take the extractive path. See llm/extractive.py.
    return ClaudeClient(), "auto → none configured (extractive answers only)"


def get_llm() -> Any:
    global _client, _resolution
    if _client is None:
        _client, _resolution = _build()
        logger.info("LLM provider: %s", _resolution)
    return _client


def provider_resolution() -> str:
    if not _resolution:
        get_llm()
    return _resolution


def reset_llm() -> None:
    """Drop the singleton (tests, or after a provider config change)."""
    global _client, _resolution, _verified
    _client, _resolution, _verified = None, "", False
