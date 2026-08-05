"""
Azure OpenAI chat client — a drop-in alternative to the Anthropic client.

Implements the same surface (`stream`, `generate`, `complete_short`, `health`,
`aclose`) so `RAGService` and the voice loop are provider-agnostic: switching
providers is one env var, not a code change.

## Implementation notes

**Raw httpx instead of the `openai` SDK.** The deployments here are addressed by
*full* chat-completions URLs with the api-version baked in. The SDK wants a base
endpoint plus deployment plus api-version and reassembles the URL itself, which
means feeding it these values requires taking them apart and hoping it puts them
back together identically. A direct POST to the URL we were given is less code,
one fewer dependency, and cannot mis-assemble anything.

**Two deployments, chosen by latency need.** GPT-4 follows the "never blend
records" instructions more reliably; GPT-3.5-turbo-16k answers noticeably faster.
Because entity gating already narrows the context to a single record, the
generation task is easy enough that the cheaper model is usually sufficient — so
voice turns use it by default and text turns use GPT-4. Both are configurable.

**No `stream_options` / `usage` in stream mode.** The 2024-02-15 api-version on
these deployments does not return usage during streaming, so token counts are
estimated from character length and flagged as such rather than silently reported
as exact.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, AsyncIterator, Optional

import httpx

from core.config import settings
from core.latency import METRICS
from core.schemas import TokenUsage
from llm.claude_client import GenerationResult

logger = logging.getLogger(__name__)

_TAG_LEAK = re.compile(
    r"</?(?:thinking|internal|scratchpad|reasoning)[^>]*>", re.IGNORECASE
)
_MARKDOWN_NOISE = re.compile(r"(\*\*|__|`{1,3}|^#{1,6}\s+)", re.MULTILINE)


class AzureOpenAIClient:
    """Azure OpenAI chat completions over the deployment URLs from config."""

    provider = "azure_openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        fast_endpoint: Optional[str] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.azure_openai_api_key
        self.endpoint = endpoint or settings.azure_openai_endpoint
        self.fast_endpoint = fast_endpoint or settings.azure_openai_fast_endpoint or self.endpoint
        self.model = settings.azure_openai_deployment
        self.fast_model = settings.azure_openai_fast_deployment or self.model
        self._client: Optional[httpx.AsyncClient] = None

    # ---------------------------------------------------------------- plumbing
    @property
    def aclient(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.llm_timeout_s, connect=10.0),
                headers={"api-key": self.api_key, "content-type": "application/json"},
            )
        return self._client

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.endpoint)

    def _target(self, voice_mode: bool) -> tuple[str, str]:
        """(url, deployment) — the fast deployment for latency-bound turns."""
        if voice_mode and self.fast_endpoint:
            return self.fast_endpoint, self.fast_model
        return self.endpoint, self.model

    @staticmethod
    def _payload(
        system: str,
        user_message: str,
        history: Optional[list[dict[str, str]]],
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for turn in history or []:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})

        return {
            "messages": messages,
            "max_tokens": max_tokens,
            # Deterministic: this is grounded extraction, not creative writing.
            # Any sampling here is a chance to drift a rupee figure.
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": stream,
        }

    # ---------------------------------------------------------------- generate
    async def generate(
        self,
        system: str,
        user_message: str,
        *,
        history: Optional[list[dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        effort: Optional[str] = None,
    ) -> GenerationResult:
        chunks: list[str] = []
        result: Optional[GenerationResult] = None
        async for event in self.stream(
            system,
            user_message,
            history=history,
            max_tokens=max_tokens,
            model=model,
            effort=effort,
        ):
            if event["type"] == "text":
                chunks.append(event["text"])
            elif event["type"] == "done":
                result = event["result"]
            elif event["type"] == "error":
                return GenerationResult(
                    text="", notes=[event["error"]], failed=True, model=self.model
                )
        if result is None:
            return GenerationResult(
                text=self._sanitize("".join(chunks)),
                notes=["stream ended without completion"],
                failed=not chunks,
                model=self.model,
            )
        return result

    async def stream(
        self,
        system: str,
        user_message: str,
        *,
        history: Optional[list[dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        effort: Optional[str] = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if not self.is_configured:
            yield {
                "type": "error",
                "error": "AZURE_OPENAI_API_KEY / endpoint are not configured",
                "recoverable": False,
            }
            return

        # `effort` is the Anthropic latency lever; here the equivalent is picking
        # the smaller deployment, so we reuse it as the voice-mode signal.
        voice_mode = effort == "low"
        url, deployment = self._target(voice_mode)
        payload = self._payload(
            system, user_message, history, max_tokens or settings.llm_max_tokens, stream=True
        )

        start = time.perf_counter()
        first_token_ms: Optional[float] = None
        collected: list[str] = []
        finish_reason: Optional[str] = None

        try:
            async with self.aclient.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise RuntimeError(f"HTTP {response.status_code}: {body[:300]}")

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if text:
                            if first_token_ms is None:
                                first_token_ms = (time.perf_counter() - start) * 1000
                                METRICS.observe("llm.first_token", first_token_ms)
                            collected.append(text)
                            yield {"type": "text", "text": text}
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

        except Exception as exc:  # noqa: BLE001
            total = (time.perf_counter() - start) * 1000
            METRICS.incr("llm.error")
            logger.error("Azure OpenAI stream failed after %.0fms: %s", total, exc)
            yield {"type": "error", "error": str(exc), "recoverable": True}
            return

        total_ms = (time.perf_counter() - start) * 1000
        METRICS.observe("llm.total", total_ms)

        text = self._sanitize("".join(collected))
        notes: list[str] = []
        if finish_reason == "length":
            notes.append("truncated at max_tokens")
        if finish_reason == "content_filter":
            notes.append("Azure content filter trimmed the response")

        yield {
            "type": "done",
            "result": GenerationResult(
                text=text,
                # Streaming on this api-version returns no usage block; estimate
                # rather than report zeros that look like a metrics bug.
                usage=TokenUsage(
                    input_tokens=len(system + user_message) // 4,
                    output_tokens=len(text) // 4,
                ),
                model=deployment,
                stop_reason=finish_reason,
                first_token_ms=round(first_token_ms, 2) if first_token_ms else None,
                total_ms=round(total_ms, 2),
                notes=[*notes, "usage estimated (not returned when streaming)"],
                failed=not text,
            ),
        }

    # ------------------------------------------------------------ small calls
    async def acomplete_short(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 100,
    ) -> Optional[str]:
        if not self.is_configured:
            return None
        # Always the fast deployment — this is the query rewriter, on the hot path.
        url = self.fast_endpoint or self.endpoint
        payload = self._payload("You rewrite search queries.", prompt, None, max_tokens, stream=False)
        try:
            response = await self.aclient.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return (data["choices"][0]["message"]["content"] or "").strip() or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("acomplete_short failed: %s", exc)
            return None

    def complete_short(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 100,
    ) -> Optional[str]:
        """Blocking variant used by the rule-based query rewriter."""
        if not self.is_configured:
            return None
        url = self.fast_endpoint or self.endpoint
        payload = self._payload("You rewrite search queries.", prompt, None, max_tokens, stream=False)
        try:
            with httpx.Client(
                timeout=httpx.Timeout(15.0, connect=10.0),
                headers={"api-key": self.api_key, "content-type": "application/json"},
            ) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return (data["choices"][0]["message"]["content"] or "").strip() or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("complete_short failed: %s", exc)
            return None

    # -------------------------------------------------------------- internals
    @staticmethod
    def _sanitize(text: str) -> str:
        text = _TAG_LEAK.sub("", text)
        text = _MARKDOWN_NOISE.sub("", text)
        return re.sub(r"[ \t]{2,}", " ", text).strip()

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Azure client close failed: %s", exc)

    def close(self) -> None:
        return None

    # ------------------------------------------------------------------ health
    async def health(self) -> dict[str, Any]:
        if not self.is_configured:
            return {
                "status": "disabled",
                "provider": self.provider,
                "model": self.model,
                "detail": "AZURE_OPENAI_API_KEY / endpoint not set",
            }
        start = time.perf_counter()
        try:
            payload = self._payload("You are a health check.", "ping", None, 4, stream=False)
            response = await self.aclient.post(self.endpoint, json=payload)
            if response.status_code >= 400:
                return {
                    "status": "down",
                    "provider": self.provider,
                    "model": self.model,
                    "detail": f"HTTP {response.status_code}: {response.text[:180]}",
                }
            return {
                "status": "ok",
                "provider": self.provider,
                "model": self.model,
                "fast_model": self.fast_model,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "down",
                "provider": self.provider,
                "model": self.model,
                "detail": str(exc),
            }
