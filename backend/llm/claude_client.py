"""
Claude client — streaming generation with prompt caching.

Model / parameter choices, and why:

* **`claude-opus-5`** is the default. The generation step here is grounded
  extraction and rephrasing over a small context, not open-ended reasoning, so
  the model rarely needs to be the bottleneck — but Opus's instruction-following
  is what keeps it inside the grounding rules under adversarial questions.
* **Thinking disabled at `effort: "low"`.** Time-to-first-token *is* the product
  in a voice turn; a thinking block would add hundreds of milliseconds of silence
  before the first audible word. Disabling thinking is only permitted at effort
  `high` or below on Opus 5, which `low` satisfies.
  Two known consequences of thinking-off are handled explicitly: the system prompt
  contains a generic "no internal or system XML tags" instruction (naming
  `<thinking>` specifically is measurably less effective), and it deliberately
  contains *no* "don't reason" instruction, which makes tag leakage worse. We
  also strip any leaked tags defensively in `_sanitize`.
* **Prompt caching on the system prompt.** The system prompt is byte-identical
  across turns, so it caches at ~0.1x on reads. `usage.cache_read_input_tokens`
  is surfaced on every response so a zero there is visible rather than silent.
* **No `temperature` / `top_p`.** Both are rejected with a 400 on Opus 5. Output
  style is controlled by the prompt instead.
* **Streaming everywhere.** Even for non-streaming callers we stream and collect,
  which sidesteps SDK HTTP timeouts and gets us the same usage numbers.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, AsyncIterator, Optional

from core.config import settings
from core.latency import METRICS
from core.schemas import TokenUsage

logger = logging.getLogger(__name__)

_TAG_LEAK = re.compile(
    r"</?(?:thinking|antml:thinking|internal|scratchpad|reasoning)[^>]*>",
    re.IGNORECASE,
)
_MARKDOWN_NOISE = re.compile(r"(\*\*|__|`{1,3}|^#{1,6}\s+)", re.MULTILINE)


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class GenerationResult:
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    stop_reason: Optional[str] = None
    refusal: bool = False
    first_token_ms: Optional[float] = None
    total_ms: float = 0.0
    notes: list[str] = field(default_factory=list)
    # True when the call itself failed (auth, rate limit, network) as opposed to
    # succeeding with an empty or refused response. Callers use this to decide
    # whether to fall back to extractive answering.
    failed: bool = False


@lru_cache(maxsize=1)
def _supported_stream_params() -> frozenset[str]:
    """Kwargs the installed SDK's `messages.stream()` actually accepts.

    `output_config` (effort) and `thinking` are the two levers that keep
    time-to-first-token low, and both are recent additions. An older SDK raises
    `TypeError: unexpected keyword argument` at call time — which surfaces as a
    total generation failure, not a degraded one. Introspecting once at import
    lets us drop unsupported kwargs and log it, so the system answers (a little
    slower) instead of erroring.
    """
    try:
        import inspect

        from anthropic.resources.messages import AsyncMessages

        params = inspect.signature(AsyncMessages.stream).parameters
        # **kwargs in the signature means anything goes.
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return frozenset({"*"})
        return frozenset(params)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not introspect the Anthropic SDK (%s); sending all params", exc)
        return frozenset({"*"})


class ClaudeClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = (
            api_key if api_key is not None else settings.resolved_anthropic_key
        )
        self.model = model or settings.llm_model
        self._client = None
        self._async_client = None
        self._unavailable_reason = ""

    # ----------------------------------------------------------------- client
    def _build_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "timeout": settings.llm_timeout_s,
            "max_retries": settings.llm_max_retries,
        }
        # An empty api_key would shadow an `ant auth login` profile or
        # ANTHROPIC_AUTH_TOKEN, so only pass it when we actually have one.
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs

    @property
    def sync(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(**self._build_kwargs())
        return self._client

    @property
    def aclient(self):
        if self._async_client is None:
            import anthropic

            self._async_client = anthropic.AsyncAnthropic(**self._build_kwargs())
        return self._async_client

    @property
    def is_configured(self) -> bool:
        return settings.llm_configured

    # ------------------------------------------------------------- parameters
    def _request_params(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        effort: Optional[str] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "messages": messages,
        }

        # cache_control marks the end of the stable prefix. Everything volatile
        # (context, question, history) is in `messages`, i.e. after this point.
        if settings.llm_enable_prompt_cache:
            params["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            params["system"] = system

        chosen_effort = effort or settings.llm_effort
        params["output_config"] = {"effort": chosen_effort}

        if settings.llm_thinking == "off":
            # `disabled` is rejected above effort `high`; guard so a config change
            # cannot turn into a 400 at request time.
            if chosen_effort in {"low", "medium", "high"}:
                params["thinking"] = {"type": "disabled"}
            else:
                params["thinking"] = {"type": "adaptive"}
        else:
            params["thinking"] = {"type": "adaptive"}

        # Drop anything this SDK build does not accept rather than letting it
        # raise TypeError and fail the whole turn.
        supported = _supported_stream_params()
        if "*" not in supported:
            for key in ("output_config", "thinking"):
                if key in params and key not in supported:
                    params.pop(key)
                    logger.warning(
                        "Anthropic SDK does not support '%s'; dropping it. "
                        "Upgrade to anthropic>=0.120.0 for latency control.",
                        key,
                    )

        return params

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
        """Non-streaming interface (implemented over the stream)."""
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

        if result is None:
            raise LLMUnavailable("stream ended without a completion event")
        if not result.text:
            result.text = self._sanitize("".join(chunks))
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
        """Yield `{"type": "text"|"done"|"error", ...}` events.

        Text deltas are emitted as they arrive so the caller can start TTS on the
        first complete sentence rather than waiting for the full answer.
        """
        if not self.is_configured:
            yield {
                "type": "error",
                "error": "ANTHROPIC_API_KEY is not configured",
                "recoverable": False,
            }
            return

        messages: list[dict[str, Any]] = []
        for turn in history or []:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})

        params = self._request_params(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            model=model,
            effort=effort,
        )

        start = time.perf_counter()
        first_token_ms: Optional[float] = None
        collected: list[str] = []

        try:
            async with self.aclient.messages.stream(**params) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - start) * 1000
                            METRICS.observe("llm.first_token", first_token_ms)
                        text = event.delta.text
                        collected.append(text)
                        yield {"type": "text", "text": text}

                final = await stream.get_final_message()

        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            total = (time.perf_counter() - start) * 1000
            METRICS.incr("llm.error")
            logger.error("Claude stream failed after %.0fms: %s", total, exc, exc_info=True)
            yield {"type": "error", "error": str(exc), "recoverable": True}
            return

        total_ms = (time.perf_counter() - start) * 1000
        METRICS.observe("llm.total", total_ms)

        text = self._extract_text(final) or "".join(collected)
        usage = self._usage(final)
        refusal = getattr(final, "stop_reason", None) == "refusal"

        notes: list[str] = []
        if refusal:
            # Opus 5 can decline via a safety classifier; surface it rather than
            # returning an empty answer that looks like a retrieval failure.
            details = getattr(final, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            notes.append(f"model declined (category={category})")
            logger.warning("Claude returned stop_reason=refusal (category=%s)", category)
        if getattr(final, "stop_reason", None) == "max_tokens":
            notes.append("truncated at max_tokens")

        if settings.llm_enable_prompt_cache and usage.cache_read_input_tokens == 0:
            # Expected exactly once per cold prefix; a persistent zero means an
            # invalidator crept into the system prompt.
            METRICS.incr("llm.cache_cold")

        yield {
            "type": "done",
            "result": GenerationResult(
                text=self._sanitize(text),
                usage=usage,
                model=getattr(final, "model", params["model"]),
                stop_reason=getattr(final, "stop_reason", None),
                refusal=refusal,
                first_token_ms=round(first_token_ms, 2) if first_token_ms else None,
                total_ms=round(total_ms, 2),
                notes=notes,
            ),
        }

    # ------------------------------------------------------------ small calls
    def complete_short(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 100,
    ) -> Optional[str]:
        """Blocking one-shot for the query rewriter (Haiku-class, no streaming)."""
        if not self.is_configured:
            return None
        try:
            response = self.sync.messages.create(
                model=model or settings.rewrite_model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if getattr(response, "stop_reason", None) == "refusal":
                return None
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    return block.text.strip()
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("complete_short failed: %s", exc)
            return None

    async def acomplete_short(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 100,
    ) -> Optional[str]:
        if not self.is_configured:
            return None
        try:
            response = await self.aclient.messages.create(
                model=model or settings.rewrite_model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if getattr(response, "stop_reason", None) == "refusal":
                return None
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    return block.text.strip()
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("acomplete_short failed: %s", exc)
            return None

    # -------------------------------------------------------------- internals
    @staticmethod
    def _extract_text(message: Any) -> str:
        parts: list[str] = []
        for block in getattr(message, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)

    @staticmethod
    def _usage(message: Any) -> TokenUsage:
        usage = getattr(message, "usage", None)
        if usage is None:
            return TokenUsage()
        return TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

    @staticmethod
    def _sanitize(text: str) -> str:
        """Defensive cleanup for TTS-bound text."""
        text = _TAG_LEAK.sub("", text)
        text = _MARKDOWN_NOISE.sub("", text)
        return re.sub(r"[ \t]{2,}", " ", text).strip()

    # ----------------------------------------------------------------- cleanup
    async def aclose(self) -> None:
        """Close the async HTTP pool.

        Without this the httpx transport is finalised by the garbage collector
        after the event loop has already closed, producing
        `RuntimeError: Event loop is closed` on shutdown — noisy, and it can mask
        a real error in the same traceback.
        """
        client, self._async_client = self._async_client, None
        if client is not None:
            try:
                await client.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Async client close failed: %s", exc)

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Sync client close failed: %s", exc)

    # ------------------------------------------------------------------ health
    async def health(self) -> dict[str, Any]:
        if not self.is_configured:
            return {
                "status": "disabled",
                "model": self.model,
                "detail": "ANTHROPIC_API_KEY not set",
            }
        start = time.perf_counter()
        try:
            # count_tokens is the cheapest authenticated round-trip available.
            await self.aclient.messages.count_tokens(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
            )
            return {
                "status": "ok",
                "model": self.model,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                "effort": settings.llm_effort,
                "thinking": settings.llm_thinking,
                "prompt_cache": settings.llm_enable_prompt_cache,
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "down", "model": self.model, "detail": str(exc)}


_client: Optional[ClaudeClient] = None


def get_llm() -> ClaudeClient:
    global _client
    if _client is None:
        _client = ClaudeClient()
    return _client
