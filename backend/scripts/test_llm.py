"""
Which LLM provider/deployment actually works, and how fast?

Probes every configured provider and deployment, reports health, measures
time-to-first-token and total latency, and checks the answer is grounded in a
supplied record. Time-to-first-token is the number that matters for a voice turn —
total latency only decides when the sentence *ends*, TTFT decides when speech
starts.

    python scripts/test_llm.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from core.config import settings  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402

setup_logging("ERROR")

SYSTEM = (
    "You answer strictly from the CONTEXT. Never invent figures. Cite with [1]. "
    "Reply in one or two plain spoken sentences, no markdown."
)
CONTEXT = """CONTEXT
[1] (source: RAG_Test_Candidate_Profiles.pdf | THIS PASSAGE IS ONLY ABOUT: Smt. Sarojini Vasireddy)
Smt. Sarojini Vasireddy is the Alliance candidate for the Srikakulam assembly constituency.
Born. 12 March 1971, in Srikakulam.
Assets declaration. Movable assets Rs. 23.7 lakh, immovable assets Rs. 2.11 crore, liabilities Rs. 9.4 lakh.
Languages. Telugu, English.

CITIZEN'S QUESTION
What are the declared assets of Smt. Sarojini Vasireddy?"""

EXPECT = ("23.7", "2.11")


async def probe(label: str, client, *, effort: str | None) -> dict:
    result: dict = {"label": label, "ok": False}

    health = await client.health()
    result["health"] = health.get("status")
    result["detail"] = health.get("detail", "")[:110]
    if health.get("status") != "ok":
        return result

    start = time.perf_counter()
    ttft = None
    chunks: list[str] = []
    error = None

    async for event in client.stream(SYSTEM, CONTEXT, max_tokens=160, effort=effort):
        if event["type"] == "text":
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            chunks.append(event["text"])
        elif event["type"] == "error":
            error = event["error"]
            break

    total = (time.perf_counter() - start) * 1000
    answer = "".join(chunks).strip()

    result.update(
        ok=bool(answer) and error is None,
        error=error,
        ttft_ms=round(ttft, 1) if ttft else None,
        total_ms=round(total, 1),
        answer=answer,
        grounded=all(token in answer for token in EXPECT),
        cited="[1]" in answer,
    )
    return result


async def main() -> int:
    from llm.azure_openai_client import AzureOpenAIClient
    from llm.claude_client import ClaudeClient
    from llm.provider import provider_resolution

    print("=" * 78)
    print("  LLM PROVIDER PROBE")
    print("=" * 78)
    print(f"  LLM_PROVIDER   : {settings.llm_provider}")
    print(f"  resolution     : {provider_resolution()}")
    print(f"  anthropic key  : {'set' if settings.resolved_anthropic_key else 'not set'}")
    print(f"  azure key      : {'set' if settings.azure_openai_api_key else 'not set'}")

    candidates: list[tuple[str, object, str | None]] = []

    if settings.resolved_anthropic_key:
        candidates.append((f"anthropic / {settings.llm_model}", ClaudeClient(), "low"))

    if settings.azure_openai_api_key and settings.azure_openai_endpoint:
        gpt4 = AzureOpenAIClient()
        candidates.append((f"azure / {settings.azure_openai_deployment}", gpt4, None))

        if settings.azure_openai_fast_endpoint:
            fast = AzureOpenAIClient()
            # effort="low" routes to the fast deployment inside the client.
            candidates.append(
                (f"azure / {settings.azure_openai_fast_deployment}", fast, "low")
            )

    if not candidates:
        print("\n  No provider credentials configured.")
        return 1

    results = []
    for label, client, effort in candidates:
        print(f"\n{'-' * 78}\n  {label}\n{'-' * 78}")
        try:
            outcome = await probe(label, client, effort=effort)
        except Exception as exc:  # noqa: BLE001
            print(f"    EXCEPTION : {type(exc).__name__}: {exc}")
            results.append({"label": label, "ok": False, "error": str(exc)})
            continue
        finally:
            await client.aclose()  # type: ignore[attr-defined]

        results.append(outcome)
        print(f"    health    : {outcome['health']} {outcome.get('detail', '')}")
        if not outcome["ok"]:
            print(f"    RESULT    : UNUSABLE — {outcome.get('error') or outcome.get('detail')}")
            continue
        print(f"    ttft      : {outcome['ttft_ms']} ms   <-- when speech can start")
        print(f"    total     : {outcome['total_ms']} ms")
        print(f"    grounded  : {outcome['grounded']} (both figures present)")
        print(f"    cited     : {outcome['cited']}")
        print(f"    answer    : {outcome['answer'][:200]}")

    working = [r for r in results if r.get("ok") and r.get("grounded")]

    print(f"\n{'=' * 78}\n  VERDICT\n{'=' * 78}")
    if not working:
        print("  No provider produced a grounded answer.")
        for r in results:
            print(f"    ✗ {r['label']}: {r.get('error') or r.get('detail') or 'ungrounded'}")
        return 1

    working.sort(key=lambda r: r["ttft_ms"] or 1e9)
    print("  Usable, fastest first (by time-to-first-token):")
    for r in working:
        print(f"    ok  {r['label']:<42} ttft={r['ttft_ms']:>7.0f} ms  total={r['total_ms']:>7.0f} ms")
    print(f"\n  RECOMMENDED for voice turns : {working[0]['label']}")
    print(f"  RECOMMENDED for text turns  : {working[-1]['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
