"""
Conversational behaviour test: small talk, context carry-over, reverse lookup.

Regression test for two reported failures:
  * "hey" / "hello" returned a candidate's date of birth instead of a greeting
  * "who is born on 14 October 1985" named a candidate born 7 September 1985
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

PDF = BACKEND_DIR.parent / "data" / "RAG_Test_Candidate_Profiles.pdf"

os.environ.setdefault("VECTOR_BACKEND", "local")
os.environ["LOCAL_INDEX_FILE"] = str(BACKEND_DIR / "data" / "conv_index.npz")
os.environ["COLLECTION_NAME"] = "conv_test"
os.environ.setdefault("RERANK_MODE", "fast")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from core.logging_config import setup_logging  # noqa: E402

setup_logging("WARNING")


async def main() -> int:
    from ingestion.service import get_ingest_service
    from llm.provider import resolve_verified
    from llm.rag_service import get_rag_service
    from retrieval.intent import Intent, classify
    from retrieval.pipeline import get_pipeline

    print("=" * 78)
    print("  CONVERSATIONAL BEHAVIOUR TEST")
    print("=" * 78)

    passes, failures = 0, []

    # ---- 1. intent classification (no models needed) ---------------------
    print(f"\n{'-' * 78}\n1. INTENT ROUTING\n{'-' * 78}")
    cases: list[tuple[str, Intent]] = [
        ("hey", Intent.GREETING),
        ("hello", Intent.GREETING),
        ("hi there", Intent.GREETING),
        ("Namaskaram", Intent.GREETING),
        ("good morning", Intent.GREETING),
        ("thanks", Intent.THANKS),
        ("thank you very much", Intent.THANKS),
        ("bye", Intent.FAREWELL),
        ("that's all", Intent.FAREWELL),
        ("ok", Intent.AFFIRM),
        ("got it", Intent.AFFIRM),
        ("who are you", Intent.IDENTITY),
        ("are you a bot?", Intent.IDENTITY),
        ("what can you do", Intent.CAPABILITY),
        ("how are you", Intent.CHITCHAT),
        # These MUST stay factual — a greeting prefix is not a greeting.
        ("hello, what does the manifesto say about roads?", Intent.FACTUAL),
        ("who is born on 14 October 1985", Intent.FACTUAL),
        ("I'm from Vijayawada", Intent.FACTUAL),
        ("what about schools there?", Intent.FACTUAL),
        ("Amma Vodi eligibility", Intent.FACTUAL),
    ]
    for utterance, expected in cases:
        got = classify(utterance).intent
        ok = got is expected
        passes += ok
        if not ok:
            failures.append(f"intent {utterance!r} → {got.value} (want {expected.value})")
        print(f"  [{'PASS' if ok else 'FAIL'}] {utterance[:44]:<46} → {got.value}")

    # ---- setup ------------------------------------------------------------
    pipeline = get_pipeline()
    pipeline.embedder.load()
    pipeline.store.ensure_collection(pipeline.embedder.dim, recreate=True)
    outcome = await get_ingest_service().ingest_file(PDF)
    if not outcome.ok:
        print(f"\ningest failed: {outcome.error}")
        return 1
    print(f"\nIndexed {len(outcome.chunks)} records")

    client, resolution = await resolve_verified()
    service = get_rag_service()
    service.llm = client
    print(f"LLM: {resolution}")

    # ---- 2. a real conversation ------------------------------------------
    print(f"\n{'-' * 78}\n2. MULTI-TURN CONVERSATION (one session)\n{'-' * 78}")
    session = "conv-demo"
    script = [
        ("hey", "greeting — must NOT be a factual answer"),
        ("who is born on 14 October 1985", "reverse lookup — the reported bug"),
        ("what are her assets?", "follow-up — must stay on the same person"),
        ("which constituency is she contesting from?", "follow-up again"),
        ("thanks", "thanks — must NOT retrieve"),
        ("hello", "mid-conversation greeting — short, not a re-introduction"),
    ]

    transcript: list[tuple[str, str]] = []
    for utterance, note in script:
        result = await service.answer(utterance, session_id=session)
        transcript.append((utterance, result.answer))
        print(f"\n  Citizen  : {utterance}")
        print(f"  Assistant: {' '.join(result.answer.split())[:200]}")
        print(f"  · {note}")
        print(f"  · model={result.model} grounded={result.grounded} "
              f"cites={[c.marker for c in result.citations]}")

    # ---- 3. assertions ---------------------------------------------------
    print(f"\n{'-' * 78}\n3. ASSERTIONS\n{'-' * 78}")

    greeting_answer = transcript[0][1].lower()
    ok = "born" not in greeting_answer and "assets" not in greeting_answer
    passes += ok
    if not ok:
        failures.append("'hey' produced a factual answer")
    print(f"  [{'PASS' if ok else 'FAIL'}] 'hey' is conversational, not factual")

    dob_answer = transcript[1][1]
    ok = "Kesineni" in dob_answer or "Jayasudha" in dob_answer
    passes += ok
    if not ok:
        failures.append(f"reverse lookup wrong: {dob_answer[:120]}")
    print(f"  [{'PASS' if ok else 'FAIL'}] reverse lookup names Dr. Jayasudha Kesineni")

    # No other candidate's DOB should appear in that answer.
    ok = "7 September 1985" not in dob_answer
    passes += ok
    if not ok:
        failures.append("reverse lookup leaked another candidate's DOB")
    print(f"  [{'PASS' if ok else 'FAIL'}] no other candidate's date of birth leaked")

    followup = transcript[2][1]
    ok = "76.4" in followup or "seventy-six" in followup.lower()
    passes += ok
    if not ok:
        failures.append(f"follow-up lost the subject: {followup[:120]}")
    print(f"  [{'PASS' if ok else 'FAIL'}] follow-up 'her assets' → Rs 76.4 lakh")

    seat = transcript[3][1]
    ok = "Adoni" in seat
    passes += ok
    if not ok:
        failures.append(f"second follow-up lost the subject: {seat[:120]}")
    print(f"  [{'PASS' if ok else 'FAIL'}] second follow-up → Adoni")

    thanks_answer = transcript[4][1].lower()
    ok = "assets" not in thanks_answer and "born" not in thanks_answer
    passes += ok
    if not ok:
        failures.append("'thanks' produced a factual answer")
    print(f"  [{'PASS' if ok else 'FAIL'}] 'thanks' is conversational")

    print(f"\n{'=' * 78}\n  checks passed : {passes}\n  failures      : {len(failures)}")
    for failure in failures:
        print(f"    - {failure}")
    print("\nVERDICT: PASS" if not failures else "\nVERDICT: FAIL")
    print("=" * 78)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
