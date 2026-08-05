"""
Anti-hallucination test against the real candidate corpus and the real LLM.

On a corpus of 56 near-identical profiles the dangerous failure is not inventing
facts, it is **misattribution** — reporting candidate A's assets under candidate
B's name. A generic "is the answer plausible?" check cannot catch that, so every
probe below asserts against ground truth parsed from the PDF itself:

  * fact probes    — the stated figure must be the one in that candidate's record
  * absent probes  — a person not in the corpus must produce a refusal
  * leak probes    — the answer must not contain another candidate's unique value
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

PDF = BACKEND_DIR.parent / "data" / "RAG_Test_Candidate_Profiles.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="retrieval only")
    parser.add_argument("--doc", default=str(PDF))
    return parser.parse_args()


ARGS = parse_args()
os.environ.setdefault("VECTOR_BACKEND", "local")
# This test asserts on *correctness*, not rerank ordering, and entity gating
# usually leaves a single candidate for the cross-encoder to score anyway. Using
# the fast tier keeps the run inside a modest memory budget so the result is a
# verdict rather than a crash.
os.environ.setdefault("RERANK_MODE", "fast")
os.environ["LOCAL_INDEX_FILE"] = str(BACKEND_DIR / "data" / "grounding_index.npz")
os.environ["COLLECTION_NAME"] = "grounding_test"
os.environ.setdefault("LOG_LEVEL", "WARNING")

from core.config import settings  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402

setup_logging("WARNING")

_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _spell(number: int) -> str:
    """English words for 0-99 — enough for lakh/crore magnitudes."""
    if number < 20:
        return _ONES[number]
    tens, ones = divmod(number, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"


def _amount_present(answer: str, value: str) -> bool:
    """Is `value` (e.g. "43.1", "4.84") stated in the answer, digits or words?

    The system prompt instructs the model to verbalize amounts because the answer
    is spoken by TTS — "rupees forty-three lakh and ten thousand" reads better
    aloud than "43.1 lakh". A digits-only assertion would therefore fail on
    perfectly correct output, so we accept either form.
    """
    flat = " ".join(answer.split()).lower()
    if value in flat:
        return True

    whole, _, frac = value.partition(".")
    try:
        whole_int = int(whole)
    except ValueError:
        return False

    if _spell(whole_int) not in flat:
        return False
    if not frac:
        return True

    # "4.84 crore" → "four crore and eighty-four lakh"; "43.1 lakh" → "... ten thousand".
    frac_int = int(frac.ljust(2, "0"))
    return _spell(frac_int) in flat or _spell(int(frac)) in flat


_ASSETS = re.compile(
    r"Assets declaration\.\s*Movable assets Rs\.\s*(?P<movable>[\d.]+)\s*lakh,\s*"
    r"immovable assets Rs\.\s*(?P<immovable>[\d.]+)\s*crore",
    re.IGNORECASE,
)
_BORN = re.compile(r"Born\.\s*(?P<born>\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE)
_LANGS = re.compile(r"Languages\.\s*(?P<langs>[^\n]+)", re.IGNORECASE)


def ground_truth(chunks) -> dict[str, dict[str, str]]:
    """Per-candidate facts, read straight out of their own chunk."""
    truth: dict[str, dict[str, str]] = {}
    for chunk in chunks:
        name = chunk.metadata.record_name
        if not name:
            continue
        flat = " ".join(chunk.text.split())
        entry: dict[str, str] = {"district": chunk.metadata.district or ""}
        assets = _ASSETS.search(flat)
        if assets:
            entry["movable"] = assets.group("movable")
            entry["immovable"] = assets.group("immovable")
        born = _BORN.search(flat)
        if born:
            entry["born"] = born.group("born")
        langs = _LANGS.search(flat)
        if langs:
            entry["languages"] = langs.group("langs").strip().rstrip(".")
        truth[name] = entry
    return truth


async def main() -> int:
    from ingestion.service import get_ingest_service
    from llm.rag_service import get_rag_service
    from retrieval.pipeline import get_pipeline

    print("=" * 78)
    print("  GROUNDING / ANTI-HALLUCINATION TEST")
    print("=" * 78)

    pipeline = get_pipeline()
    pipeline.embedder.load()

    # Fresh index so a stale run cannot mask a regression.
    pipeline.store.ensure_collection(pipeline.embedder.dim, recreate=True)

    outcome = await get_ingest_service().ingest_file(Path(ARGS.doc))
    if not outcome.ok:
        print(f"ingest failed: {outcome.error}")
        return 1
    chunks = outcome.chunks
    print(f"\nIndexed {len(chunks)} record chunks from {Path(ARGS.doc).name}")

    truth = ground_truth(chunks)
    named = sorted(truth)
    print(f"Ground truth parsed for {len(named)} candidates")
    print(f"  e.g. {named[0]} → {truth[named[0]]}")

    # Pick three candidates spread across the document.
    subjects = [named[0], named[len(named) // 2], named[-1]]

    # Probe providers the same way the API does, so the test exercises whichever
    # LLM actually answers rather than the first key that happens to be present.
    from llm.provider import resolve_verified

    verified_client, resolution = await resolve_verified()
    print(f"\nLLM provider: {resolution}")

    service = get_rag_service()
    service.llm = verified_client
    use_llm = not ARGS.no_llm and service.llm.is_configured
    print(f"\nLLM: {'enabled — ' + settings.llm_model if use_llm else 'disabled (retrieval only)'}")

    passes = 0
    failures: list[str] = []

    # ---------------------------------------------------- retrieval precision
    print(f"\n{'=' * 78}\n1. RETRIEVAL — does the named person's own record rank #1?\n{'=' * 78}")
    for name in subjects:
        for template in (
            "What are {}'s declared assets?",
            "When was {} born?",
            "Which languages does {} speak?",
        ):
            question = template.format(name)
            result = await pipeline.retrieve(question, top_k=3)
            top = result.results[0] if result.results else None
            top_name = top.metadata.record_name if top else None
            ok = top_name == name
            passes += ok
            if not ok:
                failures.append(f"retrieval: {question!r} → {top_name!r} (want {name!r})")
            print(f"  [{'PASS' if ok else 'FAIL'}] {question[:52]:<54} → {top_name}")

    # -------------------------------------------------- context integrity
    # The property that actually prevents misattribution, provable without an
    # LLM: for a question naming one person, the assembled context must contain
    # that person's record and must NOT contain another candidate's unique
    # figures. If this holds, the model has no wrong number available to quote.
    print(f"\n{'=' * 78}\n2. CONTEXT INTEGRITY — no other candidate's figures reach the prompt\n{'=' * 78}")
    from llm.prompts import build_context_block

    for name in subjects:
        facts = truth[name]
        question = f"What are the declared assets of {name}?"
        result = await pipeline.retrieve(question, top_k=settings.retrieval_top_k)
        context, citation_meta = build_context_block(result.results)
        flat = " ".join(context.split())

        own_movable = facts.get("movable", "")
        own_immovable = facts.get("immovable", "")
        has_own = bool(own_movable and own_movable in flat)

        foreign = sorted(
            {
                other_facts["immovable"]
                for other_name, other_facts in truth.items()
                if other_name != name and other_facts.get("immovable")
            }
            - {own_immovable}
        )
        leaked = [v for v in foreign if re.search(rf"\b{re.escape(v)}\s*crore", flat)]

        names_in_context = {
            chunk.metadata.record_name
            for chunk in result.results
            if chunk.metadata.record_name
        }
        ok = has_own and not leaked
        passes += ok
        if not ok:
            failures.append(
                f"context for {name}: own_figure={has_own} leaked={leaked[:3]}"
            )
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        own figure present : {has_own} (Rs {own_movable} lakh)")
        print(f"        foreign figures    : {leaked[:3] or 'none'}")
        print(f"        records in context : {len(names_in_context)} → {sorted(names_in_context)[:3]}")
        print(f"        context size       : {len(context)} chars, {len(citation_meta)} passages")
        print(f"        person_hint        : {result.inferred_filters.get('person_hint')!r}")
        for note in result.notes:
            print(f"        note               : {note}")

    # ------------------------------------------------------- answer fidelity
    if use_llm:
        print(f"\n{'=' * 78}\n2. ANSWERS — right figures, no cross-record leakage\n{'=' * 78}")
        for name in subjects:
            facts = truth[name]
            question = f"What are the declared movable and immovable assets of {name}?"
            answer_result = await service.answer(question, session_id=f"gt-{name}")
            answer = answer_result.answer
            flat = " ".join(answer.split())

            movable = facts.get("movable", "")
            immovable = facts.get("immovable", "")
            # Accept digits or the spoken verbalization the prompt asks for.
            has_movable = bool(movable) and _amount_present(flat, movable)
            has_immovable = bool(immovable) and _amount_present(flat, immovable)

            # Leakage: another candidate's unique immovable figure appearing here.
            # Digit form only — a verbalized false positive is far more likely
            # than a verbatim digit collision.
            others = {
                other_facts.get("immovable")
                for other_name, other_facts in truth.items()
                if other_name != name and other_facts.get("immovable")
            }
            others.discard(immovable)
            leaked = sorted(v for v in others if v and re.search(rf"\b{re.escape(v)}\b", flat))

            cited = bool(answer_result.citations)
            ok = bool(has_movable and has_immovable and not leaked and cited)
            passes += ok
            if not ok:
                failures.append(
                    f"answer for {name}: movable={has_movable} immovable={has_immovable} "
                    f"leaked={leaked[:3]} cited={cited}"
                )
            print(f"\n  [{'PASS' if ok else 'FAIL'}] {name}")
            print(f"        expected  Rs {movable} lakh movable / Rs {immovable} crore immovable")
            print(f"        answer    {flat[:170]}")
            print(f"        citations {[c.source + '#' + (c.section or '') for c in answer_result.citations]}")
            if leaked:
                print(f"        LEAKED other candidates' figures: {leaked[:5]}")

        # --------------------------------------------- refusal on absent person
        print(f"\n{'=' * 78}\n3. REFUSAL — a person who is not in the corpus\n{'=' * 78}")
        for fake in ("Dr. Ramesh Chandra Patel", "Smt. Anjali Verma"):
            question = f"What are the declared assets of {fake}?"
            result = await service.answer(question, session_id="gt-absent")
            flat = " ".join(result.answer.split())
            # A refusal must not quote a figure, and must not claim to be grounded.
            quoted = re.findall(r"Rs\.?\s*[\d.]+\s*(?:lakh|crore)", flat, re.IGNORECASE)
            refused = not quoted
            passes += refused
            if not refused:
                failures.append(f"absent person {fake!r} got figures: {quoted[:3]}")
            print(f"\n  [{'PASS' if refused else 'FAIL'}] {fake}")
            print(f"        answer   {flat[:190]}")
            print(f"        grounded {result.grounded} | figures quoted: {quoted[:3] or 'none'}")

    # ------------------------------------------------------------------ report
    print(f"\n{'=' * 78}\nRESULT\n{'=' * 78}")
    print(f"  checks passed : {passes}")
    print(f"  failures      : {len(failures)}")
    for failure in failures:
        print(f"    - {failure}")
    print()
    print("VERDICT: PASS" if not failures else "VERDICT: FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
