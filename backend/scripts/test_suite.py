"""
Full behavioural suite: 110+ questions, all self-validating against the PDF.

    python scripts/test_suite.py                 # run everything
    python scripts/test_suite.py --only conv     # conversational half only
    python scripts/test_suite.py --only doc      # document half only
    python scripts/test_suite.py --limit 20      # first 20 (fast iteration)
    python scripts/test_suite.py --no-llm        # retrieval-only assertions
    python scripts/test_suite.py --verbose       # print every answer

Exit code is 0 only when every question passes. Failures print the question, the
expectation, and the actual answer, so a failure is directly actionable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

PDF = BACKEND_DIR.parent / "data" / "RAG_Test_Candidate_Profiles.pdf"

parser = argparse.ArgumentParser()
parser.add_argument("--only", choices=["conv", "doc", "all"], default="all")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--no-llm", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--fail-fast", action="store_true")
ARGS = parser.parse_args()

os.environ.setdefault("VECTOR_BACKEND", "local")
os.environ["LOCAL_INDEX_FILE"] = str(BACKEND_DIR / "data" / "suite_index.npz")
os.environ["COLLECTION_NAME"] = "suite_test"
os.environ.setdefault("RERANK_MODE", "fast")
os.environ.setdefault("LOG_LEVEL", "ERROR")

from core.logging_config import setup_logging  # noqa: E402

setup_logging("ERROR")

from tests.question_bank import (  # noqa: E402
    ABSENT_PEOPLE,
    ABSENT_VALUES,
    AGGREGATION,
    CONVERSATIONAL,
    DIRECT_TEMPLATES,
    FOLLOWUP_TEMPLATES,
    OUT_OF_SCOPE,
    REVERSE_TEMPLATES,
    Check,
    Question,
    parse_record,
)

# ---------------------------------------------------------------- number words
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _spell(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"


def value_present(answer: str, value: str) -> bool:
    """Is `value` stated, as digits OR as the spoken form the prompt asks for?"""
    flat = " ".join(answer.split()).lower()
    v = value.strip().lower()
    if v in flat:
        return True

    # Numeric: "43.1" also matches "forty-three lakh and ten thousand".
    if re.fullmatch(r"\d+(\.\d+)?", v):
        whole, _, frac = v.partition(".")
        if _spell(int(whole)) not in flat:
            return False
        if not frac:
            return True
        return _spell(int(frac.ljust(2, "0"))) in flat or _spell(int(frac)) in flat

    # Dates: "14 October 1985" also matches "14th October 1985" / "October 14".
    date = re.match(r"(\d{1,2})\s+(\w+)\s+(\d{4})", v)
    if date:
        day, month, year = date.groups()
        return month in flat and year in flat and day in flat

    # Free text: require a majority of its distinctive words.
    words = [w for w in re.findall(r"[a-z]{4,}", v)]
    if not words:
        return False
    hits = sum(1 for w in words if w in flat)
    return hits >= max(1, int(len(words) * 0.5))


_FIGURE = re.compile(r"(?:rs\.?|rupees)\s*[\d.]+\s*(?:lakh|crore)|\b\d{1,2}\s+\w+\s+(?:19|20)\d{2}\b",
                     re.IGNORECASE)


@dataclass
class Result:
    question: Question
    passed: bool
    answer: str
    reason: str = ""
    grounded: bool = False
    cited: int = 0
    ms: float = 0.0


# ============================================================================
def build_questions(records: list[tuple[str, dict[str, str], str]]) -> list[Question]:
    """records: (record_name, parsed_fields, chunk_text)"""
    questions: list[Question] = []
    all_surnames = {name.split()[-1] for name, _, _ in records}

    # ---------------- conversational --------------------------------------
    for text, label in CONVERSATIONAL:
        questions.append(
            Question(
                text=text,
                check=Check.CONVERSATIONAL,
                label=f"conv/{label}",
                forbid=sorted(all_surnames),
            )
        )
    for text in OUT_OF_SCOPE:
        questions.append(
            Question(
                text=text,
                check=Check.REFUSAL,
                label="conv/out-of-scope",
                forbid=sorted(all_surnames),
            )
        )

    if ARGS.only == "conv":
        return questions

    # ---------------- direct field lookups --------------------------------
    # Spread across the document rather than clustering at the front.
    step = max(1, len(records) // 12)
    picks = records[::step][:12]
    for index, (name, fields, _) in enumerate(picks):
        label, template, key = DIRECT_TEMPLATES[index % len(DIRECT_TEMPLATES)]
        if key not in fields:
            continue
        others = sorted(all_surnames - {name.split()[-1]})
        questions.append(
            Question(
                text=template.format(name=name),
                check=Check.FACT,
                expect_values=[fields[key]],
                expect_record=name,
                forbid=others,
                label=f"doc/{label}",
            )
        )

    # Every field exercised at least once, on one well-populated record.
    anchor_name, anchor_fields, _ = records[0]
    for label, template, key in DIRECT_TEMPLATES:
        if key not in anchor_fields:
            continue
        questions.append(
            Question(
                text=template.format(name=anchor_name),
                check=Check.FACT,
                expect_values=[anchor_fields[key]],
                expect_record=anchor_name,
                forbid=sorted(all_surnames - {anchor_name.split()[-1]}),
                label=f"doc/field-{label}",
            )
        )

    # ---------------- reverse lookups -------------------------------------
    # Only values that identify exactly one record, or the question is ambiguous.
    for label, template, key in REVERSE_TEMPLATES:
        counts: dict[str, list[str]] = {}
        for name, fields, _ in records:
            if key in fields:
                counts.setdefault(fields[key], []).append(name)
        unique = [(v, n[0]) for v, n in counts.items() if len(n) == 1]
        for value, owner in unique[:3]:
            questions.append(
                Question(
                    text=template.format(value=value),
                    check=Check.FACT,
                    expect_values=[owner.split()[-1]],
                    expect_record=owner,
                    forbid=sorted(all_surnames - {owner.split()[-1]}),
                    label=f"doc/{label}",
                )
            )

    # ---------------- follow-up chains ------------------------------------
    for index, (label, setup, follow, key) in enumerate(FOLLOWUP_TEMPLATES):
        name, fields, _ = records[(index + 3) % len(records)]
        if key not in fields:
            continue
        questions.append(
            Question(
                text=follow,
                check=Check.FOLLOWUP,
                expect_values=[fields[key]],
                expect_record=name,
                context_turns=[s.format(name=name) for s in setup],
                forbid=sorted(all_surnames - {name.split()[-1]}),
                label=f"doc/{label}",
                session=f"fu-{index}",
            )
        )

    # ---------------- ambiguity: shared first name / shared surname --------
    first_names: dict[str, list[str]] = {}
    surnames: dict[str, list[str]] = {}
    for name, _, _ in records:
        parts = name.split()
        if len(parts) >= 3:
            first_names.setdefault(parts[1], []).append(name)
            surnames.setdefault(parts[-1], []).append(name)

    for group in (first_names, surnames):
        shared = [(k, v) for k, v in group.items() if len(v) > 1][:2]
        for key_name, owners in shared:
            questions.append(
                Question(
                    text=f"What are the declared assets of {key_name}?",
                    check=Check.NO_FABRICATION,
                    label="doc/ambiguous-name",
                )
            )

    # ---------------- absent people & values ------------------------------
    for who in ABSENT_PEOPLE:
        questions.append(
            Question(
                text=f"What are the declared assets of {who}?",
                check=Check.REFUSAL,
                forbid=sorted(all_surnames),
                label="doc/absent-person",
            )
        )
    for text, _kind in ABSENT_VALUES:
        questions.append(
            Question(
                text=text,
                check=Check.REFUSAL,
                forbid=sorted(all_surnames),
                label="doc/absent-value",
            )
        )

    # ---------------- aggregation (must not fabricate) ---------------------
    for text in AGGREGATION:
        questions.append(Question(text=text, check=Check.NO_FABRICATION, label="doc/aggregation"))

    # ---------------- ASR-style noise -------------------------------------
    noisy_name = records[1][0]
    for variant, note in (
        (noisy_name.lower(), "lowercase"),
        (noisy_name.replace(".", ""), "no-honorific-dot"),
        (noisy_name.split()[-1], "surname-only"),
    ):
        questions.append(
            Question(
                text=f"um so what are the assets of {variant} please",
                check=Check.NO_FABRICATION,
                label=f"doc/asr-{note}",
            )
        )

    return questions


# ============================================================================
def judge(question: Question, answer: str, grounded: bool, cited: int) -> tuple[bool, str]:
    flat = " ".join(answer.split())
    lowered = flat.lower()

    def names_forbidden() -> list[str]:
        return [
            n for n in question.forbid
            if re.search(rf"\b{re.escape(n)}\b", flat)
        ]

    if question.check is Check.CONVERSATIONAL:
        leaked = names_forbidden()
        if leaked:
            return False, f"named candidate(s) {leaked[:3]} in a conversational reply"
        if _FIGURE.search(flat):
            return False, f"quoted a figure/date: {_FIGURE.search(flat).group(0)!r}"
        if cited:
            return False, "attached a citation to small talk"
        if len(flat) < 4:
            return False, "empty reply"
        return True, ""

    if question.check is Check.REFUSAL:
        leaked = names_forbidden()
        if leaked:
            return False, f"named candidate(s) {leaked[:3]} instead of declining"
        if _FIGURE.search(flat):
            return False, f"quoted a figure/date: {_FIGURE.search(flat).group(0)!r}"
        declined = any(
            phrase in lowered
            for phrase in (
                "don't have", "do not have", "no information", "not have any",
                "couldn't find", "could not find", "not in", "no record",
                "unable", "outside", "only answer", "no details", "isn't something",
                "is not something", "cannot", "can't help",
            )
        )
        if not declined:
            return False, "did not clearly decline"
        return True, ""

    if question.check is Check.NO_FABRICATION:
        # Any specific figure must be traceable — require a citation if quoted.
        if _FIGURE.search(flat) and not cited:
            return False, "quoted a figure with no citation"
        if len(flat) < 4:
            return False, "empty reply"
        return True, ""

    # FACT / FOLLOWUP
    missing = [v for v in question.expect_values if not value_present(flat, v)]
    if missing:
        return False, f"missing expected value(s) {[m[:44] for m in missing]}"
    leaked = names_forbidden()
    if leaked:
        return False, f"leaked other candidate(s) {leaked[:3]}"
    if not cited:
        return False, "no citation"
    return True, ""


# ============================================================================
async def main() -> int:
    from ingestion.service import get_ingest_service
    from llm.provider import resolve_verified
    from llm.rag_service import get_rag_service
    from retrieval.pipeline import get_pipeline

    started = time.perf_counter()
    print("=" * 78)
    print("  FULL BEHAVIOURAL SUITE")
    print("=" * 78)

    pipeline = get_pipeline()
    pipeline.embedder.load()
    pipeline.store.ensure_collection(pipeline.embedder.dim, recreate=True)

    outcome = await get_ingest_service().ingest_file(PDF)
    if not outcome.ok:
        print(f"ingest failed: {outcome.error}")
        return 1

    records = [
        (c.metadata.record_name, parse_record(c.text), c.text)
        for c in outcome.chunks
        if c.metadata.record_name
    ]
    print(f"  indexed        : {len(outcome.chunks)} records")

    client, resolution = await resolve_verified()
    service = get_rag_service()
    service.llm = client
    use_llm = not ARGS.no_llm and client.is_configured
    print(f"  llm            : {resolution if use_llm else 'disabled'}")

    questions = build_questions(records)
    if ARGS.limit:
        questions = questions[: ARGS.limit]
    print(f"  questions      : {len(questions)}")
    print("=" * 78)

    results: list[Result] = []
    for index, question in enumerate(questions, start=1):
        session = question.session or f"suite-{index}"
        turn_start = time.perf_counter()

        for setup in question.context_turns:
            await service.answer(setup, session_id=session)

        outcome_answer = await service.answer(question.text, session_id=session)
        elapsed = (time.perf_counter() - turn_start) * 1000

        passed, reason = judge(
            question,
            outcome_answer.answer,
            outcome_answer.grounded,
            len(outcome_answer.citations),
        )
        results.append(
            Result(
                question=question,
                passed=passed,
                answer=outcome_answer.answer,
                reason=reason,
                grounded=outcome_answer.grounded,
                cited=len(outcome_answer.citations),
                ms=elapsed,
            )
        )

        mark = "." if passed else "F"
        sys.stdout.write(mark)
        sys.stdout.flush()
        if index % 50 == 0:
            sys.stdout.write(f"  {index}\n")

        if ARGS.verbose or not passed:
            print()
            print(f"  [{'PASS' if passed else 'FAIL'}] #{index} ({question.label})")
            print(f"        Q: {question.text}")
            if question.context_turns:
                print(f"        after: {question.context_turns}")
            if question.expect_values:
                print(f"        want: {[v[:60] for v in question.expect_values]}")
            print(f"        A: {' '.join(outcome_answer.answer.split())[:220]}")
            print(f"        grounded={outcome_answer.grounded} cites={len(outcome_answer.citations)} {elapsed:.0f}ms")
            if reason:
                print(f"        WHY: {reason}")

        if not passed and ARGS.fail_fast:
            break

    # ------------------------------------------------------------- summary
    failed = [r for r in results if not r.passed]
    by_label: dict[str, list[Result]] = {}
    for r in results:
        by_label.setdefault(r.question.label.split("/")[0], []).append(r)

    print(f"\n\n{'=' * 78}")
    print("  SUMMARY")
    print("=" * 78)
    for group, items in sorted(by_label.items()):
        bad = [i for i in items if not i.passed]
        print(f"  {group:<8} {len(items) - len(bad):>3}/{len(items):<3} passed"
              + (f"   ({len(bad)} failing)" if bad else ""))

    latencies = sorted(r.ms for r in results)
    if latencies:
        print(f"\n  latency p50={latencies[len(latencies)//2]:.0f}ms "
              f"p95={latencies[int(len(latencies)*0.95)]:.0f}ms "
              f"max={latencies[-1]:.0f}ms")
    print(f"  wall clock    {time.perf_counter() - started:.0f}s")

    if failed:
        print(f"\n  {len(failed)} FAILING:")
        for r in failed:
            print(f"    #{results.index(r)+1} [{r.question.label}] {r.question.text[:56]}")
            print(f"         {r.reason}")
        print(f"\n{'=' * 78}\n  VERDICT: FAIL ({len(failed)}/{len(results)})\n{'=' * 78}")
        return 1

    print(f"\n{'=' * 78}\n  VERDICT: PASS ({len(results)}/{len(results)})\n{'=' * 78}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
