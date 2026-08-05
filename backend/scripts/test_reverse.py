"""
Reverse-lookup test: query by a field VALUE instead of by name.

Regression test for a reported failure — "who is born on 14 October 1985" was
answered with a candidate born 7 September 1985. Ground truth is parsed from the
PDF, so every assertion is against the document rather than a hand-written
expectation.
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

parser = argparse.ArgumentParser()
parser.add_argument("--no-llm", action="store_true")
ARGS = parser.parse_args()

os.environ.setdefault("VECTOR_BACKEND", "local")
os.environ["LOCAL_INDEX_FILE"] = str(BACKEND_DIR / "data" / "reverse_index.npz")
os.environ["COLLECTION_NAME"] = "reverse_test"
os.environ.setdefault("RERANK_MODE", "fast")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from core.logging_config import setup_logging  # noqa: E402

setup_logging("WARNING")

def _bare(name: str) -> str:
    """Drop the honorific: "Dr. Jayasudha Kesineni" -> "Jayasudha Kesineni"."""
    parts = [p for p in name.replace(".", " ").split() if p]
    if parts and parts[0].lower() in {"dr", "sri", "shri", "smt", "mr", "mrs", "ms", "prof", "kum"}:
        parts = parts[1:]
    return " ".join(parts)


def _names_person(answer: str, name: str) -> bool:
    """Does the answer identify this specific person?"""
    return _bare(name).lower() in " ".join(answer.split()).lower()


def _other_candidates_named(answer: str, expected: str, groups) -> list[str]:
    """Other candidates named in the answer — matched on FULL name, not surname.

    Surname matching cannot work on this corpus: `Kesineni` appears 3 times,
    `Devarakonda` 3 times, `Kanna` twice. Naming the correct "Dr. Jayasudha
    Kesineni" therefore "leaked" `Kesineni` by construction, so the check failed
    for every shared-surname record no matter how the system behaved. Comparing
    full given-plus-family names distinguishes the people the surname conflates,
    which is the thing the assertion was always trying to test.
    """
    flat = " ".join(answer.split()).lower()
    expected_bare = _bare(expected).lower()
    leaked: list[str] = []
    for names in groups:
        for other in names:
            other_bare = _bare(other).lower()
            if other_bare != expected_bare and other_bare in flat:
                leaked.append(other)
    return sorted(set(leaked))


_BORN = re.compile(r"Born\.\s*(?P<born>\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE)
_MOVABLE = re.compile(r"Movable assets Rs\.\s*(?P<v>[\d.]+)\s*lakh", re.IGNORECASE)
_SEAT = re.compile(
    r"candidate for the\s+(?P<seat>[A-Z][\w\s'’.-]{2,40}?)\s+assembly", re.IGNORECASE
)


async def main() -> int:
    from ingestion.service import get_ingest_service
    from llm.provider import resolve_verified
    from llm.rag_service import get_rag_service
    from retrieval.literals import describe, selective_literals
    from retrieval.pipeline import get_pipeline

    print("=" * 78)
    print("  REVERSE LOOKUP TEST (query by value, not by name)")
    print("=" * 78)

    pipeline = get_pipeline()
    pipeline.embedder.load()
    pipeline.store.ensure_collection(pipeline.embedder.dim, recreate=True)

    outcome = await get_ingest_service().ingest_file(PDF)
    if not outcome.ok:
        print(f"ingest failed: {outcome.error}")
        return 1
    chunks = outcome.chunks
    print(f"\nIndexed {len(chunks)} records")

    # ---- ground truth -----------------------------------------------------
    by_born: dict[str, list[str]] = {}
    by_movable: dict[str, list[str]] = {}
    by_seat: dict[str, list[str]] = {}
    for chunk in chunks:
        name = chunk.metadata.record_name
        if not name:
            continue
        flat = " ".join(chunk.text.split())
        if (m := _BORN.search(flat)):
            by_born.setdefault(m.group("born"), []).append(name)
        if (m := _MOVABLE.search(flat)):
            by_movable.setdefault(m.group("v"), []).append(name)
        if (m := _SEAT.search(flat)):
            by_seat.setdefault(m.group("seat").strip(), []).append(name)

    # Prefer values that identify exactly one record.
    unique_born = [(v, n[0]) for v, n in by_born.items() if len(n) == 1][:3]
    unique_money = [(v, n[0]) for v, n in by_movable.items() if len(n) == 1][:2]
    print(f"Ground truth: {len(by_born)} distinct DOBs, {len(unique_born)} unique")

    verified_client, resolution = await resolve_verified()
    service = get_rag_service()
    service.llm = verified_client
    use_llm = not ARGS.no_llm and service.llm.is_configured
    print(f"LLM: {resolution if use_llm else 'disabled'}")

    passes = 0
    failures: list[str] = []

    # ---- 1. literal extraction -------------------------------------------
    print(f"\n{'=' * 78}\n1. LITERAL EXTRACTION\n{'=' * 78}")
    probes = [
        "who is born on 14 October 1985",
        "which candidate was born on 7 September 1985?",
        "who has movable assets of Rs. 76.4 lakh",
        "who is the candidate for Adoni",          # no literal — must not gate
        "tell me about the manifesto",             # no literal
    ]
    for probe in probes:
        found = selective_literals(probe)
        print(f"  {probe[:48]:<50} → {describe(found)}")

    # ---- 2. retrieval by date --------------------------------------------
    print(f"\n{'=' * 78}\n2. RETRIEVAL BY DATE OF BIRTH\n{'=' * 78}")
    for dob, expected in unique_born:
        question = f"who is born on {dob}"
        result = await pipeline.retrieve(question, top_k=5)
        names = [r.metadata.record_name for r in result.results]
        ok = names == [expected]
        passes += ok
        if not ok:
            failures.append(f"retrieve {dob!r} → {names} (want [{expected!r}])")
        print(f"  [{'PASS' if ok else 'FAIL'}] {question[:46]:<48} → {names}")
        print(f"         want {expected!r} | {(result.notes or ['-'])[0][:70]}")

    # ---- 3. retrieval by amount ------------------------------------------
    print(f"\n{'=' * 78}\n3. RETRIEVAL BY ASSET AMOUNT\n{'=' * 78}")
    for amount, expected in unique_money:
        question = f"which candidate has movable assets of Rs. {amount} lakh"
        result = await pipeline.retrieve(question, top_k=5)
        names = [r.metadata.record_name for r in result.results]
        ok = expected in names and len(names) <= 2
        passes += ok
        if not ok:
            failures.append(f"retrieve Rs {amount} lakh → {names} (want {expected!r})")
        print(f"  [{'PASS' if ok else 'FAIL'}] Rs. {amount} lakh → {names}")

    # ---- 4. absent value must return nothing -----------------------------
    print(f"\n{'=' * 78}\n4. ABSENT VALUE → EMPTY CONTEXT\n{'=' * 78}")
    for absent in ("29 February 1963", "Rs. 999.9 crore"):
        question = f"who is born on {absent}" if "February" in absent else f"who has assets of {absent}"
        result = await pipeline.retrieve(question, top_k=5)
        ok = len(result.results) == 0
        passes += ok
        if not ok:
            failures.append(
                f"absent {absent!r} returned {[r.metadata.record_name for r in result.results]}"
            )
        print(f"  [{'PASS' if ok else 'FAIL'}] {absent:<20} → {len(result.results)} chunk(s)")
        print(f"         {(result.notes or ['-'])[-1][:74]}")

    # ---- 5. end-to-end answers -------------------------------------------
    if use_llm:
        print(f"\n{'=' * 78}\n5. ANSWERS\n{'=' * 78}")
        for dob, expected in unique_born[:2]:
            question = f"Who is born on {dob}?"
            answer = (await service.answer(question, session_id=f"rev-{dob}")).answer
            flat = " ".join(answer.split())
            wrong = _other_candidates_named(flat, expected, by_born.values())
            ok = _names_person(flat, expected) and not wrong
            passes += ok
            if not ok:
                failures.append(f"answer for {dob}: surname={surname in flat} wrong={wrong[:3]}")
            print(f"\n  [{'PASS' if ok else 'FAIL'}] {question}")
            print(f"        want    {expected}")
            print(f"        answer  {flat[:190]}")
            if wrong:
                print(f"        NAMED OTHER CANDIDATES: {wrong[:4]}")

        # The exact reported failure.
        print(f"\n{'-' * 78}\n  REPORTED FAILURE: 'who is born on 14 October 1985'\n{'-' * 78}")
        expected_names = by_born.get("14 October 1985", [])
        answer = (await service.answer("who is born on 14 October 1985", session_id="rev-bug")).answer
        flat = " ".join(answer.split())
        print(f"    corpus says : {expected_names or '(no record with that DOB)'}")
        print(f"    answer      : {flat[:230]}")
        if expected_names:
            hit = any(_names_person(flat, n) for n in expected_names)
            other = _other_candidates_named(flat, expected_names[0], by_born.values())
            ok = hit and not other
            passes += ok
            if not ok:
                failures.append(f"reported case: correct={hit} others_named={other[:3]}")
            print(f"    verdict     : {'PASS' if ok else 'FAIL'} (correct={hit}, others={other[:3]})")

    print(f"\n{'=' * 78}\nRESULT\n{'=' * 78}")
    print(f"  checks passed : {passes}")
    print(f"  failures      : {len(failures)}")
    for failure in failures:
        print(f"    - {failure}")
    print("\nVERDICT: PASS" if not failures else "\nVERDICT: FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
