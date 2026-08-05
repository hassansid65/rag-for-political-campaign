"""Verify record-atomic chunking against a real document."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("LOG_LEVEL", "INFO")

from core.logging_config import setup_logging  # noqa: E402

setup_logging("INFO")

from ingestion.chunker import DocumentChunker  # noqa: E402
from ingestion.loader import load_document  # noqa: E402
from ingestion.metadata import extract_metadata  # noqa: E402
from ingestion.records import extract_records  # noqa: E402

DEFAULT = Path(__file__).resolve().parents[2] / "data" / "RAG_Test_Candidate_Profiles.pdf"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"not found: {path}")
        return 1

    print(f"Loading {path.name}")
    document = load_document(path)
    print(f"  pages={document.pages}  blocks={len(document.blocks)}  chars={document.char_count}")

    record_set = extract_records(document.raw_text)
    print(f"\nRecord detection: detected={record_set.detected}")
    print(f"  schema : {record_set.schema}")
    print(f"  stats  : {record_set.stats()}")

    doc_meta = extract_metadata(document.raw_text, filename=path.name)
    print(f"\nDocument metadata: category={doc_meta.category} districts={doc_meta.districts[:4]}")

    chunks = DocumentChunker().chunk(document, doc_meta, source_path=str(path))
    print(f"\nChunks produced: {len(chunks)}")

    # ---- the property that matters: exactly one entity per chunk -----------
    print(f"\n{'=' * 78}\nFIRST 3 CHUNKS (verbatim)\n{'=' * 78}")
    for chunk in chunks[:3]:
        m = chunk.metadata
        print(f"\n--- {chunk.id} | {len(chunk.text)} chars | page={m.page} ---")
        print(f"    record_name  : {m.record_name!r}")
        print(f"    constituency : {m.constituency!r}")
        print(f"    district     : {m.district!r}")
        print(f"    labels       : {m.record_labels}")
        print(f"    parent_text  : {'none (correct for records)' if not chunk.parent_text else 'PRESENT — BUG'}")
        print("    " + "-" * 60)
        for line in chunk.text.splitlines():
            print(f"    | {line}")

    # ---- integrity checks --------------------------------------------------
    print(f"\n{'=' * 78}\nINTEGRITY CHECKS\n{'=' * 78}")
    failures: list[str] = []

    named = [c for c in chunks if c.metadata.record_name]
    print(f"  chunks with a record_name           : {len(named)}/{len(chunks)}")
    if len(named) < len(chunks):
        failures.append(f"{len(chunks) - len(named)} chunk(s) have no record_name")

    # No chunk may contain two candidate titles — that is the merge failure.
    import re

    title_re = re.compile(r"(?:Dr|Sri|Smt|Shri|Ms|Mr)\.?\s+[A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+){0,3}\s+[—–-]\s")
    multi = [(c.id, len(title_re.findall(c.text))) for c in chunks if len(title_re.findall(c.text)) > 1]
    print(f"  chunks containing >1 record title   : {len(multi)}")
    if multi:
        failures.append(f"merged records in {multi[:5]}")

    # Every chunk must name its subject in the first line.
    anonymous = [
        c.id for c in chunks
        if c.metadata.record_name and c.metadata.record_name not in c.text.splitlines()[0]
    ]
    print(f"  chunks whose 1st line lacks the name: {len(anonymous)}")
    if anonymous:
        failures.append(f"anonymous leading line in {anonymous[:5]}")

    # Each candidate should appear exactly once (one record per person).
    from collections import Counter

    counts = Counter(c.metadata.record_name for c in chunks if c.metadata.record_name)
    dupes = {k: v for k, v in counts.items() if v > 1}
    print(f"  distinct entities                   : {len(counts)}")
    print(f"  entities split across chunks        : {len(dupes)} {list(dupes)[:3]}")

    sizes = [len(c.text) for c in chunks]
    print(f"  chunk size min/avg/max              : {min(sizes)} / {round(sum(sizes)/len(sizes))} / {max(sizes)}")

    districts = Counter(c.metadata.district for c in chunks)
    print(f"  district spread                     : {dict(list(districts.items())[:6])}")

    print()
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("ALL CHECKS PASSED — one entity per chunk, every chunk self-identifying")
    return 0


if __name__ == "__main__":
    sys.exit(main())
