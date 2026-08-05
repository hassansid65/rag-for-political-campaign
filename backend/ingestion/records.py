"""
Record-aware chunking — one entity per chunk.

## The problem this solves

The candidate-profile corpus is 28 pages of near-identical records:

    Dr. Nageswara Rao Kanna — Guntur West, Guntur District
    Dr. Nageswara Rao Kanna is the Alliance candidate for …
    Born. 25 April 1977, in Guntur, Guntur district.
    Education. MBBS from Guntur Medical College, 1995. …
    Profession. Journalist and editor of a Telugu regional daily.
    Political career. First-time candidate who joined the Alliance in 2022 …
    Priorities for the constituency. Upgrading the area hospital …
    Assets declaration. Movable assets Rs. 64.7 lakh, immovable assets …
    Languages. Telugu, Urdu, English, Hindi.

Every record shares the same field labels and near-identical phrasing. For a RAG
system that is the worst case, and a size-based splitter fails in two distinct
ways, both of which produce confident wrong answers:

1. **Split mid-record.** `Assets declaration.` lands in a chunk whose text never
   names a candidate. Retrieved for "what are Kiran Kumar's assets?", the LLM
   reports whatever figures it was handed — from an arbitrary candidate.
2. **Merge two records.** A 700-char window that ends inside candidate A and
   continues into candidate B invites the model to blend them: A's education with
   B's assets. This is the dominant hallucination mode on record corpora, and it
   is *not* fixed by a better prompt — the context genuinely contains both.

Cosine similarity cannot separate these records either: they differ by a proper
noun and a few numbers in ~1200 characters of otherwise identical template.

## The approach

Treat each record as one atomic chunk, regardless of size, and stamp the entity's
name onto its metadata so retrieval can match on it directly.

Detection is template-driven rather than hardcoded, so it generalises to the
scheme booklet (`Benefit.` / `Eligibility.` / `How to apply.`) and the FAQ
(`Q1.` / `A1.`) without new code:

1. Find lines shaped like a field label — `^Label.  value`.
2. Labels occurring at least `min_label_repeats` times form the record *schema*.
3. Walk the document; **a schema label repeating means a new record began.**
   Close the current record at the last heading-ish line before that repeat.
4. Emit one chunk per record. Records over `max_record_chars` fall back to
   structural splitting, with the record title prepended to each piece so no
   fragment is ever anonymous.

Step 3 is the load-bearing idea: it needs no knowledge of what a title looks
like, only that a record does not repeat its own fields.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# "Born." / "Political career." / "Assets declaration." / "How to apply."
# Bounded length and a required capital keep ordinary sentences from matching.
_FIELD_LABEL = re.compile(
    r"^\s*(?:\*\*)?(?P<label>[A-Z][A-Za-z][A-Za-z '/-]{1,38})(?:\*\*)?\.\s+(?=\S)"
)

# "Q1." / "Q." / "Question 3:" — the FAQ variant of a record boundary.
_QA_LABEL = re.compile(r"^\s*(?P<label>Q|Q\d{1,3}|Question\s*\d{0,3})[.:)]\s+", re.IGNORECASE)

# A person/entity title: "Dr. Nageswara Rao Kanna — Guntur West, Guntur District"
# The dash separator is the strong signal; markdown headings also qualify.
_TITLE_LINE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?P<title>[^\n]{6,160}?)\s*$"
)
_HAS_DASH_SPLIT = re.compile(r"\s[—–-]\s")

# Honorific-led personal names, used to extract the entity name from a title.
_PERSON_TITLE = re.compile(
    r"^\s*(?:#{1,6}\s*)?"
    r"(?P<name>(?:Dr|Sri|Shri|Smt|Mr|Ms|Mrs|Prof|Kum)\.?\s+[A-Z][\w'’.-]*"
    r"(?:\s+[A-Z][\w'’.-]*){0,4})"
)

_CONSTITUENCY = re.compile(
    r"(?:candidate for the|contesting from)\s+(?P<seat>[A-Z][\w\s'’.-]{2,40}?)\s+"
    r"(?:assembly|parliamentary|constituency)",
    re.IGNORECASE,
)


@dataclass
class Record:
    """One atomic record: a candidate, a scheme, a Q&A pair."""

    title: str
    text: str
    labels: list[str] = field(default_factory=list)
    entity_name: Optional[str] = None
    page: Optional[int] = None
    line_start: int = 0

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class RecordSet:
    records: list[Record] = field(default_factory=list)
    schema: list[str] = field(default_factory=list)
    preamble: str = ""

    @property
    def detected(self) -> bool:
        return len(self.records) >= 2 and len(self.schema) >= 2

    def stats(self) -> dict[str, object]:
        sizes = [r.char_count for r in self.records]
        return {
            "records": len(self.records),
            "schema": self.schema,
            "min_chars": min(sizes) if sizes else 0,
            "max_chars": max(sizes) if sizes else 0,
            "avg_chars": round(sum(sizes) / len(sizes)) if sizes else 0,
            "named": sum(1 for r in self.records if r.entity_name),
        }


# Honorifics and abbreviations that look exactly like a field label
# ("Smt. Sarojini …" parses as label "Smt"). Left in, they poison the schema and
# make every person's title read as the start of a field, which shifts every
# record boundary by one line.
_LABEL_STOPWORDS = {
    "dr", "sri", "shri", "smt", "mr", "mrs", "ms", "prof", "kum", "st",
    "rs", "no", "vs", "etc", "eg", "ie", "jr", "sr", "hon", "col", "capt",
    "mahatma", "late",
}


def _label_of(line: str) -> Optional[str]:
    qa = _QA_LABEL.match(line)
    if qa:
        # Normalise Q1./Q7./Question 3 to a single schema label so numbering
        # doesn't produce 30 distinct "labels" and defeat the frequency test.
        return "Q"
    match = _FIELD_LABEL.match(line)
    if not match:
        return None
    label = match.group("label").strip()
    # Reject prose that happens to start with a capitalised clause.
    if len(label.split()) > 4:
        return None
    if label.lower().rstrip(".") in _LABEL_STOPWORDS:
        return None
    return label


def detect_schema(lines: list[str], min_label_repeats: int = 3) -> list[str]:
    """Field labels that repeat often enough to constitute a record template."""
    counts: Counter[str] = Counter()
    for line in lines:
        label = _label_of(line)
        if label:
            counts[label] += 1
    schema = [label for label, count in counts.items() if count >= min_label_repeats]
    # Preserve first-appearance order — useful for reporting and for choosing the
    # canonical "first field" of a record.
    ordered: list[str] = []
    for line in lines:
        label = _label_of(line)
        if label in schema and label not in ordered:
            ordered.append(label)
    return ordered


def _looks_like_title(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 160:
        return False
    if _label_of(line):
        return False
    if stripped.startswith("#"):
        return True
    # "Name — Seat, District" is the dominant shape in this corpus.
    if _HAS_DASH_SPLIT.search(stripped) and not stripped.endswith((".", ":", ";")):
        return True
    if _PERSON_TITLE.match(stripped):
        return True
    # A short line with no terminal punctuation reads as a heading.
    return len(stripped) <= 90 and not stripped.endswith((".", ",", ";", ":"))


def _is_strong_title(line: str) -> bool:
    """A line that unambiguously names a record: "Name — Seat, District"."""
    stripped = line.strip(" #*").strip()
    if not stripped or len(stripped) > 160 or stripped.endswith((".", ":", ";")):
        return False
    return bool(_HAS_DASH_SPLIT.search(stripped))


def _strong_title_offset(block: list[str], window: int = 6) -> int:
    """Offset of the record's strong title within its first few lines, else 0."""
    for index, line in enumerate(block[:window]):
        if _label_of(line):
            break  # reached the field block; no strong title to anchor on
        if _is_strong_title(line):
            return index
    return 0


def _entity_name(title: str, body: str) -> Optional[str]:
    """The entity a record is about — the key that makes it addressable.

    Tried in order of reliability: honorific-led name in the title, name before a
    dash separator, honorific-led name anywhere in the opening lines, then the
    bare title. The body fallback matters because a page break can strip a title
    line, and a record with no name is one that can be misattributed.
    """
    for candidate in (title, *body.splitlines()[:2]):
        match = _PERSON_TITLE.match(candidate.strip())
        if match:
            return re.sub(r"\s+", " ", match.group("name")).strip().rstrip(",")

    # "Amma Vodi — Education" → "Amma Vodi"
    if _HAS_DASH_SPLIT.search(title):
        head = _HAS_DASH_SPLIT.split(title, 1)[0].strip(" #*")
        if 2 <= len(head) <= 80:
            return head

    cleaned = title.strip(" #*").strip()
    return cleaned or None


def _find_record_starts(lines: list[str], schema_set: set[str]) -> list[int]:
    """Line indices where each record begins.

    Two passes are used deliberately. The single-pass version had to *rewind*
    lines out of the current record once it discovered a boundary, and that index
    arithmetic drifted whenever blank lines were skipped — every chunk ended up
    carrying the next record's title line, which is exactly the record-merging
    failure this module exists to prevent. Computing boundaries first and slicing
    afterwards removes the arithmetic entirely.

    The rule: **a schema label repeating means a new record started.** The record
    actually begins at the last title-looking line before that repeat, because
    that is the line naming the new entity.
    """
    starts: list[int] = []
    seen: set[str] = set()
    current_start: Optional[int] = None
    # The FIRST title-looking line since the last field label, not the last one.
    # A record opens with two title-shaped lines — the heading
    # ("Sri Naveen Devarakonda - Vijayawada East, NTR District") and then the
    # summary sentence ("Sri Naveen Devarakonda is the Alliance candidate for …"),
    # which also begins with an honorific-led name. Taking the last match put the
    # boundary between them and left the heading stranded at the end of the
    # previous chunk.
    title_run_start: Optional[int] = None

    for index, line in enumerate(lines):
        label = _label_of(line)

        if label is None:
            if _looks_like_title(line):
                if title_run_start is None:
                    title_run_start = index
            elif line.strip():
                # Ordinary prose ends the run; a blank line does not, since the
                # heading and its summary sentence are separated by one.
                title_run_start = title_run_start
            continue

        if label not in schema_set:
            continue

        if current_start is None:
            # First record: start at its title if we saw one, else at this label.
            current_start = title_run_start if title_run_start is not None else index
            starts.append(current_start)
            seen = {label}
            title_run_start = None
            continue

        if label in seen:
            boundary = (
                title_run_start
                if title_run_start is not None and title_run_start > current_start
                else index
            )
            starts.append(boundary)
            current_start = boundary
            seen = {label}
            title_run_start = None
        else:
            seen.add(label)
            # Inside a record's field block, any later title-shaped line belongs
            # to the *next* record, so start a fresh run.
            title_run_start = None

    return starts


def extract_records(
    text: str,
    *,
    min_label_repeats: int = 3,
    page_map: Optional[dict[int, int]] = None,
) -> RecordSet:
    """Split `text` into atomic records using detected field-label templates."""
    lines = text.splitlines()
    schema = detect_schema(lines, min_label_repeats=min_label_repeats)
    if len(schema) < 2:
        return RecordSet()

    schema_set = set(schema)
    starts = _find_record_starts(lines, schema_set)
    if len(starts) < 2:
        return RecordSet()

    preamble = "\n".join(lines[: starts[0]]).strip()
    bounds = [*starts, len(lines)]

    records: list[Record] = []
    for position in range(len(starts)):
        start = bounds[position]
        block = lines[start : bounds[position + 1]]

        # Drop leading document-level preamble. The first record otherwise absorbs
        # the document header ("Candidate Profiles", a byline) and takes its name
        # from it, so the record for the first candidate is not addressable by
        # that candidate's name — the one thing it must be.
        offset = _strong_title_offset(block)
        if offset:
            block = block[offset:]
            start += offset

        body = "\n".join(block).strip()
        if not body:
            continue

        labels = [
            label
            for label in (_label_of(line) for line in block)
            if label and label in schema_set
        ]
        if not labels:
            continue

        title = next((ln.strip(" #*") for ln in block if ln.strip()), "")
        records.append(
            Record(
                title=title,
                text=body,
                labels=labels,
                entity_name=_entity_name(title, body),
                page=page_map.get(start) if page_map else None,
                line_start=start,
            )
        )

    record_set = RecordSet(records=records, schema=schema, preamble=preamble)
    if record_set.detected:
        logger.info("Record template detected: %s", record_set.stats())
    return record_set


def record_metadata_extras(record: Record) -> dict[str, object]:
    """Metadata that makes a record independently addressable at query time."""
    extras: dict[str, object] = {
        "record_title": record.title,
        "record_labels": record.labels,
        "is_record": True,
    }
    if record.entity_name:
        extras["record_name"] = record.entity_name
    seat = _CONSTITUENCY.search(record.text)
    if seat:
        extras["constituency"] = re.sub(r"\s+", " ", seat.group("seat")).strip()
    return extras


def build_record_text(record: Record, source_label: Optional[str] = None) -> str:
    """The chunk body for a record.

    The title is guaranteed to lead the text. A record whose first line is a bare
    field label ("Born. …") is unanswerable — the reader cannot tell *whose* birth
    date it is — and that is precisely the misattribution failure we are removing.
    """
    body = record.text.strip()
    title = record.title.strip(" #*").strip()
    if title and not body.startswith(title):
        body = f"{title}\n{body}"
    if source_label and source_label not in body[:200]:
        body = f"{source_label}\n{body}"
    return body
