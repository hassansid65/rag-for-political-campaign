"""
Literal-value extraction for reverse lookups.

## The failure this fixes

Asked *"who is born on 14 October 1985"*, the system answered with a candidate
born **7 September 1985**. That is the same misattribution class entity gating
solves, arriving through a different door:

* There is no person name in the query, so `person_hint` is empty and the entity
  gate never engages.
* Dense retrieval cannot help. All 56 profiles are the same template; the query
  embedding is roughly equidistant from every one of them, and a date contributes
  almost nothing to a 384-dim sentence vector.
* BM25 does not isolate it either. `14 October 1985` tokenizes to
  `{14, october, 1985}`, and *many* records share `1985`, `october`, and `14`
  (it appears in dates, bed counts, and rupee figures). A record matching two of
  three tokens outranks nothing in particular.

So the correct record is often not even in the candidate set, and the LLM answers
from whichever near-identical profile arrived.

## The approach

A query like this is really a **structured lookup**: "find the record whose
`Born.` field equals this value". Dense and sparse retrieval both *blur* values;
this module matches them exactly.

We extract distinctive literals (dates, rupee amounts, years, percentages,
counts), normalize them so `14 October 1985` and `14/10/1985` compare equal, and
hand them to the store for a conjunctive BM25 narrowing plus a verbatim
substring check. Exact, and cheap — 56 records is a microsecond-scale scan.

Only *distinctive* literals qualify. A bare `100` matches every "30 to 100 beds"
record and would gate to noise, so plain small integers are deliberately excluded.
"""

from __future__ import annotations

import calendar
import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): index for index, name in enumerate(calendar.month_abbr) if name})
_MONTH_NAMES = {index: calendar.month_name[index] for index in range(1, 13)}

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))

# "14 October 1985" / "14th October 1985" / "October 14, 1985"
_DATE_DMY = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH_ALT})\.?\s*,?\s*(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
_DATE_MDY = re.compile(
    rf"\b(?P<month>{_MONTH_ALT})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
# 14/10/1985 · 14-10-1985 · 1985-10-14
_DATE_NUMERIC = re.compile(r"\b(?P<a>\d{1,4})[/-](?P<b>\d{1,2})[/-](?P<c>\d{2,4})\b")

# "Rs. 76.4 lakh" · "76.4 lakh" · "2.49 crore" · "rupees 76.4 lakh"
_MONEY = re.compile(
    r"\b(?:rs\.?|rupees|inr)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>lakh|lakhs|crore|crores)\b",
    re.IGNORECASE,
)
# A decimal on its own is distinctive enough to be worth matching (76.4, 2.49).
_DECIMAL = re.compile(r"\b(?P<value>\d+\.\d+)\b")
_YEAR = re.compile(r"\b(?P<year>1[89]\d{2}|20[0-4]\d)\b")
_PERCENT = re.compile(r"\b(?P<value>\d{1,3}(?:\.\d+)?)\s*(?:percent|per cent|%)\b", re.IGNORECASE)

# "who is the candidate for Guntur West", "which nominee is contesting from Bobbili".
#
# A constituency is as distinctive a value here as a date, and it fails the same
# way under fuzzy retrieval: "Guntur West" and "Guntur East" differ by one token
# inside a thousand characters of identical template, and BM25 scores both on
# "Guntur". Worse, the reverse question has a symmetric trap — a seat that exists
# must resolve to exactly one record, and a seat that does not exist ("Timbuktu")
# must resolve to none rather than to the nearest profile.
#
# The seat must be capitalised and directly governed by a candidacy verb, which is
# what keeps "list every candidate from Guntur district" (an aggregation over a
# district, not a seat lookup) from being gated. A trailing "district" or "mandal"
# is rejected outright for the same reason.
_SEAT_QUERY = re.compile(
    r"\b(?:candidates?|nominee|mla|contesting|standing|represent(?:ing|s)?)"
    r"\s+(?:is\s+)?(?:for|from|in)\s+(?:the\s+)?"
    r"(?P<seat>[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,3})"
    r"(?:\s+(?P<kind>assembly|constituency|seat|district|mandal))?"
)
_SEAT_REJECT_KINDS = {"district", "mandal"}


@dataclass
class Literal:
    """One distinctive value found in a query."""

    kind: str                      # date | money | decimal | year | percent
    raw: str                       # as written in the query
    # Alternate spellings to match against document text. Any hit counts.
    variants: list[str] = field(default_factory=list)
    # Higher = more selective, so a date outranks a bare year when gating.
    specificity: int = 1

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        return any(variant.lower() in lowered for variant in self.variants)


def _norm_year(value: str) -> str:
    year = int(value)
    if year < 100:
        year += 2000 if year < 50 else 1900
    return str(year)


def _date_variants(day: int, month: int, year: str) -> list[str]:
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return []
    name = _MONTH_NAMES[month]
    return [
        f"{day} {name} {year}",
        f"{day:02d} {name} {year}",
        f"{name} {day}, {year}",
        f"{day}/{month}/{year}",
        f"{day:02d}/{month:02d}/{year}",
        f"{year}-{month:02d}-{day:02d}",
    ]


def _seat_variants(seat: str) -> list[str]:
    """Spellings of a constituency as the records actually write it.

    The record carries the seat twice — in its title line ("… - Guntur West,
    Guntur District") and in its opening sentence ("… candidate for the Guntur
    West assembly constituency in Guntur district"). Both are matched because the
    PDF wraps lines mid-sentence, so the longer phrase can be split across a
    newline while the title line never is.
    """
    return [f"- {seat},", f"{seat} assembly", f"the {seat} assembly constituency"]


def extract_literals(query: str) -> list[Literal]:
    """Distinctive literal values in `query`, most selective first."""
    literals: list[Literal] = []
    consumed: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < e and s < end for s, e in consumed)

    # ---- dates (most selective) ---------------------------------------------
    for pattern in (_DATE_DMY, _DATE_MDY):
        for match in pattern.finditer(query):
            month = _MONTHS.get(match.group("month").lower().rstrip("."))
            if not month:
                continue
            variants = _date_variants(int(match.group("day")), month, match.group("year"))
            if not variants:
                continue
            literals.append(
                Literal(kind="date", raw=match.group(0), variants=variants, specificity=5)
            )
            consumed.append(match.span())

    for match in _DATE_NUMERIC.finditer(query):
        if overlaps(*match.span()):
            continue
        a, b, c = match.group("a"), match.group("b"), match.group("c")
        candidates: list[tuple[int, int, str]] = []
        if len(a) == 4:                                  # yyyy-mm-dd
            candidates.append((int(c), int(b), a))
        else:                                            # dd/mm/yyyy
            candidates.append((int(a), int(b), _norm_year(c)))
        for day, month, year in candidates:
            variants = _date_variants(day, month, year)
            if variants:
                literals.append(
                    Literal(kind="date", raw=match.group(0), variants=variants, specificity=5)
                )
                consumed.append(match.span())

    # ---- money -------------------------------------------------------------
    for match in _MONEY.finditer(query):
        if overlaps(*match.span()):
            continue
        value = match.group("value")
        unit = match.group("unit").lower().rstrip("s")
        literals.append(
            Literal(
                kind="money",
                raw=match.group(0).strip(),
                variants=[
                    f"Rs. {value} {unit}",
                    f"Rs {value} {unit}",
                    f"{value} {unit}",
                ],
                specificity=4,
            )
        )
        consumed.append(match.span())

    # ---- percentages -------------------------------------------------------
    for match in _PERCENT.finditer(query):
        if overlaps(*match.span()):
            continue
        value = match.group("value")
        literals.append(
            Literal(
                kind="percent",
                raw=match.group(0).strip(),
                variants=[f"{value} percent", f"{value}%", f"{value} per cent"],
                specificity=3,
            )
        )
        consumed.append(match.span())

    # ---- standalone decimals ----------------------------------------------
    for match in _DECIMAL.finditer(query):
        if overlaps(*match.span()):
            continue
        value = match.group("value")
        literals.append(
            Literal(kind="decimal", raw=value, variants=[value], specificity=3)
        )
        consumed.append(match.span())

    # ---- constituency named as the thing being looked up -------------------
    for match in _SEAT_QUERY.finditer(query):
        if overlaps(*match.span()):
            continue
        if (match.group("kind") or "").lower() in _SEAT_REJECT_KINDS:
            continue
        seat = " ".join(match.group("seat").split()).strip(" .,")
        if not seat:
            continue
        literals.append(
            Literal(
                kind="seat",
                raw=seat,
                variants=_seat_variants(seat),
                specificity=4,
            )
        )
        consumed.append(match.span())

    # ---- bare years (least selective; only if nothing better) -------------
    if not literals:
        for match in _YEAR.finditer(query):
            year = match.group("year")
            literals.append(
                Literal(kind="year", raw=year, variants=[year], specificity=1)
            )

    literals.sort(key=lambda item: item.specificity, reverse=True)
    return literals


def selective_literals(query: str, min_specificity: int = 3) -> list[Literal]:
    """Only literals selective enough to gate on.

    A bare year ("1985") is shared by many records, and a small integer like
    "100" appears in every "30 to 100 beds" priority line — gating on those would
    replace one wrong answer with a differently wrong one.
    """
    return [lit for lit in extract_literals(query) if lit.specificity >= min_specificity]


def match_all(literals: Iterable[Literal], text: str) -> bool:
    """True when every literal appears in `text` — a conjunctive record match."""
    literals = list(literals)
    return bool(literals) and all(lit.matches(text) for lit in literals)


def match_any(literals: Iterable[Literal], text: str) -> Optional[Literal]:
    for lit in literals:
        if lit.matches(text):
            return lit
    return None


def describe(literals: Iterable[Literal]) -> str:
    return ", ".join(f"{lit.kind}:{lit.raw}" for lit in literals) or "none"
