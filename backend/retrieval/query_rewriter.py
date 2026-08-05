"""
Query understanding: rewrite, expand, and infer filters.

A raw voice utterance is a bad search query. It arrives as "yeah so what about
the school thing for my daughter", carries pronouns that only resolve against
history, and mixes a *filter* ("I'm from Vijayawada") with an *information need*
("what will you do about roads").

Three layers, cheapest first — this ordering matters because the expensive one is
on the critical path of a spoken turn:

1. **Rule layer (~0.1 ms).** Strip fillers, expand campaign-domain acronyms and
   Telugu/Hindi transliterations, and infer district/category/topic filters from
   the gazetteer. Deterministic, free, and handles the majority of utterances.
2. **Contextual layer (~0.1 ms).** If the utterance is a follow-up (short,
   pronoun-led, no new entity), splice in the previous query's subject so it
   becomes self-contained.
3. **LLM layer (~250–400 ms, optional).** Only invoked when the rule layers
   judge the query still under-specified. Runs on Haiku, not Opus, and is
   skipped entirely in voice mode unless it can be overlapped with retrieval.

We also emit *multiple* query variants when useful. Retrieving with both "Amma
Vodi eligibility" and the literal utterance and fusing the results reliably beats
either alone, at the cost of one extra vector search (~3 ms).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from core.config import settings
from ingestion.metadata import (
    CATEGORY_SIGNALS,
    TOPIC_SIGNALS,
    district_alias_index,
    find_districts,
    resolve_district,
)
from memory.conversation import Session

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- fillers
_FILLERS = (
    "um", "uh", "erm", "hmm", "ah", "oh", "like", "you know", "i mean",
    "actually", "basically", "so yeah", "well", "okay so", "ok so",
    "let me think", "kind of", "sort of",
)
_FILLER_RE = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in _FILLERS) + r")\b[,.]?\s*",
    re.IGNORECASE,
)
_POLITE_PREFIX = re.compile(
    r"^\s*(?:hi|hello|hey|namaste|namaskaram|please|can you|could you|"
    r"i want to know|i would like to know|tell me|do you know|i'd like to know)"
    r"[,\s]*",
    re.IGNORECASE,
)

# Domain synonyms/transliterations. Voters say "rythu", docs say "farmer".
_SYNONYMS: dict[str, list[str]] = {
    "rythu": ["farmer", "agriculture"],
    "raithu": ["farmer", "agriculture"],
    "vidya": ["education", "school"],
    "amma vodi": ["education", "school fees", "mother"],
    "arogya": ["health", "hospital"],
    "aarogyasri": ["health insurance", "hospital"],
    "pension": ["old age pension", "welfare"],
    "ration": ["public distribution", "food security"],
    "job": ["employment", "recruitment"],
    "jobs": ["employment", "recruitment"],
    "kalyanam": ["marriage", "welfare"],
    "illu": ["house", "housing"],
    "neeru": ["water", "irrigation"],
    "road": ["roads", "infrastructure"],
    "current": ["electricity", "power"],
    "bill": ["electricity bill", "power tariff"],
    "sarpanch": ["panchayat", "local body"],
    "mla": ["legislative assembly", "candidate"],
    "cm": ["chief minister"],
    "dbt": ["direct benefit transfer"],
    "msp": ["minimum support price"],
    "shg": ["self help group", "women"],
    "phc": ["primary health centre"],
    "rtc": ["road transport corporation", "bus"],
}

_PRONOUN_FOLLOWUP = re.compile(
    r"\b(it|its|that|those|these|there|them|they|their|theirs|this|same|"
    r"he|him|his|she|her|hers|"
    r"the scheme|the district|the candidate|the amount|the person)\b",
    re.IGNORECASE,
)
_ELLIPTIC = re.compile(
    r"^\s*(?:and|also|what about|how about|ok(?:ay)?|then|but|"
    r"what else|anything else|more)\b",
    re.IGNORECASE,
)

_ORIGIN_STATEMENT = re.compile(
    r"\b(?:i'?m from|i am from|i live in|i belong to|i stay in|"
    r"we'?re from|we are from|my district is|my constituency is)\b",
    re.IGNORECASE,
)


@dataclass
class RewriteResult:
    original: str
    effective: str                              # primary query for retrieval
    variants: list[str] = field(default_factory=list)   # extra queries to fuse
    filters: dict[str, Any] = field(default_factory=dict)
    inferred: dict[str, Any] = field(default_factory=dict)
    is_followup: bool = False
    is_origin_statement: bool = False
    used_llm: bool = False
    notes: list[str] = field(default_factory=list)

    def all_queries(self) -> list[str]:
        seen: dict[str, None] = {}
        for q in [self.effective, *self.variants, self.original]:
            key = q.strip()
            if key and key.lower() not in {k.lower() for k in seen}:
                seen[key] = None
        return list(seen)


class QueryRewriter:
    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------- api
    def rewrite(
        self,
        query: str,
        session: Optional[Session] = None,
        use_llm: Optional[bool] = None,
        voice_mode: bool = False,
    ) -> RewriteResult:
        original = query.strip()
        result = RewriteResult(original=original, effective=original)

        cleaned = self._clean(original)
        result.effective = cleaned or original

        # -- filter inference ------------------------------------------------
        inferred = self._infer_filters(original, session)
        result.inferred = inferred
        result.filters = dict(inferred)
        result.is_origin_statement = bool(_ORIGIN_STATEMENT.search(original))

        # -- follow-up resolution -------------------------------------------
        if session is not None:
            result.is_followup = self._is_followup(cleaned, session)
            if result.is_followup:
                contextual = self._splice_context(cleaned, session)
                if contextual and contextual != cleaned:
                    result.effective = contextual
                    result.notes.append("resolved follow-up from history")

                    # The subject of a pronoun-led follow-up lives in the spliced
                    # history, never in the utterance itself, so inferring the
                    # person from the raw text finds nothing. Without re-deriving
                    # it here the entity gate stands down on exactly the turns
                    # that need it most: "what are their declared assets?" pulled
                    # five near-identical profiles and answered from whichever
                    # one happened to rank first — a different candidate.
                    if not inferred.get("person_hint"):
                        person = _person_in_query(contextual)
                        if person:
                            inferred["person_hint"] = person
                            result.filters = dict(inferred)
                            result.notes.append(f"follow-up subject: {person}")

        # -- expansion variants ---------------------------------------------
        result.variants = self._expansions(result.effective, inferred)

        # -- optional LLM pass ----------------------------------------------
        should_use_llm = (
            settings.enable_llm_query_rewrite if use_llm is None else use_llm
        ) and self._llm is not None and not voice_mode

        if should_use_llm and self._needs_llm(result):
            rewritten = self._llm_rewrite(result, session)
            if rewritten:
                result.variants = self._dedupe(
                    [result.effective, *result.variants, rewritten]
                )[1:]
                result.effective = rewritten
                result.used_llm = True
                result.notes.append("llm rewrite")

        return result

    # -------------------------------------------------------------- rule layer
    @staticmethod
    def _clean(query: str) -> str:
        text = _POLITE_PREFIX.sub("", query)
        text = _FILLER_RE.sub("", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" ,.-")
        # Guard against over-stripping: "um, ok?" must not become "".
        return text if len(text) >= 2 else query.strip()

    def _infer_filters(self, query: str, session: Optional[Session]) -> dict[str, Any]:
        """Derive retrieval filters from the utterance, then session slots."""
        inferred: dict[str, Any] = {}

        districts = find_districts(query)
        if districts:
            inferred["district"] = districts[0]
            if len(districts) > 1:
                inferred["districts"] = districts[:3]
        elif session is not None and session.district:
            # Sticky slot: the voter told us where they're from three turns ago.
            inferred["district"] = session.district
            inferred["_district_from"] = "session"

        lowered = query.lower()

        # Category is only inferred on a *strong* signal. Over-filtering by
        # category is worse than not filtering: asking "who is my candidate"
        # while the answer sits in the manifesto returns nothing.
        for category, signals in CATEGORY_SIGNALS.items():
            if any(sig in lowered for sig in signals if len(sig) > 6):
                inferred["category_hint"] = category
                break

        topic_scores: dict[str, int] = {}
        for topic, signals in TOPIC_SIGNALS.items():
            score = sum(1 for sig in signals if sig in lowered)
            if score:
                topic_scores[topic] = score
        if topic_scores:
            inferred["topic_hint"] = max(topic_scores, key=lambda t: topic_scores[t])

        # A named person in the query is the single strongest retrieval signal on
        # a record corpus. 56 candidate profiles differ by a proper noun inside
        # ~1000 chars of identical template, so cosine similarity between them is
        # ~0.95 and the correct record does not reliably win on embeddings alone.
        # The name is matched against `record_name` and boosted hard downstream.
        person = _person_in_query(query)
        if person:
            inferred["person_hint"] = person

        return inferred

    @staticmethod
    def _is_followup(query: str, session: Session) -> bool:
        if not session.history_pairs():
            return False

        # A greeting or acknowledgement is not an elliptical follow-up. Splicing
        # the previous question into "hello" is what turned a greeting into a
        # repeat of the last factual answer.
        from retrieval.intent import classify as _classify

        if not _classify(query).needs_retrieval:
            return False

        words = query.split()
        if _ELLIPTIC.match(query):
            return True
        # Short + pronoun-bearing + no concrete place name => needs context.
        if len(words) <= 8 and _PRONOUN_FOLLOWUP.search(query):
            return not find_districts(query)
        return False

    @staticmethod
    def _splice_context(query: str, session: Session) -> str:
        """Prefix the previous user query's content words to make this standalone."""
        pairs = session.history_pairs(limit=2)
        if not pairs:
            return query
        previous_user = pairs[-1][0]
        # Take the content-bearing tail of the previous turn as the subject.
        subject = _FILLER_RE.sub("", _POLITE_PREFIX.sub("", previous_user)).strip(" ,.?")
        subject_words = [w for w in subject.split() if len(w) > 2][:10]
        if not subject_words:
            return query
        stripped = _ELLIPTIC.sub("", query).strip(" ,.?")
        if not stripped:
            stripped = query
        return f"{stripped} (regarding: {' '.join(subject_words)})"

    @staticmethod
    def _expansions(query: str, inferred: dict[str, Any]) -> list[str]:
        """Cheap deterministic variants worth an extra vector search."""
        variants: list[str] = []
        lowered = query.lower()

        expanded_terms: list[str] = []
        for term, synonyms in _SYNONYMS.items():
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                expanded_terms.extend(synonyms)
        if expanded_terms:
            variants.append(f"{query} {' '.join(dict.fromkeys(expanded_terms))}")

        # A district-scoped variant helps when the utterance names the place but
        # the relevant chunk uses the canonical district label instead of the city.
        district = inferred.get("district")
        if district and district.lower() not in lowered:
            variants.append(f"{query} {district} district")

        return QueryRewriter._dedupe(variants)[:2]

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for item in items:
            key = re.sub(r"\s+", " ", item.strip().lower())
            if key and key not in seen:
                seen[key] = None
        # Preserve original casing of first occurrence.
        out: list[str] = []
        used: set[str] = set()
        for item in items:
            key = re.sub(r"\s+", " ", item.strip().lower())
            if key and key not in used:
                used.add(key)
                out.append(item.strip())
        return out

    # --------------------------------------------------------------- llm layer
    @staticmethod
    def _needs_llm(result: RewriteResult) -> bool:
        """Only pay for an LLM call when the rules plainly did not disambiguate."""
        text = result.effective
        if result.is_followup and "regarding:" not in text:
            return True
        words = [w for w in re.split(r"\W+", text) if w]
        if len(words) <= 3:
            return True
        if _PRONOUN_FOLLOWUP.search(text) and len(words) <= 10:
            return True
        return False

    def _llm_rewrite(self, result: RewriteResult, session: Optional[Session]) -> Optional[str]:
        history = session.transcript(limit=3) if session else ""
        slots = session.slots() if session else {}
        prompt = (
            "Rewrite the voter's latest utterance into one self-contained search "
            "query for a document retrieval system.\n"
            "Rules: keep it under 25 words; resolve pronouns using the "
            "conversation; keep proper nouns exactly; do not answer the question; "
            "do not add facts. Reply with the query text only.\n\n"
            f"Known context: {slots or 'none'}\n"
            f"Conversation:\n{history or '(none)'}\n\n"
            f"Latest utterance: {result.original}\n"
            "Search query:"
        )
        try:
            rewritten = self._llm.complete_short(
                prompt,
                model=settings.rewrite_model,
                max_tokens=80,
            )
        except Exception as exc:  # noqa: BLE001 — rewriting is best-effort
            logger.warning("LLM query rewrite failed: %s", exc)
            return None

        if not rewritten:
            return None
        candidate = rewritten.strip().strip('"').splitlines()[0].strip()
        # Reject a "rewrite" that is really an answer or a refusal.
        if not candidate or len(candidate) > 300 or len(candidate.split()) > 40:
            return None
        if candidate.lower().startswith(("i cannot", "i can't", "sorry")):
            return None
        return candidate


_HONORIFIC_NAME = re.compile(
    r"\b(?:Dr|Sri|Shri|Smt|Mr|Mrs|Ms|Prof|Kum)\.?\s+"
    r"(?P<name>[A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+){0,4})"
)
# Two or more consecutive capitalised words, for names given without honorifics.
_BARE_NAME = re.compile(r"\b(?P<name>[A-Z][a-z’'-]{2,}(?:\s+[A-Z][a-z’'-]{2,}){1,3})\b")

# Capitalised words that begin sentences or name places, not people.
_NAME_STOPWORDS = {
    "what", "who", "when", "where", "why", "how", "is", "are", "the", "and",
    "tell", "me", "about", "does", "do", "can", "could", "would", "please",
    "andhra", "pradesh", "telangana", "alliance", "assembly", "district",
    "constituency", "amma", "vodi", "rythu", "bharosa", "aarogyasri", "india",
    "government", "chief", "minister", "primary", "health", "centres",
}


# Qualifiers that turn a town name into a constituency name: "Guntur West",
# "Vijayawada East", "Kakinada City", "Anantapur Urban".
_SEAT_QUALIFIERS = {
    "east", "west", "north", "south", "central", "city", "urban", "rural",
    "town", "metro", "i", "ii", "one", "two",
}


def _strip_possessive(name: str) -> str:
    """"Anuradha Merugu's" → "Anuradha Merugu"."""
    cleaned = re.sub(r"[’']s\b", "", name)
    return re.sub(r"\s+", " ", cleaned).strip(" .,'’-")


def _is_place(name: str) -> bool:
    """True when a capitalised phrase is a seat or a district, not a person.

    "Who is the candidate for Guntur West?" produced the *person* hint
    "Guntur West". No record is named that, so the entity gate did exactly what it
    is designed to do for an unknown person and dropped every profile — and a
    question the document answers in its first line came back as "I don't have
    that". A constituency is a value to look up, never a subject to gate on.

    The test is deliberately strict: an exact gazetteer alias, or an alias
    followed only by directional qualifiers. A looser check ("does this phrase
    contain any place alias") misfires on real names — "Devarakonda" is both a
    candidate surname here and a town in Nalgonda.
    """
    tokens = [t.lower().strip(".,'’-") for t in name.split() if t.strip(".,'’-")]
    if not tokens:
        return False
    aliases = district_alias_index()
    if " ".join(tokens) in aliases:
        return True
    return tokens[0] in aliases and all(t in _SEAT_QUALIFIERS for t in tokens[1:])


def _person_in_query(query: str) -> Optional[str]:
    """Extract a person's name from the query, or None.

    Honorific-led matches are trusted outright. Bare capitalised sequences are
    accepted only when no token is a question word or a known place/scheme term,
    because "Amma Vodi" and "Guntur West" are capitalised but are not people.
    """
    match = _HONORIFIC_NAME.search(query)
    if match:
        return _strip_possessive(re.sub(r"\s+", " ", match.group("name")).strip())

    for candidate in _BARE_NAME.finditer(query):
        name = re.sub(r"\s+", " ", candidate.group("name")).strip()
        tokens = [t.lower().strip(".'’-") for t in name.split()]
        if any(token in _NAME_STOPWORDS for token in tokens):
            continue
        if _is_place(name):
            continue
        # Reject a match that starts at position 0 and is just a capitalised
        # sentence opener followed by one more word.
        if candidate.start() == 0 and len(tokens) < 3:
            continue
        return name
    return None


def name_match_score(query_name: str, record_name: Optional[str]) -> float:
    """Symmetric token-overlap (F1) in [0, 1] between a queried and a record name.

    Exact string equality is too brittle — voters say "Kiran Kumar" for
    "Dr. Kiran Kumar Gollapudi", and ASR drops honorifics. But the scoring must be
    **symmetric**, which is the part that is easy to get wrong.

    An earlier version scored coverage-of-query plus a surname bonus. On this
    corpus that gave "Sarojini Vasireddy" vs "Padmavathi Vasireddy" a score of
    0.85 — a shared surname alone was nearly a full match, so three different
    women named Vasireddy all survived entity gating and their three different
    asset declarations all reached the prompt.

    F1 penalises a name that is missing a token in *either* direction:

        Sarojini Vasireddy  vs Sarojini Vasireddy    → 1.00  (same person)
        Sarojini Vasireddy  vs Padmavathi Vasireddy  → 0.50  (surname only)
        Kiran Kumar         vs Kiran Kumar Gollapudi → 0.80  (partial, correct)

    Because gating thresholds relative to the *best* score, a partial query still
    matches — and two equally-partial matches are correctly kept as ambiguous.
    """
    if not query_name or not record_name:
        return 0.0

    honorifics = {"dr", "sri", "shri", "smt", "mr", "mrs", "ms", "prof", "kum"}

    def tokens(value: str) -> set[str]:
        out = set()
        for raw in value.split():
            token = raw.lower().strip(".,;:!?()\"'’-")
            # Drop the possessive: "Merugu's" must match "Merugu". Voters phrase
            # these questions possessively far more often than not ("what are
            # X's assets"), and leaving the 's attached halves the F1 score and
            # made the entity gate treat the right record as a non-match.
            if token.endswith(("'s", "’s")):
                token = token[:-2]
            elif token.endswith("s'") or token.endswith("s’"):
                token = token[:-1]
            token = token.strip(".,'’-")
            if len(token) > 1 and token not in honorifics:
                out.add(token)
        return out

    q = tokens(query_name)
    r = tokens(record_name)
    if not q or not r:
        return 0.0

    overlap = q & r
    if not overlap:
        return 0.0

    recall = len(overlap) / len(q)       # how much of the query name is present
    precision = len(overlap) / len(r)    # how much of the record name is queried
    return round(2 * precision * recall / (precision + recall), 4)


def apply_filters(
    inferred: dict[str, Any],
    explicit: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Merge inferred filters with explicit API filters — explicit always wins.

    Only `district` and `language` are promoted from inference to a hard filter.
    `category_hint` / `topic_hint` stay hints and are used for *boosting* in the
    pipeline instead: guessing a category wrong silently empties the result set,
    and a voter asking a vague question is exactly when you can least afford that.
    """
    merged: dict[str, Any] = {}

    district = inferred.get("district")
    if district:
        merged["district"] = district
    if inferred.get("districts"):
        merged["districts"] = inferred["districts"]

    explicit = explicit or {}
    for key, value in explicit.items():
        if value in (None, "", [], {}):
            continue
        if key == "district":
            resolved = resolve_district(str(value)) or value
            merged["district"] = resolved
            merged.pop("districts", None)
        elif key == "districts":
            merged["districts"] = [resolve_district(str(v)) or v for v in value]
            merged.pop("district", None)
        else:
            merged[key] = value

    return merged
