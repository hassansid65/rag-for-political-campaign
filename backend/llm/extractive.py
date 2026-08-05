"""
Extractive fallback answering — grounded answers without an LLM.

## Why this exists

If the Anthropic key is missing, invalid, rate-limited, or the API is briefly
unreachable, the honest-but-useless behaviour is "I couldn't find that in the
campaign documents." That sentence is a lie: retrieval *did* find it. The
information is sitting in the retrieved chunk; only the paraphrasing step is
unavailable.

So instead of apologising, we answer **extractively**: pull the sentence that
actually contains the answer out of the top-ranked chunk and return it verbatim
with its citation. Verbatim text cannot hallucinate — there is no generation step
in which a figure could drift — so this path is strictly *more* faithful than the
LLM path, just less fluent.

This matters beyond outages. It means a reviewer who clones the repo with no API
key still gets grounded, cited answers rather than a broken demo, and it gives the
voice loop something to speak when generation fails mid-turn.

## How the field is chosen

Record chunks have a known shape (`Born.` / `Education.` / `Assets declaration.`),
so a question asking about assets can be mapped to that field directly. For
non-record chunks we fall back to picking the sentences with the highest query
term overlap. Both are deterministic.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

from core.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

# Question intent → the record field labels that answer it. The FIRST entry that
# matches wins, so this list is ordered by specificity, not alphabetically.
#
# Two things here are load-bearing and were both wrong:
#
# **Word boundaries.** Triggers used to be tested with `in`, as bare substrings.
# "age" therefore matched inside "langu-age-s" and inside the candidate name
# "N-age-swara Rao", so "Which languages does Dr. Nageswara Rao Kanna speak?"
# answered with his date of birth. Every trigger is now anchored at a word start.
#
# **Order.** "Describe the political career of X" contains both "career" and
# "political", and "What are X's priorities for the constituency?" contains both
# "priorities" and "constituency". Whichever entry is consulted first decides the
# answer, so the narrower reading has to come first: political career before
# profession, priorities before constituency.
_FIELD_INTENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"asset|wealth|propert|liabilit|affidavit|crore|lakh|declared|net worth|"
     r"how rich|how much (?:is|are) (?:he|she|they)",
     ("Assets declaration",)),
    (r"criminal|court case|charge|pending case|convict",
     ("Assets declaration",)),
    (r"priorit|promise|pledge|plan for|agenda|focus|manifesto for|"
     r"will (?:they|he|she) do|commit",
     ("Priorities for the constituency",)),
    (r"language|speaks?\b|which tongue|mother tongue",
     ("Languages",)),
    (r"political career|politics|political (?:experience|background|history)|"
     r"elected|mla\b|party position|held office|previous term|public office",
     ("Political career",)),
    (r"constituenc|seat\b|contest|standing (?:for|from)|which area|represent|"
     r"candidate (?:for|from)|district",
     ("__summary__",)),
    (r"born|birth|dob\b|how old|aged\b|age of",
     ("Born",)),
    (r"educat|qualification|studied|study|degree|college|school|graduat|mbbs|"
     r"engineer|university",
     ("Education",)),
    (r"profession|occupation|job\b|works? as|occupation|"
     r"does for a living|employed|business",
     ("Profession", "Political career")),
    (r"career|experience|background",
     ("Political career", "Profession")),
)

_INTENT_MATCHERS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = tuple(
    (re.compile(rf"\b(?:{pattern})", re.IGNORECASE), labels)
    for pattern, labels in _FIELD_INTENTS
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
_STOPWORDS = {
    "what", "who", "when", "where", "why", "how", "is", "are", "was", "were",
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "from",
    "by", "with", "about", "do", "does", "did", "can", "could", "would", "will",
    "tell", "me", "my", "i", "am", "his", "her", "their", "they", "he", "she",
    "please", "much", "many", "any", "have", "has", "had", "be", "been",
}


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9]+", text.lower())
        if len(word) > 2 and word not in _STOPWORDS
    }


def _field_block(chunk_text: str, label: str) -> Optional[str]:
    """The text of one labelled field, e.g. everything after "Assets declaration."."""
    pattern = re.compile(
        rf"^\s*{re.escape(label)}\.\s*(?P<body>.+?)"
        rf"(?=\n\s*[A-Z][A-Za-z '/-]{{1,38}}\.\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(chunk_text)
    if not match:
        return None
    body = re.sub(r"\s+", " ", match.group("body")).strip()
    return body or None


def _summary_sentence(chunk_text: str) -> Optional[str]:
    """The record's opening claim — "X is the Alliance candidate for Y…"."""
    for line in chunk_text.splitlines():
        flat = re.sub(r"\s+", " ", line).strip()
        if not flat:
            continue
        # Skip the bare title line; take the first real sentence.
        if " is the " in flat or " is a " in flat:
            return flat if flat.endswith(".") else f"{flat}."
    return None


def _intent_labels(question: str) -> tuple[str, ...]:
    for matcher, labels in _INTENT_MATCHERS:
        if matcher.search(question):
            return labels
    return ()


def _best_sentences(text: str, question: str, limit: int = 2) -> str:
    """Highest query-overlap sentences, in original order."""
    query_tokens = _tokens(question)
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(re.sub(r"\s+", " ", text)) if s.strip()]
    if not sentences:
        return ""
    if not query_tokens:
        return " ".join(sentences[:limit])

    scored = [
        (len(query_tokens & _tokens(sentence)), index, sentence)
        for index, sentence in enumerate(sentences)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = [item for item in scored[:limit] if item[0] > 0]
    if not chosen:
        return sentences[0]
    chosen.sort(key=lambda item: item[1])   # restore reading order
    return " ".join(item[2] for item in chosen)


def _subject_mismatch(question: str, chunks: Sequence[RetrievedChunk]) -> Optional[str]:
    """If the question names a person we have no record for, return that name.

    This guard is the whole difference between a safe fallback and a dangerous
    one. Entity gating only narrows the context when a name *matches* something;
    when it matches nothing the gate stands down and retrieval returns whatever
    ranked highest. The LLM path survives that because the prompt forbids
    answering about an unnamed person — but a mechanical extractor has no such
    judgement. Without this check, "assets of Dr. Ramesh Chandra Patel" (who does
    not exist in the corpus) confidently returned a different candidate's
    declaration, which is the exact misattribution failure the whole design is
    built to prevent.
    """
    from retrieval.query_rewriter import _person_in_query, name_match_score

    person = _person_in_query(question)
    if not person:
        return None

    # Only meaningful when the corpus is record-shaped; a manifesto chunk is not
    # expected to carry a record_name.
    record_names = [c.metadata.record_name for c in chunks if c.metadata.record_name]
    if not record_names:
        return None

    best = max((name_match_score(person, name) for name in record_names), default=0.0)
    return None if best >= 0.6 else person


def extractive_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
) -> Optional[tuple[str, int]]:
    """Build a verbatim answer from the retrieved chunks.

    Returns `(answer_text, citation_index)` where `citation_index` is 1-based and
    matches the numbering in the context block — so the citation resolver treats
    this exactly like an LLM answer. Returns None when nothing usable was
    retrieved, or when the question is about someone the corpus does not cover.
    """
    if not chunks:
        return None

    # Refuse rather than substitute a different person's record.
    missing = _subject_mismatch(question, chunks)
    if missing:
        logger.info("Extractive refusal: no record matches %r", missing)
        return (
            f"I don't have a profile for {missing} in the campaign documents. "
            "I can only answer about the candidates included in them.",
            0,
        )

    top = chunks[0]
    text = top.chunk_text or top.text
    name = top.metadata.record_name
    labels = _intent_labels(question)

    # ---- record path: answer from the specific field the question asks about
    if top.metadata.is_record and labels:
        if labels == ("__summary__",):
            summary = _summary_sentence(text)
            if summary:
                return f"{summary} [1]", 1

        for label in labels:
            body = _field_block(text, label)
            if not body:
                continue
            subject = name or top.metadata.record_title or "This candidate"
            phrasing = {
                "Assets declaration": f"{subject} has declared {body[0].lower()}{body[1:]}",
                "Born": f"{subject} was born on {body}",
                "Education": f"{subject}'s education: {body}",
                "Profession": f"{subject} works as follows: {body}",
                "Political career": f"{subject}'s political career: {body}",
                "Priorities for the constituency": f"{subject}'s stated priorities are {body}",
                "Languages": f"{subject} speaks {body}",
            }.get(label, f"{subject}: {body}")
            return f"{_tidy(phrasing)} [1]", 1

    # ---- record path, no recognised intent: lead with the summary
    if top.metadata.is_record:
        summary = _summary_sentence(text)
        detail = _best_sentences(text, question, limit=1)
        parts = [p for p in (summary, detail) if p and p != summary]
        combined = " ".join([summary or "", *parts]).strip() or detail
        if combined:
            return f"{_tidy(combined)} [1]", 1

    # ---- generic path: most relevant sentences from the top chunk
    excerpt = _best_sentences(text, question, limit=2)
    if excerpt:
        return f"{_tidy(excerpt)} [1]", 1
    return None


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    if text and not text.endswith((".", "!", "?")):
        text = f"{text}."
    return text
