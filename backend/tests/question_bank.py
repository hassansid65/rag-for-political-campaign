"""
The question bank: 50 conversational + 60 hard document questions.

## Why expectations are derived, not written

Hand-writing 60 expected answers against a 56-record PDF guarantees two things:
transcription mistakes, and a suite that silently rots when the document changes.
Instead we parse every record's fields out of the PDF and *generate* the
questions and their ground truth together. A question can then assert against the
document rather than against my memory of it.

## The two halves test opposite failure modes

**Conversational (50).** Must NOT produce a factual answer. The reported bug was
"tell me what can you do for me" returning a candidate's constituency priorities —
so the assertion is negative: no candidate name, no rupee figure, no date, no
citation. This half catches over-eager retrieval.

**Document (60).** Must produce the *right* value, cite it, and never name another
record. This half catches misattribution.

Some questions are deliberately unanswerable by design (cross-record aggregation
like "who has the highest assets" needs all 56 records, and top-k is 5). Those are
marked `NO_FABRICATION`: the pass condition is that the system does not invent an
answer, not that it produces one. Asserting otherwise would be testing a feature
the architecture intentionally does not have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Check(str, Enum):
    CONVERSATIONAL = "conversational"   # must not answer factually
    FACT = "fact"                       # must contain expected value + citation
    REFUSAL = "refusal"                 # must decline, quote no figures
    NO_FABRICATION = "no_fabrication"   # may decline; must not invent
    FOLLOWUP = "followup"               # multi-turn; subject must persist


@dataclass
class Question:
    text: str
    check: Check
    # For FACT: values that must appear (digits or spoken form).
    expect_values: list[str] = field(default_factory=list)
    # For FACT/FOLLOWUP: the record the answer must be about.
    expect_record: Optional[str] = None
    # Substrings that must NOT appear.
    forbid: list[str] = field(default_factory=list)
    # Preceding turns in the same session (FOLLOWUP).
    context_turns: list[str] = field(default_factory=list)
    label: str = ""
    session: Optional[str] = None


# ===========================================================================
#  CONVERSATIONAL — 50 utterances that must never yield a factual answer
# ===========================================================================
CONVERSATIONAL: list[tuple[str, str]] = [
    # greetings
    ("hey", "greeting"),
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hii", "greeting"),
    ("heyy", "greeting"),
    ("hello there", "greeting"),
    ("good morning", "greeting"),
    ("good evening", "greeting"),
    ("namaste", "greeting"),
    ("namaskaram", "greeting"),
    ("yo", "greeting"),
    ("hey there", "greeting"),
    # capability — the reported bug lives here
    ("tell me what can you do for me", "capability (reported bug)"),
    ("what can you do", "capability"),
    ("what can you do for me", "capability"),
    ("what do you do", "capability"),
    ("how can you help me", "capability"),
    ("what can I ask you", "capability"),
    ("what are you able to do", "capability"),
    ("help", "capability"),
    ("what do you know", "capability"),
    ("tell me how you can help", "capability"),
    ("so what do you actually do", "capability"),
    ("can you help me", "capability"),
    ("what all can you tell me", "capability"),
    # identity
    ("who are you", "identity"),
    ("what are you", "identity"),
    ("are you a bot", "identity"),
    ("are you a real person", "identity"),
    ("is this a human", "identity"),
    ("what is your name", "identity"),
    ("am I talking to a machine", "identity"),
    # chit-chat
    ("how are you", "chitchat"),
    ("how are you doing", "chitchat"),
    ("what's up", "chitchat"),
    ("can you hear me", "chitchat"),
    ("are you there", "chitchat"),
    ("hello?", "chitchat"),
    ("testing", "chitchat"),
    # thanks / affirm / farewell
    ("thanks", "thanks"),
    ("thank you", "thanks"),
    ("thanks a lot", "thanks"),
    ("thank you very much", "thanks"),
    ("ok", "affirm"),
    ("okay", "affirm"),
    ("got it", "affirm"),
    ("alright", "affirm"),
    ("i see", "affirm"),
    ("bye", "farewell"),
    ("goodbye", "farewell"),
    ("that's all", "farewell"),
]

# Out-of-scope questions: not small talk, but not in the documents either. The
# assistant must decline rather than reach for a candidate record.
OUT_OF_SCOPE: list[str] = [
    "what is the capital of France",
    "who won the cricket world cup",
    "what is the weather today",
    "how do I cook biryani",
    "what is the price of gold",
    "tell me a joke",
    "what is 2 plus 2",
    "who is the president of America",
    "recommend a good movie",
    "what time is it",
]


# ===========================================================================
#  DOCUMENT QUESTION TEMPLATES — instantiated per record from parsed fields
# ===========================================================================
FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    "born": re.compile(r"Born\.\s*(?P<v>[^.]+?\d{4})", re.IGNORECASE),
    "education": re.compile(r"Education\.\s*(?P<v>.+?)(?=\s+(?:Profession|Political)\.)", re.IGNORECASE | re.DOTALL),
    "profession": re.compile(r"Profession\.\s*(?P<v>.+?)(?=\s+Political\.)", re.IGNORECASE | re.DOTALL),
    "career": re.compile(r"Political career\.\s*(?P<v>.+?)(?=\s+Priorities)", re.IGNORECASE | re.DOTALL),
    "priorities": re.compile(r"Priorities for the constituency\.\s*(?P<v>.+?)(?=\s+Assets)", re.IGNORECASE | re.DOTALL),
    "languages": re.compile(r"Languages\.\s*(?P<v>[^.]+)", re.IGNORECASE),
    "movable": re.compile(r"Movable assets Rs\.\s*(?P<v>[\d.]+)\s*lakh", re.IGNORECASE),
    "immovable": re.compile(r"immovable assets Rs\.\s*(?P<v>[\d.]+)\s*crore", re.IGNORECASE),
    "liabilities": re.compile(r"liabilities Rs\.\s*(?P<v>[\d.]+)\s*lakh", re.IGNORECASE),
    "seat": re.compile(r"candidate for the\s+(?P<v>[A-Z][\w\s'’.-]{2,40}?)\s+assembly", re.IGNORECASE),
    "district": re.compile(r"assembly constituency in\s+(?P<v>[A-Z][\w\s'’.-]{2,40}?)\s+district", re.IGNORECASE),
}


def parse_record(text: str) -> dict[str, str]:
    """Pull every field out of one record's chunk text."""
    flat = " ".join(text.split())
    out: dict[str, str] = {}
    for key, pattern in FIELD_PATTERNS.items():
        match = pattern.search(flat)
        if match:
            out[key] = " ".join(match.group("v").split()).strip(" .,")
    return out


# (label, question template, field to assert on)
DIRECT_TEMPLATES: list[tuple[str, str, str]] = [
    ("dob", "When was {name} born?", "born"),
    ("dob-possessive", "What is {name}'s date of birth?", "born"),
    ("education", "What is the educational qualification of {name}?", "education"),
    ("profession", "What is {name}'s profession?", "profession"),
    ("career", "Describe the political career of {name}.", "career"),
    ("priorities", "What are {name}'s priorities for the constituency?", "priorities"),
    ("languages", "Which languages does {name} speak?", "languages"),
    ("movable", "What are the movable assets declared by {name}?", "movable"),
    ("immovable", "What immovable assets has {name} declared?", "immovable"),
    ("liabilities", "What liabilities has {name} declared?", "liabilities"),
    ("seat", "Which constituency is {name} contesting from?", "seat"),
    ("district", "Which district is {name} from?", "district"),
]

# Reverse lookups: query by value, expect the record.
REVERSE_TEMPLATES: list[tuple[str, str, str]] = [
    ("rev-dob", "Who was born on {value}?", "born"),
    ("rev-dob2", "Which candidate is born on {value}?", "born"),
    ("rev-movable", "Which candidate declared movable assets of Rs. {value} lakh?", "movable"),
    ("rev-immovable", "Who declared immovable assets of Rs. {value} crore?", "immovable"),
    ("rev-seat", "Who is the candidate for {value}?", "seat"),
]

# Follow-up chains: the subject must survive a pronoun.
FOLLOWUP_TEMPLATES: list[tuple[str, list[str], str, str]] = [
    ("fu-assets", ["Tell me about {name}."], "What are their declared assets?", "movable"),
    ("fu-dob", ["Who is {name}?"], "When were they born?", "born"),
    ("fu-langs", ["What is {name}'s profession?"], "And which languages do they speak?", "languages"),
    ("fu-seat", ["Tell me about {name}."], "Which seat are they contesting?", "seat"),
]

# Cross-record aggregation — architecturally out of reach with top-k retrieval.
# Pass condition is "does not fabricate", not "answers correctly".
AGGREGATION: list[str] = [
    "Which candidate has the highest declared assets?",
    "How many candidates are there in total?",
    "List every candidate from Guntur district.",
    "Which candidates were born before 1970?",
    "What is the average declared movable asset value?",
    "Who is the youngest candidate?",
    "How many candidates speak Tamil?",
    "Which district has the most candidates?",
]

# People who are not in the corpus at all.
ABSENT_PEOPLE: list[str] = [
    "Dr. Ramesh Chandra Patel",
    "Smt. Anjali Verma",
    "Sri Mohan Das Gupta",
    "Dr. Priya Sharma",
    "Smt. Kavita Reddy Iyer",
]

# Values that appear nowhere in the corpus.
ABSENT_VALUES: list[tuple[str, str]] = [
    ("Who was born on 29 February 1963?", "date"),
    ("Which candidate declared movable assets of Rs. 999.9 lakh?", "amount"),
    ("Who declared immovable assets of Rs. 87.65 crore?", "amount"),
    ("Who is the candidate for Timbuktu?", "seat"),
]
