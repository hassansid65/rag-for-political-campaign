"""
Utterance intent classification.

## The failure this fixes

A citizen typed "hey", then "hello", and both times got a full factual answer
about a candidate's date of birth. Every utterance was being pushed through
retrieval, so a greeting retrieved *something* — and with conversation memory
resolving it as a follow-up, "hello" became a restatement of the previous
question.

That is not a grounding bug; it is a missing conversational layer. A real
assistant recognises that "thanks" is not a search query.

## Why rules and not an LLM

This runs on every single turn, including every voice turn, and the decision is
between about eight categories that are lexically obvious. A rule pass costs
~0.05 ms and never fails open; an LLM classifier would add 300–600 ms to the
critical path of a spoken conversation to answer a question a regex already
answers. Anything genuinely ambiguous falls through to `FACTUAL`, which is the
safe default — it just means we retrieve.

Handling these locally also removes them from the hallucination surface entirely:
a greeting never reaches the generator with a retrieved context it might quote.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Intent(str, Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    THANKS = "thanks"
    AFFIRM = "affirm"              # "ok", "got it", "sure"
    CAPABILITY = "capability"      # "what can you do?"
    IDENTITY = "identity"          # "who are you?"
    CHITCHAT = "chitchat"          # "how are you?"
    FACTUAL = "factual"            # needs retrieval — the default


@dataclass
class IntentResult:
    intent: Intent
    confidence: float = 1.0
    reply: Optional[str] = None    # canned response for non-factual intents

    @property
    def needs_retrieval(self) -> bool:
        return self.intent is Intent.FACTUAL


# Trailing words that cannot change the intent. Deliberately a closed list.
#
# The first version ended every pattern with `.{0,24}$` to tolerate trailing
# words, and that wildcard is a trap in both directions: it let "what do you know
# about Guntur West" read as small talk because the topic happened to fit inside
# the character budget, while still missing "what can you do *for me*" the moment
# the tail ran long. Enumerating the filler is a few more bytes and cannot swallow
# a topic.
_TAIL = (
    r"(?:\s+(?:for|to|with|about|of|me|us|you|here|there|now|then|today|"
    r"exactly|actually|really|please|sir|madam|ji|anna|akka|everyone|folks|"
    r"guys|all|anything|something|things?|stuff|else|though|again)){0,4}"
    r"[\s,.!?]*$"
)

# Conversational lead-ins that wrap the real utterance without changing it:
# "tell me what you can do", "could you help me", "i want to know what you do".
# The reported bug was exactly this — "tell me what can you do for me" fell
# through to retrieval and came back with a candidate's constituency priorities,
# because the capability pattern was anchored at "^what".
#
# Stripping is *additive*, not a replacement: we match the utterance both as
# spoken and with the lead-in removed. "can you hear me" is a pattern in its own
# right and would lose itself if "can you" were always discarded.
_LEADIN = re.compile(
    r"^(?:"
    r"(?:so|and|now|but|ok(?:ay)?|well|yeah|yep|yes|hmm+)\b[\s,]*"
    r"|(?:please|kindly)\b[\s,]*"
    r"|i\s+(?:just\s+)?(?:want|wanna|wanted|need|would\s+like|'?d\s+like)"
    r"\s+to\s+know\b[\s,]*"
    r"|(?:can|could|would|will)\s+you(?:\s+please)?"
    r"(?:\s+(?:tell|let)\s+me(?:\s+know)?)?\b[\s,]*"
    r"|(?:do|did)\s+you\s+know\b[\s,]*"
    r"|tell\s+me\b[\s,]*"
    r"|let\s+me\s+know\b[\s,]*"
    r")+",
    re.IGNORECASE,
)

# Past this length an utterance is a real question even if it opens like small
# talk ("hello, what does the manifesto say about roads").
_MAX_SMALLTALK_WORDS = 8

# Anchored patterns. Each must match the *whole* utterance (modulo punctuation,
# filler and `_TAIL`) so that a greeting used as a prefix stays FACTUAL.
_PATTERNS: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (
        Intent.GREETING,
        re.compile(
            r"^(?:hi|hey+|hello+|helo|hii+|yo|namaste|namaskaram|namaskar|"
            r"good\s*(?:morning|afternoon|evening|day)|"
            r"vanakkam|salaam|assalam[ou]?\s*alaikum)"
            r"(?:\s+(?:there|everyone|sir|madam|ji|anna|akka))?$",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.FAREWELL,
        re.compile(
            r"^(?:bye+|goodbye|good\s*bye|see\s*you|see\s*ya|talk\s*later|"
            r"that'?s?\s*all|that\s*is\s*all|i'?m\s*done|nothing\s*else|"
            r"no\s*(?:thanks?|thank\s*you)|dhanyavad[au]?lu|"
            r"good\s*night)"
            r"(?:\s+(?:now|then|for\s*now))?$",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.THANKS,
        re.compile(
            r"^(?:thanks?|thank\s*you|thx|ty|thanks?\s*a\s*lot|"
            r"thank\s*you\s*(?:very\s*much|so\s*much)|dhanyavadalu|"
            r"chala\s*thanks)"
            r"(?:\s+(?:very\s*much|a\s*lot|so\s*much|again))?$",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.AFFIRM,
        re.compile(
            r"^(?:ok(?:ay)?|k|sure|got\s*it|understood|alright|all\s*right|"
            r"fine|good|great|nice|cool|i\s*see|makes\s*sense|"
            r"yes|yeah|yep|yup|no|nope)$",
            re.IGNORECASE,
        ),
    ),
    (
        Intent.CAPABILITY,
        re.compile(
            r"^(?:"
            # "what can/do you (actually) do / know / help with / tell me"
            r"what\s+(?:all\s+)?(?:can|could|do)\s+you\s+"
            r"(?:actually\s+|really\s+|exactly\s+)?"
            r"(?:do|help\s+(?:me\s+)?with|help|know|offer|answer|tell\s+me)"
            r"|what\s+(?:all\s+)?(?:can|could)\s+you\s+tell\s+me"
            r"|how\s+(?:can|could|do)\s+you\s+help"
            # "tell me how you can help" arrives here with the lead-in removed.
            r"|how\s+you\s+can\s+help"
            r"|what\s+are\s+you\s+(?:for|able\s+to\s+do|capable\s+of|good\s+for)"
            r"|what\s+kind\s+of\s+(?:questions|things|stuff)"
            r"|what\s+do\s+you\s+have"
            r"|what\s+(?:can|should|could)\s+i\s+ask"
            r"|help"
            r")" + _TAIL,
            re.IGNORECASE,
        ),
    ),
    (
        Intent.IDENTITY,
        re.compile(
            r"^(?:"
            r"who\s+are\s+you"
            r"|who'?s?\s+this"
            r"|what\s+are\s+you"
            # "what's your name" / "whats your name" / "what is your name".
            r"|what(?:'?s|\s+is)?\s+your\s+name"
            r"|are\s+you\s+(?:a\s+)?(?:real\s+)?"
            r"(?:bot|robot|human|person|machine|computer|programme?|ai|real)"
            r"|is\s+this\s+(?:a\s+)?(?:real\s+)?(?:bot|robot|human|person|machine|ai)"
            r"|am\s+i\s+(?:talking|speaking|chatting)\s+(?:to|with)\s+(?:a\s+)?"
            r"(?:real\s+)?(?:bot|robot|human|person|machine|computer|ai)"
            r")" + _TAIL,
            re.IGNORECASE,
        ),
    ),
    (
        Intent.CHITCHAT,
        re.compile(
            r"^(?:"
            r"how\s+are\s+you(?:\s+doing)?"
            r"|how'?s?\s+it\s+going|how\s+do\s+you\s+do"
            r"|what'?s?\s+up|wassup|sup"
            r"|are\s+you\s+(?:there|listening|awake|around|online)"
            r"|(?:can|do)\s+you\s+hear\s+me"
            # Left over once "can you" is stripped as a lead-in.
            r"|hear\s+me"
            r"|hello\?+|test(?:ing)?"
            r")" + _TAIL,
            re.IGNORECASE,
        ),
    ),
)

# Stripped before matching so "um, hi there!" still reads as a greeting.
_TRIM = re.compile(
    r"^(?:um+|uh+|erm+|so|well|hmm+|ah+|oh+|please)\b[\s,]*|[\s,.!?]+$",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    cleaned = text.strip()
    for _ in range(3):                       # a couple of stacked fillers
        cleaned = _TRIM.sub("", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned)


def _strip_leadin(text: str) -> str:
    """Drop a conversational wrapper. Returns `text` unchanged if nothing is left."""
    stripped = _LEADIN.sub("", text, count=1).strip(" ,.!?")
    return stripped if len(stripped) >= 2 else text


# Replies are varied so a repeated "hi" doesn't produce an identical line, and
# each one steers toward what the assistant can actually answer.
_GREETING_REPLIES = (
    "Hello! I'm the campaign's assistant. I can tell you about our manifesto, "
    "the candidates, welfare schemes, or anything specific to your district. "
    "What would you like to know?",
    "Hi there! Ask me anything about the campaign — schemes, candidates, or your "
    "district. Where are you calling from?",
    "Namaskaram! I'm here to answer questions about the campaign. What can I help "
    "you with today?",
)

_FAREWELL_REPLIES = (
    "Thank you for your time. Do reach out any time you need information about "
    "the campaign.",
    "Goodbye, and thank you for your interest. Take care.",
)

_THANKS_REPLIES = (
    "You're welcome. Anything else you'd like to know?",
    "Happy to help. Is there anything else about the campaign I can answer?",
)

_AFFIRM_REPLIES = (
    "Is there anything else you'd like to know?",
    "Anything else I can help with?",
)

_CAPABILITY_REPLY = (
    "I answer questions from the campaign's published documents. That includes "
    "the manifesto and its promises, individual candidates and their backgrounds, "
    "welfare schemes and who is eligible, and district-level information. Tell me "
    "where you're from and I'll keep the answers local to you."
)

_IDENTITY_REPLY = (
    "I'm the campaign's voice assistant. I'm an AI, and I only answer from the "
    "campaign's published documents — so if something isn't in them, I'll tell you "
    "rather than guess. What would you like to know?"
)

_CHITCHAT_REPLY = (
    "I'm doing well, thank you — and I can hear you clearly. What would you like "
    "to know about the campaign?"
)


def classify(utterance: str, *, has_history: bool = False) -> IntentResult:
    """Classify an utterance. Falls through to FACTUAL when unsure."""
    text = _normalize(utterance or "")
    if not text:
        return IntentResult(Intent.GREETING, 1.0, _pick(_GREETING_REPLIES))

    # As spoken first, then with the lead-in removed — see `_LEADIN`.
    variants = [text]
    stripped = _strip_leadin(text)
    if stripped != text:
        variants.append(stripped)

    for variant in variants:
        if len(variant.split()) > _MAX_SMALLTALK_WORDS:
            continue
        for intent, pattern in _PATTERNS:
            if pattern.match(variant):
                return IntentResult(intent, 1.0, _reply_for(intent, has_history=has_history))

    return IntentResult(Intent.FACTUAL)


def _reply_for(intent: Intent, *, has_history: bool) -> Optional[str]:
    if intent is Intent.GREETING:
        # Mid-conversation "hi" shouldn't re-introduce the whole assistant.
        return (
            "Hello again — what else would you like to know?"
            if has_history
            else _pick(_GREETING_REPLIES)
        )
    if intent is Intent.FAREWELL:
        return _pick(_FAREWELL_REPLIES)
    if intent is Intent.THANKS:
        return _pick(_THANKS_REPLIES)
    if intent is Intent.AFFIRM:
        return _pick(_AFFIRM_REPLIES)
    if intent is Intent.CAPABILITY:
        return _CAPABILITY_REPLY
    if intent is Intent.IDENTITY:
        return _IDENTITY_REPLY
    if intent is Intent.CHITCHAT:
        return _CHITCHAT_REPLY
    return None


def _pick(options: tuple[str, ...]) -> str:
    return random.choice(options)
