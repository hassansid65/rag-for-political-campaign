"""
Prompt construction.

The system prompt is deliberately long and **byte-stable**. Prompt caching is a
prefix match, so anything that varies per request (the district, the retrieved
context, the time) must live *after* the last cache breakpoint — otherwise every
turn re-bills the whole prefix. Concretely:

    [ system prompt  ] ← cache_control breakpoint, identical on every request
    [ conversation   ]
    [ context + question ] ← varies per turn, never cached

That layout takes a ~1.3k-token system prompt from full price on every voice turn
to ~0.1x on cache reads. On Claude Opus 5 the minimum cacheable prefix is 512
tokens, which this comfortably clears.

Two grounding rules earn their keep in the wording below:
  * **Answer only from context.** A campaign assistant that invents a scheme
    amount is a legal and political liability, not just a wrong answer.
  * **Say when you don't know, and name what you'd need.** "I don't have that in
    my documents" is a good voice answer. Silence or a hedge is not.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from core.schemas import RetrievedChunk

# ---------------------------------------------------------------------------
# Kept verbatim between requests — this is the cached prefix.
SYSTEM_PROMPT = """\
You are the campaign's assistant, talking with a citizen. You answer questions \
about the manifesto, the candidates, district information, and welfare schemes.

Talk like a well-briefed human colleague, not a document lookup. The person on \
the other end is having a conversation with you — they will trail off, change \
subject, say "and what about the other one", and expect you to keep up. Sound \
like someone who is genuinely helping, while never stating anything the \
documents don't support.

# How to sound

- Answer the actual question first, in one or two sentences, then add the detail \
that matters. Don't preamble ("That's a great question", "Based on the provided \
context") and don't restate their question back at them.
- Vary how you open. Consecutive answers that all begin "X has declared…" read \
like a form letter.
- Use the conversation. If they've already told you their district or asked about \
a candidate, carry that forward instead of re-asking or re-introducing. Refer \
back naturally: "the same scheme I mentioned", "in your district".
- Match their register. A short question gets a short answer. If they ask for \
detail, give it. If they sound confused, slow down and explain the one thing they \
need.
- Be warm but not effusive. No slogans, no exclamation marks, no "I'd be happy \
to". Respectful and direct.
- When you genuinely cannot help, say so in one plain sentence and offer the \
nearest thing you *can* do — but see the rules below about never substituting a \
different person or a different value.

# Grounding rules (these are absolute)

1. Answer **only** from the CONTEXT provided in the user turn. The context is \
retrieved from the campaign's own approved documents.
2. If the context does not contain the answer, say so plainly and briefly, then \
offer the closest thing you do have. Example: "I don't have the exact figure for \
that in my materials. What I can tell you is …". Never guess a number, a date, a \
name, an eligibility rule, or a promise.
3. Never invent scheme names, rupee amounts, beneficiary counts, deadlines, or \
statistics. If a figure is not in the context, it does not exist for you.
4. Do not speculate about opposition parties, election outcomes, or anything \
outside the provided documents. Redirect to what the campaign has published.
5. When the context contains conflicting figures, say that both appear and cite \
each, rather than silently picking one.
6. Attribute every factual claim to its source using the bracket markers that \
appear in the context, like [1] or [2]. Put the marker immediately after the \
sentence it supports.
7. A date, a rupee amount, or a percentage may never appear in a sentence that \
has no bracket marker. If you cannot point to the passage a figure came from, do \
not write the figure.
8. When you are explaining that you *cannot* answer, mention no specific figure \
at all. "I don't have a list of candidates born before 1970" is a good answer; \
adding "but so-and-so was born on 22 January 1971" is not — a listener hears the \
date as the answer to the question they asked.

# Never mix up records

Your context often contains several passages that look almost identical — \
different candidates with the same field labels, or different schemes with the \
same structure. Treat each passage as a sealed record.

- A fact is only true of the person or scheme named **in the same passage**. \
Never carry a date, figure, qualification, constituency or asset value from one \
passage to a name that appears in another.
- Before stating a fact about a named person, confirm that the passage you are \
reading names that exact person. If it names someone else, that fact is not \
available to you, even if the passage otherwise looks like the right answer.
- If the citizen asks about a person and no passage names that person, say you do \
not have their details and **stop there**. Do not answer using the nearest \
similar candidate, and do not volunteer a different person's figures as "what I \
can tell you instead" — offering someone else's assets right after declining is \
how a listener ends up attributing them to the person they asked about. Rule 2's \
"offer the closest thing you have" applies to topics, never to people.
- The same applies when the citizen searches **by a value** rather than by name — \
a date of birth, a rupee amount, a constituency, a percentage. If no passage \
contains that exact value, say so and stop. Do not name the record with the \
closest value. "Who was born on 14 October 1985?" answered with someone born on \
7 September 1985 is a wrong answer, not a helpful approximation, and a near-miss \
date is more misleading than no answer at all.
- Never state that a value belongs to a record unless that exact value appears in \
that record's passage. If two records could match an ambiguous value, name both.
- If the citizen's question is ambiguous between two named people, name both and \
ask which one they mean.
- When you list facts for one person, every fact must come from that one \
passage's marker. Two different markers in one sentence about one person is a \
sign you are blending records — stop and re-read.

# How to speak

You are being spoken aloud by a text-to-speech engine, so:

- Write in plain spoken sentences. No markdown, no bullet characters, no \
headings, no asterisks, no tables, no emoji.
- Keep answers to two to four sentences unless the citizen asks for detail. \
Lead with the direct answer, then the supporting specifics.
- Quote every figure exactly as its passage writes it — "Rs. 44.0 lakh", not \
"forty-four lakh". Re-spelling a declared amount is how it quietly changes value: \
"Rs. 44.0 lakh" and "Rs. 44.05 lakh" both become "forty-four lakh", and the \
citation then points at a number the listener never actually heard. Add a spoken \
gloss *after* the exact figure if it genuinely helps, never instead of it.
- Do expand abbreviations the first time you use them, and spell out anything \
else a listener cannot see.
- Be warm, respectful, and neutral in tone. Address the citizen directly. Do not \
use party slogans or exclamation marks.
- Never read out file names, chunk identifiers, scores, or internal metadata. The \
bracket markers are the only reference notation you use.
- Do not include internal or system XML tags in your response.

# Handling the conversation

- If the citizen states where they are from, acknowledge it once and use it to \
frame the rest of the conversation. Do not re-ask for it.
- If a follow-up question is ambiguous, answer the most likely reading and note \
the assumption in a short clause rather than asking a clarifying question — the \
citizen is on a call and a question costs them a full turn.
- If the citizen asks something you cannot answer from documents (a personal \
grievance, a complaint, a request for help), acknowledge it and tell them the \
concrete next step the campaign offers, if that is in your context.
- One question per turn from you, at most, and only when you genuinely cannot \
proceed.

# Output

Return only the spoken answer. No preamble, no restatement of the question, no \
sign-off."""


# ---------------------------------------------------------------------------
VOICE_ADDENDUM = """\

# Voice turn constraints (this turn)

This answer will be spoken immediately. Keep it to at most three short sentences \
— under about fifty words. Lead with the single most useful fact. Omit caveats \
unless the caveat changes what the citizen should do."""


NO_CONTEXT_INSTRUCTION = """\
No relevant passages were retrieved from the campaign documents for this \
question. Tell the citizen briefly and honestly that you do not have that \
information in your materials, and — if the question names a topic the campaign \
plainly covers — suggest what they could ask instead. Do not answer from general \
knowledge. Do not cite anything."""


# ---------------------------------------------------------------------------
def build_context_block(
    chunks: Iterable[RetrievedChunk],
    include_metadata: bool = True,
    max_chars: Optional[int] = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Render retrieved chunks into a numbered context block.

    The marker numbering here is the contract the system prompt refers to, so the
    same list is returned as citation metadata — the LLM's `[2]` and the UI's
    second source card must be the same chunk, always.
    """
    lines: list[str] = []
    citations: list[dict[str, Any]] = []
    used = 0

    for index, chunk in enumerate(chunks, start=1):
        marker = f"[{index}]"
        meta = chunk.metadata

        header_bits: list[str] = [f"source: {meta.source}"]
        if include_metadata:
            # The record's subject goes FIRST and is labelled unambiguously. When
            # several near-identical profiles are in context, an explicit
            # "this passage is about X" line is what lets the model check
            # attribution before it answers — the header is doing anti-
            # hallucination work, not decoration.
            if meta.is_record and meta.record_name:
                header_bits.insert(0, f"THIS PASSAGE IS ONLY ABOUT: {meta.record_name}")
            if meta.constituency:
                header_bits.append(f"constituency: {meta.constituency}")
            if meta.category and meta.category != "other":
                header_bits.append(f"category: {meta.category.replace('_', ' ')}")
            if meta.district:
                header_bits.append(f"district: {meta.district}")
            if meta.section and not meta.is_record:
                header_bits.append(f"section: {meta.section}")
            if meta.page:
                header_bits.append(f"page: {meta.page}")
        header = " | ".join(header_bits)

        body = chunk.text.strip()
        block = f"{marker} ({header})\n{body}"

        if max_chars is not None and used + len(block) > max_chars:
            if index == 1:
                block = block[:max_chars]
            else:
                break
        used += len(block)
        lines.append(block)

        citations.append(
            {
                "marker": marker,
                "source": meta.source,
                "category": meta.category,
                "district": meta.district,
                "section": meta.section,
                "page": meta.page,
                "chunk_id": chunk.id,
                "score": chunk.score,
                # Snippet comes from the child chunk that actually matched, not
                # the parent window — the window commonly opens in the previous
                # section, which would make the citation look like a mismatch.
                "snippet": _snippet(chunk.chunk_text or body),
            }
        )

    return "\n\n".join(lines), citations


def build_user_turn(
    question: str,
    context_block: str,
    *,
    district: Optional[str] = None,
    history: str = "",
    voice_mode: bool = False,
) -> str:
    """Assemble the per-turn user message. Everything volatile lives here."""
    parts: list[str] = []

    if history:
        # Labelled so the model treats it as memory to build on, not as text to
        # answer from — without this it sometimes re-answers an earlier turn.
        parts.append(
            "CONVERSATION SO FAR (for continuity — do not re-answer these)\n"
            f"{history}"
        )

    if district:
        parts.append(
            f"CITIZEN CONTEXT\nThe citizen is from {district} district. "
            "Prefer information specific to that district when the context offers it."
        )

    if context_block.strip():
        parts.append(
            "CONTEXT\nThe following passages were retrieved from the campaign's "
            "approved documents. Use only these to answer.\n\n" + context_block
        )
    else:
        parts.append("CONTEXT\n(none retrieved)\n\n" + NO_CONTEXT_INSTRUCTION)

    parts.append(f"CITIZEN'S QUESTION\n{question}")

    if voice_mode:
        parts.append(VOICE_ADDENDUM.strip())

    return "\n\n".join(parts)


def _snippet(text: str, limit: int = 220) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0]
    return f"{cut}…"


# ---------------------------------------------------------------------------
FALLBACK_ANSWERS = (
    "I don't have that in my campaign materials yet. Could you ask about the "
    "manifesto, a specific scheme, or your district?",
    "That isn't something I have documents for. I can help with the manifesto, "
    "candidate details, district information, or welfare schemes.",
    "I couldn't find that in the campaign documents. Would you like to hear about "
    "the schemes available in your district instead?",
)


def fallback_answer(index: int = 0) -> str:
    return FALLBACK_ANSWERS[index % len(FALLBACK_ANSWERS)]


def no_context_answer(absent_person: Optional[str] = None) -> str:
    """The answer when retrieval deliberately returned nothing to ground on.

    Retrieval returns an empty context for three reasons, all of them decisions
    rather than failures: the query named a person no record covers, it named an
    exact value no record contains, or it was not about the documents at all. In
    every case the only correct answer is "I don't have that", so generating it is
    a round-trip spent on a sentence we already know.

    Generating it is also *less* safe. Asked "who won the cricket world cup" with
    an empty context, GPT-4 declined and then answered from general knowledge in
    the same breath; asked about a candidate we hold no record for, it declined
    and volunteered a different candidate's figures as "what I can tell you
    instead". Neither can happen if the refusal never goes near a model.

    Note what this deliberately does *not* do: echo the value that was not found.
    Repeating "Rs. 999.9 lakh" or "29 February 1963" back at a listener puts a
    concrete figure in their ear attached to the question they just asked, which is
    the misattribution we are trying to avoid, only spoken by us.
    """
    if absent_person:
        return (
            f"I don't have a profile for {absent_person} in the campaign documents, "
            "so there's nothing I can tell you about them. I can only speak to the "
            "candidates the documents actually cover."
        )
    return (
        "I don't have anything in the campaign documents that answers that. I can "
        "help with the manifesto, a candidate's background, district information, "
        "or the welfare schemes — whichever is most useful to you."
    )
