"""
Conversation memory.

A voice assistant lives on follow-ups. "I'm from Vijayawada" … "what about
schools there?" … "and the pension?" — none of those later utterances is a
searchable query on its own. Two mechanisms handle that:

1. **Turn history** — the last N (user, assistant) pairs, fed to the query
   rewriter so it can resolve pronouns and ellipsis.
2. **Sticky slots** — a district (or category/topic) stated once is remembered
   for the rest of the session and applied as a retrieval filter automatically.
   This is the single highest-leverage feature for the assignment's use case,
   and it is *not* the same as history: history helps rewrite the text, slots
   constrain the search space.

Storage is an in-process TTL dict. That is the right call for a single-node demo
and the wrong call for a horizontally-scaled deployment — swap `SessionStore` for
Redis and nothing above it changes (the interface is 4 methods).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.config import settings
from ingestion.metadata import resolve_district


@dataclass
class Turn:
    role: str                       # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    citations: list[dict[str, Any]] = field(default_factory=list)
    grounded: bool = True


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    turns: list[Turn] = field(default_factory=list)

    # sticky slots
    district: Optional[str] = None
    category: Optional[str] = None
    topic: Optional[str] = None
    language: str = "en"

    # last-retrieval state, used for partial-transcript reuse
    last_query: str = ""
    last_effective_query: str = ""
    last_partial: str = ""
    last_chunk_ids: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add_turn(
        self,
        role: str,
        content: str,
        citations: Optional[list[dict[str, Any]]] = None,
        grounded: bool = True,
    ) -> None:
        self.turns.append(
            Turn(role=role, content=content, citations=citations or [], grounded=grounded)
        )
        # Keep 2x max_turns entries so we always have max_turns *pairs*.
        limit = settings.memory_max_turns * 2
        if len(self.turns) > limit:
            self.turns = self.turns[-limit:]
        self.last_seen = time.time()

    def history_pairs(self, limit: Optional[int] = None) -> list[tuple[str, str]]:
        """Recent (user, assistant) pairs, oldest first."""
        limit = limit or settings.memory_max_turns
        pairs: list[tuple[str, str]] = []
        pending_user: Optional[str] = None
        for turn in self.turns:
            if turn.role == "user":
                pending_user = turn.content
            elif pending_user is not None:
                pairs.append((pending_user, turn.content))
                pending_user = None
        return pairs[-limit:]

    def transcript(self, limit: Optional[int] = None) -> str:
        """Compact transcript for the rewriter / LLM prompt."""
        lines: list[str] = []
        for user, assistant in self.history_pairs(limit):
            lines.append(f"Voter: {user}")
            lines.append(f"Assistant: {assistant}")
        return "\n".join(lines)

    def slots(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in {
                "district": self.district,
                "category": self.category,
                "topic": self.topic,
                "language": self.language,
            }.items()
            if v
        }

    def update_slots_from_text(self, text: str) -> dict[str, Any]:
        """Detect a district mention and make it sticky. Returns what changed."""
        changed: dict[str, Any] = {}
        district = _district_from_utterance(text)
        if district and district != self.district:
            self.district = district
            changed["district"] = district
        return changed

    def to_public(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": len(self.turns),
            "age_s": round(time.time() - self.created_at, 1),
            "idle_s": round(time.time() - self.last_seen, 1),
            "slots": self.slots(),
            "last_query": self.last_query,
        }


class SessionStore:
    """TTL-bounded in-process session store."""

    def __init__(self, ttl_s: Optional[int] = None) -> None:
        self.ttl_s = ttl_s or settings.session_ttl_s
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def get(self, session_id: Optional[str]) -> Session:
        sid = (session_id or "").strip() or "default"
        with self._lock:
            self._evict_expired()
            session = self._sessions.get(sid)
            if session is None:
                session = Session(session_id=sid)
                self._sessions[sid] = session
            session.last_seen = time.time()
            return session

    def reset(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            self._evict_expired()
            return [s.to_public() for s in self._sessions.values()]

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _evict_expired(self) -> None:
        cutoff = time.time() - self.ttl_s
        expired = [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]
        for sid in expired:
            self._sessions.pop(sid, None)


# ---------------------------------------------------------------------- helpers
_ORIGIN_HINTS = (
    "i'm from", "im from", "i am from", "i live in", "i belong to", "i stay in",
    "we are from", "we're from", "my district", "my constituency", "my village",
    "my town", "my city", "from ", "in ", "at ",
)


def _district_from_utterance(text: str) -> Optional[str]:
    """Resolve a district from an utterance, preferring explicit origin phrases.

    "I'm from Vijayawada" should set the slot. "Is the Vijayawada metro funded?"
    mentions a district but is a topical question, not a statement of origin — we
    still resolve it (harmless, and usually what the voter wants), but origin
    phrases win when both appear.
    """
    lowered = text.lower()
    for hint in _ORIGIN_HINTS:
        idx = lowered.find(hint)
        if idx == -1:
            continue
        tail = text[idx + len(hint) : idx + len(hint) + 60]
        district = resolve_district(tail)
        if district:
            return district
    return resolve_district(text)


_session_store: Optional[SessionStore] = None
_store_lock = threading.Lock()


def get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        with _store_lock:
            if _session_store is None:
                _session_store = SessionStore()
    return _session_store
