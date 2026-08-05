"""
Structure-aware recursive chunking.

Two ideas do most of the work here:

1. **Section-first, then recursive split.** We group blocks under their heading
   breadcrumb and only fall back to a recursive character split (paragraph →
   sentence → clause → word) when a section overflows `chunk_size`. A flat
   character splitter over the whole document routinely cuts a scheme's
   eligibility list away from its name; splitting inside a section boundary
   almost never does.

2. **Small child chunks, large parent windows.** Retrieval quality peaks with
   chunks small enough to be about one thing (~700 chars). Answer quality peaks
   with more surrounding context. So we embed the child and hand the LLM the
   parent window around it (`parent_window_chars`). Best of both, no extra
   vectors.

Each chunk carries a `contextual_header` (source › section breadcrumb) that is
prepended *for embedding only*. A chunk reading "Rs. 15,000 per year for two
children" is unretrievable on its own; with "Amma Vodi › Eligibility" in front
of it, it is.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from core.config import settings
from core.schemas import Chunk, ChunkMetadata
from ingestion.loader import Block, LoadedDocument
from ingestion.metadata import (
    ExtractedMetadata,
    STATE_BY_DISTRICT,
    find_districts,
    find_schemes,
    find_topics,
)
from ingestion.records import build_record_text, extract_records, record_metadata_extras

logger = logging.getLogger(__name__)

# Split separators, coarse to fine. Order matters: we always prefer the largest
# separator that yields pieces under the budget.
_SEPARATORS: tuple[str, ...] = (
    "\n\n",      # paragraph
    "\n",        # line
    ". ",        # sentence
    "; ",        # clause
    ", ",        # phrase
    " ",         # word
    "",          # character (last resort)
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'ఀ-౿])")
_QA_PAIR = re.compile(
    r"(?:^|\n)\s*(?:Q\d*[.:)]|Question\s*\d*[.:])\s*(?P<q>.+?)"
    r"\n\s*(?:A\d*[.:)]|Answer\s*\d*[.:])\s*(?P<a>.+?)"
    r"(?=\n\s*(?:Q\d*[.:)]|Question\s*\d*[.:])|\Z)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class _Section:
    """Contiguous blocks sharing a heading breadcrumb."""

    path: list[str] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)

    @property
    def heading(self) -> Optional[str]:
        return self.path[-1] if self.path else None

    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks).strip()

    def first_page(self) -> Optional[int]:
        for block in self.blocks:
            if block.page is not None:
                return block.page
        return None


# --------------------------------------------------------------- section grouping
def _group_into_sections(blocks: list[Block]) -> list[_Section]:
    sections: list[_Section] = []
    breadcrumb: list[str] = []
    current = _Section(path=[])

    for block in blocks:
        if block.kind == "heading":
            if current.blocks:
                sections.append(current)
            level = max(1, block.level or 1)
            breadcrumb = breadcrumb[: level - 1]
            breadcrumb.append(block.text.strip())
            current = _Section(path=list(breadcrumb))
            # Keep the heading in the body too — it is often the highest-signal
            # phrase in the chunk ("Amma Vodi", "Eligibility Criteria").
            current.blocks.append(block)
        else:
            current.blocks.append(block)

    if current.blocks:
        sections.append(current)
    return [s for s in sections if s.text()]


# ------------------------------------------------------------ recursive splitting
def _recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split `text` into <= chunk_size pieces, preferring the coarsest separator."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    for sep in _SEPARATORS:
        if sep == "":
            # Hard character split — only reached for pathological input
            # (e.g. a single 5000-char token). Keep the overlap so we don't
            # slice a number in half with no recovery.
            step = max(1, chunk_size - overlap)
            return [text[i : i + chunk_size] for i in range(0, len(text), step)]

        parts = text.split(sep)
        if len(parts) == 1:
            continue

        # Reassemble parts greedily up to chunk_size, carrying a tail overlap.
        pieces: list[str] = []
        buffer = ""
        for part in parts:
            candidate = part if not buffer else f"{buffer}{sep}{part}"
            if len(candidate) <= chunk_size:
                buffer = candidate
                continue

            if buffer:
                pieces.append(buffer)
                buffer = _tail_overlap(buffer, overlap)
                candidate = part if not buffer else f"{buffer}{sep}{part}"
                if len(candidate) <= chunk_size:
                    buffer = candidate
                    continue

            # A single part still exceeds the budget — recurse with a finer sep.
            if len(part) > chunk_size:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                pieces.extend(_recursive_split(part, chunk_size, overlap))
            else:
                buffer = part

        if buffer:
            pieces.append(buffer)

        result = [p.strip() for p in pieces if p.strip()]
        if result:
            return result

    return [text[:chunk_size]]


def _tail_overlap(text: str, overlap: int) -> str:
    """Take the last ~`overlap` chars of `text`, snapped to a sentence/word edge."""
    if overlap <= 0 or len(text) <= overlap:
        return ""
    tail = text[-overlap:]
    # Prefer starting the overlap at a sentence boundary so the carried context
    # reads as a whole thought rather than a fragment.
    match = _SENTENCE_END.search(tail)
    if match and len(tail) - match.end() > overlap * 0.3:
        return tail[match.end() :]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


# -------------------------------------------------------------------- FAQ pairing
def _split_faq(text: str) -> list[str]:
    """Keep each Q&A pair intact — an answer without its question is noise."""
    pairs: list[str] = []
    for match in _QA_PAIR.finditer(text):
        q = re.sub(r"\s+", " ", match.group("q")).strip()
        a = re.sub(r"\s+", " ", match.group("a")).strip()
        if q and a:
            pairs.append(f"Q: {q}\nA: {a}")
    return pairs


# ------------------------------------------------------------------ table chunks
def _split_table(text: str, chunk_size: int) -> list[str]:
    """Split a rendered table on row boundaries, repeating the header each time."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 2:
        return [text]

    header = lines[0]
    divider = lines[1] if set(lines[1].replace("|", "").strip()) <= {"-", " "} else ""
    body = lines[2:] if divider else lines[1:]
    prefix = "\n".join(x for x in (header, divider) if x)

    chunks: list[str] = []
    buffer: list[str] = []
    for row in body:
        candidate_len = len(prefix) + sum(len(r) + 1 for r in buffer) + len(row)
        if buffer and candidate_len > chunk_size:
            chunks.append(prefix + "\n" + "\n".join(buffer))
            buffer = []
        buffer.append(row)
    if buffer:
        chunks.append(prefix + "\n" + "\n".join(buffer))
    return chunks


# ============================================================================
class DocumentChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_chars: int | None = None,
        parent_window_chars: int | None = None,
        enable_parent_expansion: bool | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_chars = min_chunk_chars or settings.min_chunk_chars
        self.parent_window_chars = parent_window_chars or settings.parent_window_chars
        self.enable_parent_expansion = (
            settings.enable_parent_expansion
            if enable_parent_expansion is None
            else enable_parent_expansion
        )

    # ---------------------------------------------------------------- public
    def chunk(
        self,
        document: LoadedDocument,
        doc_meta: ExtractedMetadata,
        doc_id: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> list[Chunk]:
        doc_id = doc_id or self._doc_id(document)
        ingested_at = datetime.now(timezone.utc).isoformat()
        full_text = document.raw_text

        # Record-atomic path first: on a corpus of near-identical entity records,
        # keeping each record whole is worth far more than any size tuning,
        # because the failure it prevents (attributing A's facts to B) is silent.
        if settings.chunk_strategy in {"auto", "record"}:
            record_chunks = self._chunk_records(
                document=document,
                doc_meta=doc_meta,
                doc_id=doc_id,
                source_path=source_path,
                ingested_at=ingested_at,
            )
            if record_chunks:
                return record_chunks
            if settings.chunk_strategy == "record":
                logger.warning(
                    "CHUNK_STRATEGY=record but no record template was found in %s; "
                    "falling back to structural chunking",
                    document.source,
                )

        sections = _group_into_sections(document.blocks)

        raw_chunks: list[tuple[str, _Section, Optional[int]]] = []
        for section in sections:
            for piece in self._split_section(section, doc_meta.category):
                if len(piece) >= self.min_chunk_chars or section.heading:
                    raw_chunks.append((piece, section, section.first_page()))

        # Merge runaway-small chunks into their neighbour within the same section.
        merged = self._merge_small(raw_chunks)

        chunks: list[Chunk] = []
        total = len(merged)
        # Chunks are emitted in document order, so a monotonic cursor resolves
        # repeated boilerplate ("**Eligibility.**") to the right occurrence
        # instead of always matching the first one.
        cursor = 0
        for index, (text, section, page) in enumerate(merged):
            char_start, cursor = self._locate(full_text, text, cursor)
            char_end = char_start + len(text)

            metadata = self._build_metadata(
                text=text,
                section=section,
                page=page,
                doc_meta=doc_meta,
                document=document,
                doc_id=doc_id,
                source_path=source_path,
                index=index,
                total=total,
                char_start=char_start,
                char_end=char_end,
                ingested_at=ingested_at,
            )

            parent_text = None
            if self.enable_parent_expansion:
                parent_text = self._parent_window(full_text, char_start, char_end)

            chunks.append(
                Chunk(
                    id=f"{doc_id}-{index:04d}",
                    text=text,
                    metadata=metadata,
                    parent_text=parent_text,
                )
            )

        return chunks

    @staticmethod
    def contextual_header(metadata: ChunkMetadata) -> str:
        """Prefix used at embedding time only — never shown to the user."""
        parts: list[str] = []
        if metadata.district:
            parts.append(metadata.district)
        if metadata.category and metadata.category != "other":
            parts.append(metadata.category.replace("_", " "))
        breadcrumb = " › ".join(metadata.section_path[-2:]) if metadata.section_path else ""
        if breadcrumb:
            parts.append(breadcrumb)
        source_stem = re.sub(r"\.[a-z0-9]+$", "", metadata.source, flags=re.IGNORECASE)
        parts.append(source_stem.replace("_", " ").replace("-", " "))
        return " | ".join(dict.fromkeys(p for p in parts if p))

    @classmethod
    def embedding_text(cls, chunk: Chunk) -> str:
        header = cls.contextual_header(chunk.metadata)
        return f"{header}\n{chunk.text}" if header else chunk.text

    # --------------------------------------------------------------- internals
    def _split_section(self, section: _Section, category: str) -> list[str]:
        pieces: list[str] = []

        for block in section.blocks:
            if block.kind == "table":
                pieces.extend(_split_table(block.text, self.chunk_size))
                continue
            if category == "faq" or _QA_PAIR.search(block.text):
                pairs = _split_faq(block.text)
                if pairs:
                    pieces.extend(pairs)
                    continue
            pieces.append(block.text)

        # Re-pack the non-atomic pieces so we don't emit one tiny chunk per
        # paragraph, then split anything still over budget.
        packed: list[str] = []
        buffer = ""
        for piece in pieces:
            if len(piece) > self.chunk_size:
                if buffer:
                    packed.append(buffer)
                    buffer = ""
                packed.extend(_recursive_split(piece, self.chunk_size, self.chunk_overlap))
                continue
            candidate = piece if not buffer else f"{buffer}\n\n{piece}"
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                packed.append(buffer)
                buffer = piece
        if buffer:
            packed.append(buffer)

        return [p.strip() for p in packed if p.strip()]

    def _merge_small(
        self, items: list[tuple[str, _Section, Optional[int]]]
    ) -> list[tuple[str, _Section, Optional[int]]]:
        out: list[tuple[str, _Section, Optional[int]]] = []
        for text, section, page in items:
            if (
                out
                and len(text) < self.min_chunk_chars
                and out[-1][1] is section
                and len(out[-1][0]) + len(text) <= self.chunk_size * 1.3
            ):
                prev_text, prev_sec, prev_page = out[-1]
                out[-1] = (f"{prev_text}\n\n{text}", prev_sec, prev_page)
            else:
                out.append((text, section, page))
        return out

    def _parent_window(self, full_text: str, start: int, end: int) -> Optional[str]:
        """Expand a chunk's span outward to a ~parent_window_chars context window.

        Boundary snapping has to expand the window, not collapse it. Taking the
        *first* paragraph break after `end` looks reasonable but returns a
        zero-width window whenever the chunk already ends on a break — which for
        paragraph-packed chunks is almost always. So: snap the low edge to the
        first break at/after the padded start, and the high edge to the *last*
        break at/before the padded end.
        """
        if not full_text:
            return None

        span = end - start
        pad = max(0, (self.parent_window_chars - span) // 2)
        if pad == 0:
            return None

        lo = max(0, start - pad)
        hi = min(len(full_text), end + pad)

        para_lo = full_text.find("\n\n", lo, start)
        if para_lo != -1 and para_lo + 2 < start:
            lo = para_lo + 2

        para_hi = full_text.rfind("\n\n", end, hi)
        if para_hi > end:
            hi = para_hi

        window = full_text[lo:hi].strip()
        # Only worth carrying if it adds meaningful surrounding context.
        return window if len(window) > span + 80 else None

    def _build_metadata(
        self,
        *,
        text: str,
        section: _Section,
        page: Optional[int],
        doc_meta: ExtractedMetadata,
        document: LoadedDocument,
        doc_id: str,
        source_path: Optional[str],
        index: int,
        total: int,
        char_start: int,
        char_end: int,
        ingested_at: str,
    ) -> ChunkMetadata:
        # Chunk-level district beats document-level: a state manifesto mentions
        # 26 districts, but the paragraph about the Vijayawada metro is about NTR.
        scope = f"{' '.join(section.path)} {text}"
        chunk_districts = find_districts(scope)
        district = chunk_districts[0] if chunk_districts else doc_meta.district
        districts = chunk_districts or doc_meta.districts

        chunk_topics = find_topics(text, limit=3) or doc_meta.topics
        schemes = find_schemes(text, limit=4) or doc_meta.scheme_names

        return ChunkMetadata(
            doc_id=doc_id,
            source=document.source,
            source_path=source_path,
            category=doc_meta.category,  # type: ignore[arg-type]
            district=district,
            districts=districts[:8],
            state=STATE_BY_DISTRICT.get(district or "", doc_meta.state),
            topic=chunk_topics[0] if chunk_topics else doc_meta.topic,
            topics=chunk_topics[:4],
            section=section.heading,
            section_path=section.path[-3:],
            page=page,
            language=document.detected_language,
            candidate=doc_meta.candidate,
            party=doc_meta.party,
            scheme_names=schemes,
            entities=doc_meta.entities[:5],
            chunk_index=index,
            total_chunks=total,
            char_start=char_start,
            char_end=char_end,
            ingested_at=ingested_at,
        )

    # ------------------------------------------------------- record chunking
    def _chunk_records(
        self,
        *,
        document: LoadedDocument,
        doc_meta: ExtractedMetadata,
        doc_id: str,
        source_path: Optional[str],
        ingested_at: str,
    ) -> list[Chunk]:
        """One chunk per detected record. Returns [] when no template is found."""
        record_set = extract_records(
            document.raw_text,
            min_label_repeats=settings.record_min_label_repeats,
        )
        if not record_set.detected:
            return []

        # Map a record's first line back to a page so citations stay accurate.
        page_lookup = self._line_to_page(document)

        chunks: list[Chunk] = []
        for record in record_set.records:
            body = build_record_text(record)
            page = page_lookup.get(record.line_start)
            extras = record_metadata_extras(record)

            # Oversized records are the only case we split. Prepending the title
            # to every piece keeps each fragment attributable.
            if len(body) > settings.max_record_chars:
                pieces = _recursive_split(body, self.chunk_size, self.chunk_overlap)
                title = record.title.strip(" #*")
                pieces = [
                    p if p.startswith(title) else f"{title}\n{p}" for p in pieces
                ]
            else:
                pieces = [body]

            for piece in pieces:
                chunks.append(
                    Chunk(
                        id="",  # assigned below, once the total is known
                        text=piece,
                        metadata=self._record_metadata(
                            text=piece,
                            record=record,
                            extras=extras,
                            doc_meta=doc_meta,
                            document=document,
                            doc_id=doc_id,
                            source_path=source_path,
                            page=page,
                            ingested_at=ingested_at,
                        ),
                        # No parent window: the record *is* the unit of meaning,
                        # and expanding it would pull in the neighbouring record —
                        # exactly the contamination this strategy exists to avoid.
                        parent_text=None,
                    )
                )

        total = len(chunks)
        for index, chunk in enumerate(chunks):
            chunk.id = f"{doc_id}-{index:04d}"
            chunk.metadata.chunk_index = index
            chunk.metadata.total_chunks = total

        logger.info(
            "Record chunking %s: %d records → %d chunks (avg %d chars, %d named)",
            document.source,
            len(record_set.records),
            total,
            round(sum(len(c.text) for c in chunks) / max(1, total)),
            sum(1 for c in chunks if c.metadata.record_name),
        )
        return chunks

    def _record_metadata(
        self,
        *,
        text: str,
        record,
        extras: dict,
        doc_meta: ExtractedMetadata,
        document: LoadedDocument,
        doc_id: str,
        source_path: Optional[str],
        page: Optional[int],
        ingested_at: str,
    ) -> ChunkMetadata:
        # District comes from the record's own text, never the document's — a
        # 28-page profile book spans many districts and the document-level value
        # would tag every candidate with whichever district appears most.
        districts = find_districts(f"{record.title} {text}")
        district = districts[0] if districts else None
        topics = find_topics(text, limit=3) or doc_meta.topics

        return ChunkMetadata(
            doc_id=doc_id,
            source=document.source,
            source_path=source_path,
            category=doc_meta.category,  # type: ignore[arg-type]
            district=district,
            districts=districts[:8],
            state=STATE_BY_DISTRICT.get(district or "", doc_meta.state),
            topic=topics[0] if topics else doc_meta.topic,
            topics=topics[:4],
            section=record.title.strip(" #*") or None,
            section_path=[record.title.strip(" #*")] if record.title else [],
            page=page,
            language=document.detected_language,
            candidate=extras.get("record_name") if doc_meta.category == "candidate_profile" else doc_meta.candidate,
            party=doc_meta.party,
            scheme_names=find_schemes(text, limit=4),
            entities=[extras["record_name"]] if extras.get("record_name") else [],
            char_start=0,
            char_end=len(text),
            ingested_at=ingested_at,
            is_record=True,
            record_name=extras.get("record_name"),
            record_title=extras.get("record_title"),
            record_labels=list(extras.get("record_labels") or []),
            constituency=extras.get("constituency"),
        )

    @staticmethod
    def _line_to_page(document: LoadedDocument) -> dict[int, Optional[int]]:
        """Line index in raw_text → page number, from the loader's block pages."""
        mapping: dict[int, Optional[int]] = {}
        line = 0
        for block in document.blocks:
            for offset in range(len(block.text.splitlines()) or 1):
                mapping[line + offset] = block.page
            # Blocks are joined with "\n\n" in raw_text.
            line += (len(block.text.splitlines()) or 1) + 1
        return mapping

    @staticmethod
    def _locate(full_text: str, chunk_text: str, cursor: int) -> tuple[int, int]:
        """Find a chunk's offset in the source text, scanning forward from `cursor`.

        Chunks are re-joined from blocks, so an exact match on the whole chunk can
        fail on whitespace. We probe with a decreasing prefix and fall back to the
        cursor itself — a wrong-but-monotonic offset still yields a usable parent
        window, whereas returning 0 would window the document header every time.
        """
        if not full_text or not chunk_text:
            return cursor, cursor

        for probe_len in (160, 80, 40):
            probe = chunk_text[:probe_len]
            if len(probe) < 12:
                break
            found = full_text.find(probe, cursor)
            if found == -1:
                found = full_text.find(probe)      # wrapped: re-ordered chunk
            if found != -1:
                return found, found + max(1, len(chunk_text) // 2)

        return cursor, cursor + max(1, len(chunk_text) // 2)

    @staticmethod
    def _doc_id(document: LoadedDocument) -> str:
        # Content-hash the document so re-uploading the same file is idempotent
        # (same doc_id -> same chunk ids -> Milvus upsert replaces in place).
        digest = hashlib.sha1(
            f"{document.source}:{document.raw_text[:20000]}".encode("utf-8", "ignore")
        ).hexdigest()[:16]
        return f"doc_{digest}" if digest else f"doc_{uuid.uuid4().hex[:16]}"
