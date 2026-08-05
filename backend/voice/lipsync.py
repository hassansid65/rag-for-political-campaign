"""
Lip-sync cue generation.

Two producers, one output format — the frontend animation loop never has to know
which one ran:

1. **Azure visemes (preferred).** Azure emits viseme events during synthesis for
   free. Converting them to mouth cues costs microseconds and adds *zero* latency
   to the voice turn.
2. **Rhubarb Lip Sync (fallback).** Shells out to the `rhubarb` binary from the
   ds-catalogue-bot repo. Higher-fidelity phonetic alignment, but it costs a
   subprocess plus a WAV read — typically 300–900 ms for a short answer. Fine for
   pre-rendered audio, too slow to sit on the critical path of every turn.

Both emit Rhubarb's schema (`{"mouthCues": [{start, end, value}], "metadata": …}`)
because that is what the existing `AvatarWithLipSync` component consumes, and the
avatar's viseme morph targets are keyed to those letters.
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from core.config import settings
from core.latency import METRICS
from voice.azure_speech import Viseme

logger = logging.getLogger(__name__)

# Rhubarb's 9 shapes. A = closed (p/b/m) … X = rest.
REST_SHAPE = "X"
_MIN_CUE_MS = 30.0


@dataclass
class MouthCue:
    start: float          # seconds
    end: float
    value: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "value": self.value}


def cues_from_azure_visemes(
    visemes: Sequence[Viseme],
    duration_s: float,
) -> list[MouthCue]:
    """Convert Azure viseme events into contiguous mouth cues.

    Azure gives a *start offset* per viseme, so each cue's end is the next cue's
    start. We also collapse consecutive identical shapes — the avatar lerps
    between targets, and re-issuing the same shape produces a visible stutter.
    """
    if not visemes:
        return [MouthCue(0.0, max(duration_s, 0.1), REST_SHAPE)]

    ordered = sorted(visemes, key=lambda v: v.offset_ms)
    cues: list[MouthCue] = []

    if ordered[0].offset_ms > _MIN_CUE_MS:
        cues.append(MouthCue(0.0, ordered[0].offset_ms / 1000.0, REST_SHAPE))

    for index, viseme in enumerate(ordered):
        start = viseme.offset_ms / 1000.0
        if index + 1 < len(ordered):
            end = ordered[index + 1].offset_ms / 1000.0
        else:
            end = max(duration_s, start + 0.08)
        if end - start < _MIN_CUE_MS / 1000.0:
            end = start + _MIN_CUE_MS / 1000.0

        shape = viseme.shape
        if cues and cues[-1].value == shape:
            cues[-1] = MouthCue(cues[-1].start, end, shape)
        else:
            cues.append(MouthCue(start, end, shape))

    # Always land on rest, or the mouth freezes open when audio ends.
    if cues and cues[-1].value != REST_SHAPE:
        cues.append(MouthCue(cues[-1].end, cues[-1].end + 0.12, REST_SHAPE))

    return cues


def cues_from_text_heuristic(text: str, duration_s: float) -> list[MouthCue]:
    """Last-resort cue generation with no audio analysis at all.

    Used only when Azure gave us no visemes and Rhubarb is unavailable. It maps
    characters to shapes and distributes them evenly across the known audio
    duration. The result is not phonetically accurate, but a mouth that moves
    plausibly in time with speech reads far better than a static face.
    """
    letter_shapes = {
        **{c: "A" for c in "pbm"},
        **{c: "B" for c in "ijky"},
        **{c: "C" for c in "eltdnh"},
        **{c: "D" for c in "a"},
        **{c: "E" for c in "o"},
        **{c: "F" for c in "uwr"},
        **{c: "G" for c in "fv"},
        **{c: "H" for c in "szc"},
    }
    letters = [ch for ch in text.lower() if ch.isalpha()]
    if not letters or duration_s <= 0:
        return [MouthCue(0.0, max(duration_s, 0.1), REST_SHAPE)]

    step = duration_s / len(letters)
    cues: list[MouthCue] = []
    for index, char in enumerate(letters):
        shape = letter_shapes.get(char, "C")
        start = index * step
        end = start + step
        if cues and cues[-1].value == shape:
            cues[-1] = MouthCue(cues[-1].start, end, shape)
        else:
            cues.append(MouthCue(start, end, shape))

    cues.append(MouthCue(duration_s, duration_s + 0.1, REST_SHAPE))
    return cues


# ------------------------------------------------------------------- Rhubarb
def find_rhubarb() -> Optional[str]:
    """Locate the rhubarb binary, including the copy vendored from the ref repo."""
    if settings.rhubarb_path and Path(settings.rhubarb_path).exists():
        return settings.rhubarb_path

    found = shutil.which("rhubarb")
    if found:
        return found

    system = platform.system().lower()
    binary = "rhubarb.exe" if system == "windows" else "rhubarb"
    roots = [
        Path(__file__).resolve().parent.parent / "bin",
        Path(__file__).resolve().parent.parent / "Rhubarb-Lip-Sync-1.13.0-Windows",
        Path(__file__).resolve().parent.parent / "Rhubarb",
    ]
    for root in roots:
        candidate = root / binary
        if candidate.exists():
            return str(candidate)
    return None


def cues_from_rhubarb(
    wav_path: Path,
    dialog_text: Optional[str] = None,
    recognizer: str = "phonetic",
) -> Optional[list[MouthCue]]:
    """Run Rhubarb over a WAV file. Returns None when unavailable or on failure."""
    binary = find_rhubarb()
    if not binary:
        logger.debug("Rhubarb binary not found; skipping")
        return None

    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "cues.json"
        command = [
            binary,
            "-f", "json",
            "-o", str(out_path),
            str(wav_path),
            "-r", recognizer,
        ]
        # The 'pocketSphinx' recognizer aligns far better when given the script,
        # but only supports English dialog files.
        dialog_path: Optional[Path] = None
        if dialog_text and recognizer == "pocketSphinx":
            dialog_path = Path(tmpdir) / "dialog.txt"
            dialog_path.write_text(dialog_text, encoding="utf-8")
            command.extend(["--dialogFile", str(dialog_path)])

        try:
            subprocess.run(command, check=True, capture_output=True, timeout=60)
        except FileNotFoundError:
            logger.warning("Rhubarb binary vanished at %s", binary)
            return None
        except subprocess.TimeoutExpired:
            logger.warning("Rhubarb timed out after 60s")
            return None
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Rhubarb failed: %s", exc.stderr.decode("utf-8", "replace")[:300]
            )
            return None

        if not out_path.exists():
            return None
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Rhubarb produced unreadable JSON: %s", exc)
            return None

    elapsed = (time.perf_counter() - start) * 1000
    METRICS.observe("lipsync.rhubarb", elapsed)
    logger.debug("Rhubarb produced %d cues in %.0fms", len(payload.get("mouthCues", [])), elapsed)

    return [
        MouthCue(
            start=float(cue.get("start", 0.0)),
            end=float(cue.get("end", 0.0)),
            value=str(cue.get("value", REST_SHAPE)),
        )
        for cue in payload.get("mouthCues", [])
    ]


# ------------------------------------------------------------------- unified
def build_lipsync(
    *,
    text: str,
    duration_s: float,
    visemes: Optional[Sequence[Viseme]] = None,
    wav_path: Optional[Path] = None,
    prefer: str = "azure",
) -> dict[str, Any]:
    """Produce a Rhubarb-schema lipsync payload from whatever inputs we have."""
    if not settings.enable_lipsync:
        return {
            "mouthCues": [MouthCue(0.0, max(duration_s, 0.1), REST_SHAPE).to_dict()],
            "metadata": {"source": "disabled", "duration": duration_s},
        }

    source = "heuristic"
    cues: list[MouthCue] = []

    if prefer == "rhubarb" and wav_path is not None:
        rhubarb_cues = cues_from_rhubarb(wav_path, dialog_text=text)
        if rhubarb_cues:
            cues, source = rhubarb_cues, "rhubarb"

    if not cues and visemes:
        cues, source = cues_from_azure_visemes(visemes, duration_s), "azure-visemes"

    if not cues and wav_path is not None:
        rhubarb_cues = cues_from_rhubarb(wav_path, dialog_text=text)
        if rhubarb_cues:
            cues, source = rhubarb_cues, "rhubarb"

    if not cues:
        cues = cues_from_text_heuristic(text, duration_s)
        source = "heuristic"

    return {
        "mouthCues": [cue.to_dict() for cue in cues],
        "metadata": {
            "source": source,
            "duration": round(duration_s, 3),
            "cues": len(cues),
        },
    }


def lipsync_health() -> dict[str, Any]:
    binary = find_rhubarb()
    return {
        "enabled": settings.enable_lipsync,
        "rhubarb": binary or "not found",
        "primary_source": "azure-visemes",
        "fallbacks": ["rhubarb" if binary else None, "heuristic"],
    }
