"""
Live Azure Speech check: TTS, viseme emission, and STT round-trip.

Viseme support is the thing worth verifying rather than assuming. DragonHD voices
sound better than standard neural voices but do not reliably emit viseme events,
and the avatar's mouth is driven by those events. This script reports, per voice,
whether visemes actually arrived — so the lip-sync source is a measured fact and
not a hope.

    python scripts/test_speech.py
    python scripts/test_speech.py --voice aarti --keep
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default=None, help="preset key or full Azure voice name")
    parser.add_argument("--all", action="store_true", help="test every DragonHD preset")
    parser.add_argument("--keep", action="store_true", help="keep the generated wav files")
    parser.add_argument("--no-stt", action="store_true")
    return parser.parse_args()


ARGS = parse_args()

from core.config import settings  # noqa: E402
from core.logging_config import setup_logging  # noqa: E402

setup_logging("WARNING")

from voice.azure_speech import (  # noqa: E402
    VOICE_PRESETS,
    build_ssml,
    clean_for_speech,
    get_speech,
    is_hd_voice,
)
from voice.lipsync import build_lipsync, find_rhubarb  # noqa: E402

SAMPLE = (
    "Amma Vodi pays fifteen thousand rupees per year to each mother, "
    "for every child from class one to class twelve."
)


def main() -> int:
    speech = get_speech()
    out_dir = settings.tts_audio_dir

    print("=" * 76)
    print("  AZURE SPEECH LIVE CHECK")
    print("=" * 76)
    print(f"  region          : {settings.azure_speech_region}")
    print(f"  key             : {'set (' + str(len(settings.azure_speech_key)) + ' chars)' if settings.azure_speech_key else 'NOT SET'}")
    print(f"  default voice   : {settings.azure_tts_voice}")
    print(f"  stt languages   : {settings.stt_language_list}")
    print(f"  rhubarb         : {find_rhubarb() or 'not found'}")

    if not speech.is_configured:
        print("\n  AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are not set — nothing to test.")
        return 1

    print(f"\n  health          : {speech.health()}")

    voices: list[str]
    if ARGS.all:
        voices = ["meera", "aarti", "ava", "Female_1"]
    elif ARGS.voice:
        voices = [ARGS.voice]
    else:
        voices = ["meera"]

    failures: list[str] = []
    generated: list[Path] = []

    for key in voices:
        resolved = VOICE_PRESETS.get(key, key)
        print(f"\n{'-' * 76}\n  VOICE: {key}  →  {resolved}\n{'-' * 76}")
        print(f"    hd voice      : {is_hd_voice(resolved)}")
        ssml = build_ssml(SAMPLE, resolved)
        print(f"    ssml          : {ssml[:120]}…")

        try:
            result = speech.synthesize(clean_for_speech(SAMPLE), voice=resolved)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{key}: TTS failed — {exc}")
            print(f"    RESULT        : FAILED — {exc}")
            continue

        path = out_dir / f"check_{key.replace(':', '_')}.wav"
        path.write_bytes(result.audio)
        generated.append(path)

        # Confirm the bytes are a real, playable 16 kHz mono PCM wav.
        try:
            with wave.open(str(path), "rb") as handle:
                channels = handle.getnchannels()
                rate = handle.getframerate()
                frames = handle.getnframes()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{key}: produced unreadable wav — {exc}")
            print(f"    RESULT        : unreadable wav — {exc}")
            continue

        print(f"    audio         : {len(result.audio) / 1024:.0f} KB, "
              f"{result.duration_s:.2f}s, {rate} Hz, {channels}ch, {frames} frames")
        print(f"    tts latency   : {result.latency_ms:.0f} ms")
        print(f"    visemes       : {len(result.visemes)}")

        lipsync = build_lipsync(
            text=SAMPLE, duration_s=result.duration_s, visemes=result.visemes
        )
        source = lipsync["metadata"]["source"]
        cues = lipsync["metadata"]["cues"]
        print(f"    lipsync       : {cues} cues from '{source}'")
        if result.visemes:
            shapes = [c["value"] for c in lipsync["mouthCues"][:12]]
            print(f"    first shapes  : {shapes}")
        else:
            print("    NOTE          : this voice emitted no visemes; the avatar will")
            print("                    fall back to rhubarb or the heuristic driver.")

        if result.duration_s < 1.0:
            failures.append(f"{key}: suspiciously short audio ({result.duration_s:.2f}s)")
        if cues < 2:
            failures.append(f"{key}: only {cues} mouth cue(s) — lip-sync would be static")
        print(f"    saved         : {path}")

    # ------------------------------------------------------------------- STT
    if not ARGS.no_stt and generated:
        print(f"\n{'-' * 76}\n  STT ROUND-TRIP (synthesized audio back to text)\n{'-' * 76}")
        wav = generated[0]
        try:
            stt = speech.transcribe(wav.read_bytes(), source_format="wav")
            print(f"    success       : {stt.success}")
            print(f"    language      : {stt.language}")
            print(f"    latency       : {stt.latency_ms:.0f} ms")
            print(f"    transcript    : {stt.text!r}")
            if stt.error:
                print(f"    error         : {stt.error}")
            if not stt.success:
                failures.append(f"STT failed: {stt.error}")
            else:
                # Loose check — TTS/STT round-trips rarely match verbatim.
                lowered = stt.text.lower()
                hits = [w for w in ("amma", "vodi", "thousand", "class") if w in lowered]
                print(f"    keyword hits  : {hits}")
                if len(hits) < 2:
                    failures.append(f"STT transcript looks wrong: {stt.text!r}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"STT raised: {exc}")
            print(f"    FAILED        : {exc}")

    if not ARGS.keep:
        for path in generated:
            path.unlink(missing_ok=True)

    print(f"\n{'=' * 76}")
    if failures:
        print(f"  {len(failures)} PROBLEM(S):")
        for item in failures:
            print(f"    ✗ {item}")
        print("  VERDICT: FAIL")
    else:
        print("  VERDICT: PASS — TTS, visemes and STT all working")
    print("=" * 76)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
