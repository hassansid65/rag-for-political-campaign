"""
Azure Cognitive Services Speech — STT and TTS.

Ported from the ds-catalogue-bot integration and extended for this use case:

* **Auto language detection across en-IN / te-IN / hi-IN.** Campaign calls are
  code-switched; a single-locale recognizer transcribes Telugu as garbage English.
* **SSML with a prosody rate.** Slightly faster than default (+8%) reads as
  natural for informational speech and shaves real wall-clock off long answers.
* **Sentence-level synthesis.** `synthesize_sentences` lets the caller start
  playback on sentence one while sentence two is still being generated, which is
  the single biggest perceived-latency win in the whole voice path.
* **Viseme capture.** Azure emits viseme IDs during synthesis at no extra cost.
  When available we use those instead of shelling out to Rhubarb — same lip-sync
  quality, none of the ffmpeg + subprocess latency.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from core.config import settings
from core.latency import METRICS

logger = logging.getLogger(__name__)

# Azure viseme id (0-21) -> Oculus/Rhubarb-style mouth shape letter.
# Mapping the 22 Azure ids onto the 9 Rhubarb shapes keeps one animation path in
# the frontend regardless of which backend produced the cues.
AZURE_VISEME_TO_SHAPE: dict[int, str] = {
    0: "X",   # silence
    1: "D",   # æ, ə, ʌ
    2: "D",   # ɑ
    3: "E",   # ɔ
    4: "C",   # ɛ, ʊ
    5: "C",   # ɝ
    6: "B",   # j, i, ɪ
    7: "F",   # w, u
    8: "E",   # o
    9: "E",   # aʊ
    10: "E",  # ɔɪ
    11: "D",  # aɪ
    12: "C",  # h
    13: "F",  # ɹ
    14: "C",  # l
    15: "H",  # s, z
    16: "H",  # ʃ, tʃ, dʒ, ʒ
    17: "H",  # ð
    18: "G",  # f, v
    19: "C",  # d, t, n, θ
    20: "B",  # k, g, ŋ
    21: "A",  # p, b, m
}

VOICE_PRESETS: dict[str, str] = {
    # DragonHD — the configured default set. Noticeably more natural prosody than
    # standard neural voices, at the cost of narrower SSML support.
    "meera": "en-IN-Meera:DragonHDLatestNeural",
    "aarti": "en-IN-Aarti:DragonHDLatestNeural",
    "ava": "en-US-Ava:DragonHDLatestNeural",
    # Standard neural fallbacks. Kept because they support the full SSML surface
    # (express-as styles, prosody rate) and reliably emit viseme events, which
    # DragonHD voices do not always do.
    "Female_1": "en-IN-NeerjaNeural",
    "Female_2": "en-IN-AnanyaNeural",
    "Male_1": "en-IN-PrabhatNeural",
    "Male_2": "en-US-EricNeural",
    "Telugu_Female": "te-IN-ShrutiNeural",
    "Telugu_Male": "te-IN-MohanNeural",
    "Hindi_Female": "hi-IN-SwaraNeural",
}

LANGUAGE_VOICE: dict[str, str] = {
    "en": "en-IN-Meera:DragonHDLatestNeural",
    # No DragonHD Telugu/Hindi voice exists yet, so these stay on standard neural.
    "te": "te-IN-ShrutiNeural",
    "hi": "hi-IN-SwaraNeural",
}


def is_hd_voice(voice: str) -> bool:
    """DragonHD / HD voices need a reduced SSML payload.

    They reject or silently ignore `mstts:express-as` and `prosody`, and an
    unsupported element can fail the whole synthesis rather than degrade — so we
    detect them by name and emit plain SSML instead.
    """
    lowered = voice.lower()
    return "dragonhd" in lowered or ":hd" in lowered or lowered.endswith("hdlatestneural")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")
_EMOJI = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf" "\U0001f1e6-\U0001f1ff" "]+",
    flags=re.UNICODE,
)


class SpeechUnavailable(RuntimeError):
    pass


# Domain vocabulary the generic en-IN acoustic model reliably mis-hears. Scheme
# names and district aliases dominate real voter questions, so biasing toward them
# is the single cheapest accuracy win available on the STT side.
_STATIC_PHRASES: tuple[str, ...] = (
    "Amma Vodi", "Rythu Bharosa", "Aarogyasri", "Aarogya Aasara",
    "Jagananna Vidya Deevena", "Jagananna Vasathi Deevena", "Pension Kanuka",
    "Navaratnalu", "Rythu Bharosa Kendram", "Disha Act", "DWCRA",
    "Sachivalayam", "Grama Sachivalayam", "Ward Sachivalayam",
    "Village Volunteer", "Ward Volunteer", "Crop Cultivator Rights Card",
    "Polavaram", "Prakasam Barrage", "Nagarjuna Sagar",
    "gram panchayat", "mandal", "constituency", "assembly constituency",
    "ration card", "white ration card", "Aadhaar",
    "Primary Health Centre", "Direct Benefit Transfer",
    "minimum support price", "self help group",
)


@lru_cache(maxsize=1)
def campaign_phrase_list(limit: int = 500) -> tuple[str, ...]:
    """Scheme names + district names/aliases, deduplicated and length-capped.

    Built from the same gazetteer the retriever uses, so a phrase the STT is
    biased toward is always a phrase the index can actually match.
    """
    phrases: list[str] = list(_STATIC_PHRASES)
    try:
        from ingestion.metadata import DISTRICT_GAZETTEER

        for district, aliases in DISTRICT_GAZETTEER.items():
            phrases.append(district)
            phrases.extend(a for a in aliases if len(a) > 4)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Gazetteer unavailable for phrase list: %s", exc)

    seen: dict[str, None] = {}
    for phrase in phrases:
        cleaned = phrase.strip()
        # Azure ignores very short phrases and caps the list length.
        if len(cleaned) > 3 and cleaned.lower() not in {k.lower() for k in seen}:
            seen[cleaned] = None
        if len(seen) >= limit:
            break
    return tuple(seen)


@dataclass
class Viseme:
    offset_ms: float
    viseme_id: int

    @property
    def shape(self) -> str:
        return AZURE_VISEME_TO_SHAPE.get(self.viseme_id, "X")


@dataclass
class TTSResult:
    audio: bytes
    format: str = "wav"
    duration_s: float = 0.0
    visemes: list[Viseme] = field(default_factory=list)
    voice: str = ""
    latency_ms: float = 0.0
    path: Optional[Path] = None

    def audio_b64(self) -> str:
        return base64.b64encode(self.audio).decode("ascii")


@dataclass
class STTResult:
    text: str
    success: bool
    language: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------- text
def clean_for_speech(text: str) -> str:
    """Strip anything a TTS engine would read aloud as noise.

    Citation markers get special handling: `[1]` must be removed, not read as
    "bracket one", but the sentence spacing has to survive the removal.
    """
    text = _EMOJI.sub("", text)
    text = re.sub(r"\[\d{1,2}\]", "", text)                  # citation markers
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"`{1,3}(.+?)`{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[(.+?)\]\((?:.+?)\)", r"\1", text)      # markdown links
    text = re.sub(r"^\s*[-*+•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = text.replace("|", ", ").replace("#", "").replace(">", "")
    text = re.sub(r"\bRs\.?\s*", "rupees ", text)
    text = re.sub(r"\n\s*\n", ". ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()


def split_sentences(text: str, min_chars: int = 40) -> list[str]:
    """Split into TTS-sized units, merging fragments too short to synthesize well."""
    raw = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    out: list[str] = []
    for sentence in raw:
        if out and len(out[-1]) < min_chars:
            out[-1] = f"{out[-1]} {sentence}"
        else:
            out.append(sentence)
    return out


def voice_locale(voice: str) -> str:
    """Locale prefix of an Azure voice name, e.g. "en-IN-Meera:DragonHD…" → "en-IN"."""
    parts = voice.split("-")
    return f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else "en-IN"


def build_ssml(text: str, voice: str, rate: Optional[str] = None) -> str:
    """SSML for one utterance, shaped to what the chosen voice actually accepts.

    DragonHD voices ignore `mstts:express-as` and `prosody`; including them risks
    failing the request outright instead of degrading, so HD voices get plain
    SSML. Their default delivery is already conversational, which is the reason
    the rate bump exists for standard voices in the first place.
    """
    locale = voice_locale(voice)
    escaped = (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )

    if is_hd_voice(voice):
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="{locale}"><voice name="{voice}">{escaped}</voice></speak>'
        )

    rate = rate or settings.azure_tts_rate
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{locale}">'
        f'<voice name="{voice}">'
        f'<mstts:express-as style="friendly">'
        f'<prosody rate="{rate}">{escaped}</prosody>'
        f"</mstts:express-as></voice></speak>"
    )


# ============================================================================
class AzureSpeech:
    def __init__(
        self,
        key: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        self.key = key if key is not None else settings.azure_speech_key
        self.region = region or settings.azure_speech_region
        self.endpoint = settings.azure_speech_endpoint
        self._sdk = None
        self._lock = threading.Lock()
        self._unavailable = ""

    # ---------------------------------------------------------------- plumbing
    @property
    def sdk(self):
        if self._sdk is None:
            with self._lock:
                if self._sdk is None:
                    try:
                        import azure.cognitiveservices.speech as speechsdk

                        self._sdk = speechsdk
                    except ImportError as exc:  # pragma: no cover
                        self._unavailable = (
                            "azure-cognitiveservices-speech is not installed"
                        )
                        raise SpeechUnavailable(self._unavailable) from exc
        return self._sdk

    @property
    def is_configured(self) -> bool:
        return bool(self.key and self.region)

    def _speech_config(self, voice: Optional[str] = None):
        if not self.is_configured:
            raise SpeechUnavailable(
                "AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are not configured"
            )
        speechsdk = self.sdk
        if self.endpoint:
            config = speechsdk.SpeechConfig(subscription=self.key, endpoint=self.endpoint)
        else:
            config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        if voice:
            config.speech_synthesis_voice_name = voice
        # Riff 16k mono PCM: what Rhubarb wants and what the browser plays
        # natively, so no transcode step in the hot path.
        config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
        )
        return config

    # -------------------------------------------------------------------- TTS
    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        language: str = "en",
        capture_visemes: bool = True,
    ) -> TTSResult:
        speechsdk = self.sdk
        resolved_voice = self._resolve_voice(voice, language)
        config = self._speech_config(resolved_voice)

        visemes: list[Viseme] = []
        # Pull audio into memory rather than a temp file — a disk round-trip per
        # sentence is pure added latency in a streaming voice loop.
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)

        if capture_visemes:
            def _on_viseme(evt) -> None:
                # Azure ticks are 100 ns units.
                visemes.append(
                    Viseme(offset_ms=evt.audio_offset / 10000.0, viseme_id=evt.viseme_id)
                )

            synthesizer.viseme_received.connect(_on_viseme)

        ssml = build_ssml(text, resolved_voice, rate)
        start = time.perf_counter()
        result = synthesizer.speak_ssml_async(ssml).get()
        latency_ms = (time.perf_counter() - start) * 1000
        METRICS.observe("tts.synthesize", latency_ms)

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio = bytes(result.audio_data)
            return TTSResult(
                audio=audio,
                duration_s=_wav_duration(audio),
                visemes=visemes,
                voice=resolved_voice,
                latency_ms=round(latency_ms, 2),
            )

        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            raise SpeechUnavailable(
                f"TTS canceled: {details.reason} — {details.error_details}"
            )
        raise SpeechUnavailable(f"TTS failed: {result.reason}")

    def synthesize_sentences(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        language: str = "en",
    ) -> list[TTSResult]:
        """One TTSResult per sentence, so playback can begin on the first."""
        sentences = split_sentences(text)
        results: list[TTSResult] = []
        for sentence in sentences:
            try:
                results.append(
                    self.synthesize(
                        sentence, voice=voice, rate=rate, language=language
                    )
                )
            except SpeechUnavailable as exc:
                logger.warning("Sentence TTS failed (%s): %s", sentence[:40], exc)
        return results

    def synthesize_to_file(
        self,
        text: str,
        path: Path,
        *,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        language: str = "en",
    ) -> TTSResult:
        result = self.synthesize(text, voice=voice, rate=rate, language=language)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(result.audio)
        result.path = path
        return result

    def _resolve_voice(self, voice: Optional[str], language: str) -> str:
        if voice:
            # Accept both a preset key ("Female_1") and a raw Azure voice name.
            return VOICE_PRESETS.get(voice, voice)
        if language and language != "en":
            return LANGUAGE_VOICE.get(language, settings.azure_tts_voice)
        return settings.azure_tts_voice

    # -------------------------------------------------------------------- STT
    def transcribe(
        self,
        audio: bytes,
        *,
        language: Optional[str] = None,
        auto_detect: bool = True,
        source_format: str = "auto",
    ) -> STTResult:
        """Transcribe an audio blob. Converts to 16 kHz mono WAV if needed."""
        speechsdk = self.sdk
        start = time.perf_counter()

        wav_bytes, warning = ensure_wav_16k_mono(audio, source_format=source_format)
        if wav_bytes is None:
            return STTResult(
                text="",
                success=False,
                error=warning or "Could not decode the submitted audio",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        # Azure's file recognizer needs a real path, and it keeps the handle open
        # until the recognizer is released. Write into our own temp dir rather than
        # NamedTemporaryFile so cleanup is ours to control, and never let a failed
        # delete propagate — on Windows the SDK can still hold the file when we
        # try to remove it, which previously raised
        # `WinError 32: file is being used by another process` and failed the
        # whole transcription *after* it had already succeeded.
        temp_dir = settings.tts_audio_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"stt_{uuid.uuid4().hex[:12]}.wav"
        temp_path.write_bytes(wav_bytes)

        recognizer = None
        try:
            config = self._speech_config()
            audio_config = speechsdk.audio.AudioConfig(filename=str(temp_path))

            if auto_detect and len(settings.stt_language_list) > 1:
                # Auto-detect accepts at most 4 candidates.
                detector = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                    languages=settings.stt_language_list[:4]
                )
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=config,
                    audio_config=audio_config,
                    auto_detect_source_language_config=detector,
                )
            else:
                config.speech_recognition_language = language or settings.azure_stt_language
                recognizer = speechsdk.SpeechRecognizer(
                    speech_config=config, audio_config=audio_config
                )

            self._apply_phrase_list(recognizer)

            result = recognizer.recognize_once()
            latency_ms = (time.perf_counter() - start) * 1000
            METRICS.observe("stt.recognize", latency_ms)

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                detected: Optional[str] = None
                try:
                    detected = speechsdk.AutoDetectSourceLanguageResult(result).language
                except Exception:  # noqa: BLE001
                    detected = language or settings.azure_stt_language
                return STTResult(
                    text=result.text,
                    success=True,
                    language=detected,
                    latency_ms=round(latency_ms, 2),
                )

            if result.reason == speechsdk.ResultReason.NoMatch:
                return STTResult(
                    text="",
                    success=False,
                    error="No speech recognized — speak a little louder or longer",
                    latency_ms=round(latency_ms, 2),
                )

            details = getattr(result, "cancellation_details", None)
            return STTResult(
                text="",
                success=False,
                error=f"Recognition canceled: {getattr(details, 'error_details', result.reason)}",
                latency_ms=round(latency_ms, 2),
            )
        finally:
            # Release the SDK's handle before deleting, then delete best-effort.
            # A leftover temp file is a trivial problem; losing a successful
            # transcription to a cleanup error is not.
            recognizer = None
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Could not remove STT temp file %s: %s", temp_path, exc)

    # ------------------------------------------------------------ phrase list
    def _apply_phrase_list(self, recognizer) -> None:
        """Bias recognition toward campaign vocabulary.

        Measured problem: a clean TTS render of "Amma Vodi pays fifteen thousand
        rupees" came back from STT as **"Amma oodipes ₹15,000"**. The generic
        en-IN model has no reason to know Telugu scheme names, yet those are
        precisely the highest-value query terms — a mangled scheme name means the
        BM25 branch matches nothing and retrieval falls back to fuzzy semantics.

        A phrase list is a cheap, no-training fix: it raises the prior on these
        exact strings without constraining the model to them. Costs nothing at
        runtime and needs no custom-model deployment.
        """
        try:
            speechsdk = self.sdk
            grammar = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
            for phrase in campaign_phrase_list():
                grammar.addPhrase(phrase)
        except Exception as exc:  # noqa: BLE001 — biasing is best-effort
            logger.debug("Could not apply phrase list: %s", exc)

    # ----------------------------------------------------------------- health
    def health(self) -> dict[str, Any]:
        if not self.is_configured:
            return {
                "status": "disabled",
                "detail": "AZURE_SPEECH_KEY / AZURE_SPEECH_REGION not set",
            }
        try:
            self.sdk  # noqa: B018 — import check
        except SpeechUnavailable as exc:
            return {"status": "down", "detail": str(exc)}
        return {
            "status": "ok",
            "region": self.region,
            "voice": settings.azure_tts_voice,
            "stt_languages": settings.stt_language_list,
            "ffmpeg": bool(_find_ffmpeg()),
        }


# ---------------------------------------------------------------- audio utils
def _wav_duration(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or 16000
            return round(frames / float(rate), 3)
    except Exception:  # noqa: BLE001
        return 0.0


def _find_ffmpeg() -> Optional[str]:
    if settings.ffmpeg_path and Path(settings.ffmpeg_path).exists():
        return settings.ffmpeg_path
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = [
        Path.home() / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe",
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"),
        Path("/usr/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
        Path("/opt/homebrew/bin/ffmpeg"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _is_wav_16k_mono(audio: bytes) -> bool:
    try:
        with wave.open(io.BytesIO(audio), "rb") as handle:
            return (
                handle.getnchannels() == 1
                and handle.getframerate() == 16000
                and handle.getsampwidth() == 2
            )
    except Exception:  # noqa: BLE001
        return False


def ensure_wav_16k_mono(
    audio: bytes, source_format: str = "auto"
) -> tuple[Optional[bytes], Optional[str]]:
    """Return 16 kHz mono PCM WAV bytes, converting via ffmpeg when required.

    Browsers record WebM/Opus, which Azure's file recognizer cannot read. ffmpeg
    is therefore a hard requirement for browser-captured audio — we say so
    explicitly instead of failing with an opaque SDK error.
    """
    if audio[:4] == b"RIFF" and _is_wav_16k_mono(audio):
        return audio, None

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return None, (
            "ffmpeg not found. Browser audio is WebM/Opus and must be transcoded "
            "to 16 kHz mono WAV. Install ffmpeg or set FFMPEG_PATH."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        suffix = {"webm": ".webm", "ogg": ".ogg", "mp3": ".mp3", "wav": ".wav"}.get(
            source_format, ".bin"
        )
        src = Path(tmpdir) / f"in{suffix}"
        dst = Path(tmpdir) / "out.wav"
        src.write_bytes(audio)
        try:
            subprocess.run(
                [
                    ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(src),
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(dst),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except subprocess.CalledProcessError as exc:
            return None, f"ffmpeg failed: {exc.stderr.decode('utf-8', 'replace')[:300]}"
        except subprocess.TimeoutExpired:
            return None, "ffmpeg timed out after 30s"
        return dst.read_bytes(), None


_speech: Optional[AzureSpeech] = None


def get_speech() -> AzureSpeech:
    global _speech
    if _speech is None:
        _speech = AzureSpeech()
    return _speech
