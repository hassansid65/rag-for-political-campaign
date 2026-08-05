"""
Voice endpoints: STT, TTS, a one-shot voice turn, and the streaming WebSocket.

`/voice/turn` is the HTTP-friendly path — one request in, answer + audio +
lip-sync out. It matches the `/lip-sync` contract from ds-catalogue-bot so the
existing avatar component works unchanged.

`/ws/voice` is the low-latency path and where speculative retrieval actually pays
off: the client streams `partial` transcripts while the user is still speaking, we
retrieve against them speculatively, and on `final` we reuse that work.
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from typing import Annotated, Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from core.config import settings
from core.latency import Trace
from core.schemas import (
    LipSyncPayload,
    STTResponse,
    TTSRequest,
    VoiceTurnRequest,
    VoiceTurnResponse,
)
from ingestion.metadata import resolve_district
from memory.conversation import get_session_store
from voice.azure_speech import (
    SpeechUnavailable,
    VOICE_PRESETS,
    clean_for_speech,
    get_speech,
)
from voice.lipsync import build_lipsync
from voice.streaming import StreamingRetriever, VoiceTurnPipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])


# --------------------------------------------------------------------- models
class STTBase64Request(BaseModel):
    audio: str = Field(..., description="Base64-encoded audio (webm/opus, wav, mp3)")
    format: str = "auto"
    language: Optional[str] = None
    auto_detect: bool = True


# ------------------------------------------------------------------------ STT
@router.post("/voice/stt", response_model=STTResponse)
async def speech_to_text(request: STTBase64Request) -> STTResponse:
    """Transcribe base64 audio. Browser WebM/Opus is transcoded via ffmpeg."""
    speech = get_speech()
    if not speech.is_configured:
        raise HTTPException(
            status_code=503,
            detail="Azure Speech is not configured (set AZURE_SPEECH_KEY / AZURE_SPEECH_REGION)",
        )

    try:
        audio = base64.b64decode(request.audio, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}") from exc

    if len(audio) < 1024:
        return STTResponse(text="", success=False, error="Recording is too short")

    try:
        result = speech.transcribe(
            audio,
            language=request.language,
            auto_detect=request.auto_detect,
            source_format=request.format,
        )
    except SpeechUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return STTResponse(
        text=result.text,
        success=result.success,
        language=result.language,
        duration_ms=result.latency_ms,
        error=result.error,
    )


@router.post("/voice/stt-file", response_model=STTResponse)
async def speech_to_text_file(
    file: Annotated[UploadFile, File()],
    language: Annotated[Optional[str], Form()] = None,
) -> STTResponse:
    """Multipart variant of /voice/stt — handy for curl and Postman."""
    speech = get_speech()
    if not speech.is_configured:
        raise HTTPException(status_code=503, detail="Azure Speech is not configured")

    data = await file.read()
    await file.close()
    suffix = (file.filename or "").rsplit(".", 1)[-1].lower() if file.filename else "auto"
    result = speech.transcribe(data, language=language, source_format=suffix)
    return STTResponse(
        text=result.text,
        success=result.success,
        language=result.language,
        duration_ms=result.latency_ms,
        error=result.error,
    )


# ------------------------------------------------------------------------ TTS
@router.post("/voice/tts")
async def text_to_speech(request: TTSRequest) -> dict:
    """Synthesize speech and return base64 WAV plus lip-sync cues."""
    speech = get_speech()
    if not speech.is_configured:
        raise HTTPException(status_code=503, detail="Azure Speech is not configured")

    spoken = clean_for_speech(request.text)
    if not spoken:
        raise HTTPException(status_code=400, detail="Nothing speakable in `text`")

    try:
        result = speech.synthesize(spoken, voice=request.voice, rate=request.rate)
    except SpeechUnavailable as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    lipsync = build_lipsync(
        text=spoken, duration_s=result.duration_s, visemes=result.visemes
    )
    return {
        "text": request.text,
        "spoken_text": spoken,
        "audio": result.audio_b64(),
        "audio_format": "wav",
        "duration_s": result.duration_s,
        "voice": result.voice,
        "lipsync": lipsync,
        "tts_ms": result.latency_ms,
    }


@router.get("/voice/voices")
async def voices() -> dict:
    """Voice presets available to the UI."""
    return {
        "presets": VOICE_PRESETS,
        "default": settings.azure_tts_voice,
        "stt_languages": settings.stt_language_list,
    }


# ----------------------------------------------------------------- voice turn
@router.post("/voice/turn", response_model=VoiceTurnResponse)
async def voice_turn(request: VoiceTurnRequest) -> VoiceTurnResponse:
    """One complete voice turn: retrieve → generate → synthesize → lip-sync.

    Wire-compatible with the `/lip-sync` endpoint the avatar component already
    speaks, so the frontend needs no changes to animate this.
    """
    from llm.rag_service import get_rag_service

    trace = Trace(name="voice_turn_http")
    service = get_rag_service()

    filters = None
    if request.filters:
        filters = request.filters.model_dump(exclude_none=True)
        if filters.get("district"):
            filters["district"] = resolve_district(filters["district"]) or filters["district"]
        filters = {k: v for k, v in filters.items() if v not in (None, "", [], {})} or None

    result = await service.answer(
        request.message,
        session_id=request.session_id,
        filters=filters,
        voice_mode=True,
        trace=trace,
    )

    spoken = clean_for_speech(result.answer)
    audio_b64: Optional[str] = None
    lipsync_payload: Optional[LipSyncPayload] = None

    speech = get_speech()
    if speech.is_configured and spoken:
        session = get_session_store().get(request.session_id)
        try:
            with trace.stage("tts"):
                tts = speech.synthesize(
                    spoken, voice=request.voice, language=session.language
                )
            audio_b64 = tts.audio_b64()
            with trace.stage("lipsync"):
                lipsync_payload = LipSyncPayload(
                    **build_lipsync(
                        text=spoken, duration_s=tts.duration_s, visemes=tts.visemes
                    )
                )
        except SpeechUnavailable as exc:
            # Text without audio is still a usable answer; don't 500 the turn.
            logger.warning("Voice turn TTS failed: %s", exc)

    return VoiceTurnResponse(
        text=result.answer,
        spoken_text=spoken,
        session_id=result.session_id,
        audio=audio_b64,
        lipsync=lipsync_payload,
        citations=result.citations if request.include_citations else [],
        grounded=result.grounded,
        facialExpression="default",
        animation="Talking" if audio_b64 else "Idle",
        timings_ms={**trace.finish(), "notes": result.notes},
    )


# Legacy alias so the ds-catalogue-bot frontend contract keeps working verbatim.
@router.post("/lip-sync", response_model=VoiceTurnResponse, include_in_schema=False)
async def lip_sync_alias(request: VoiceTurnRequest) -> VoiceTurnResponse:
    return await voice_turn(request)


# ------------------------------------------------------------------ WebSocket
@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket) -> None:
    """Streaming voice channel.

    Client → server messages:
        {"type": "partial", "text": "what is amma vodi"}   (fires speculation)
        {"type": "final",   "text": "...", "voice": "Female_1"}
        {"type": "cancel"}                                 (barge-in)
        {"type": "ping"}

    Server → client events:
        ready · speculation · retrieval · delta · audio · final · error · pong

    `partial` is the whole point of this endpoint: it lets retrieval overlap with
    the user still talking. See voice/streaming.py for the reuse logic.
    """
    await websocket.accept()
    session_id = websocket.query_params.get("session_id") or uuid.uuid4().hex[:12]
    session = get_session_store().get(session_id)

    retriever = StreamingRetriever(session=session)
    turn_pipeline = VoiceTurnPipeline()

    await websocket.send_json(
        {
            "type": "ready",
            "session_id": session_id,
            "district": session.district,
            "speech_configured": get_speech().is_configured,
            "partial_min_chars": settings.partial_min_chars,
        }
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "error": "malformed JSON"})
                continue

            kind = message.get("type")

            if kind == "ping":
                await websocket.send_json({"type": "pong"})

            elif kind == "partial":
                text = (message.get("text") or "").strip()
                fired = await retriever.on_partial(text)
                if fired:
                    await websocket.send_json(
                        {"type": "speculation", "state": "fired", "text": text}
                    )

            elif kind == "cancel":
                retriever.cancel()
                await websocket.send_json({"type": "speculation", "state": "cancelled"})

            elif kind == "final":
                text = (message.get("text") or "").strip()
                if not text:
                    await websocket.send_json({"type": "error", "error": "empty transcript"})
                    continue

                filters = message.get("filters")
                if isinstance(filters, dict) and filters.get("district"):
                    filters["district"] = (
                        resolve_district(filters["district"]) or filters["district"]
                    )

                try:
                    async for event in turn_pipeline.run(
                        text,
                        session_id=session_id,
                        retriever=retriever,
                        voice=message.get("voice"),
                        filters=filters if isinstance(filters, dict) else None,
                        synthesize=bool(message.get("synthesize", True)),
                    ):
                        await websocket.send_json(event)
                except WebSocketDisconnect:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("Voice turn failed: %s", exc, exc_info=True)
                    await websocket.send_json({"type": "error", "error": str(exc)})

            elif kind == "reset":
                get_session_store().reset(session_id)
                session = get_session_store().get(session_id)
                retriever = StreamingRetriever(session=session)
                await websocket.send_json({"type": "ready", "session_id": session_id})

            else:
                await websocket.send_json(
                    {"type": "error", "error": f"unknown message type: {kind!r}"}
                )

    except WebSocketDisconnect:
        logger.info("Voice WebSocket disconnected (session=%s)", session_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Voice WebSocket error: %s", exc, exc_info=True)
        try:
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001
            pass
    finally:
        retriever.cancel()
        logger.info(
            "Voice session %s speculation stats: %s", session_id, retriever.stats_dict()
        )
