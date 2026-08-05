/**
 * Browser voice capture + the streaming voice WebSocket.
 *
 * Two capture strategies, and the choice matters for latency:
 *
 * 1. **Web Speech API (preferred when present).** Chrome/Edge expose
 *    `webkitSpeechRecognition`, which emits *interim* results while the user is
 *    still speaking. Those interim transcripts are what we forward as `partial`
 *    messages so the backend can retrieve speculatively. This is the only path
 *    that unlocks the overlap — no interim results, no speculation.
 * 2. **MediaRecorder → Azure STT (fallback).** Records WebM/Opus, detects
 *    end-of-speech via RMS silence, then posts the blob to /voice/stt. Correct
 *    everywhere, but strictly serial: nothing can start until the user stops.
 *
 * Both paths converge on the same `final` message, so the backend and the avatar
 * behave identically either way.
 */

import { BACKEND_URL } from './api';
import type { Citation, MouthCue, RetrievedChunk } from './api';

// -------------------------------------------------------------------- WS events
export type VoiceServerEvent =
  | { type: 'ready'; session_id: string; district: string | null; speech_configured: boolean; partial_min_chars: number }
  | { type: 'speculation'; state?: string; text?: string; stats?: Record<string, unknown>; notes?: string[] }
  | { type: 'retrieval'; sources: Citation[]; effective_query: string; district: string | null; timings_ms: Record<string, unknown> }
  | { type: 'delta'; text: string }
  | { type: 'sentence'; text: string; audio?: null; error?: string }
  | { type: 'audio'; text: string; audio: string; audio_format: string; duration_s: number; lipsync: { mouthCues: MouthCue[]; metadata: Record<string, unknown> }; voice: string; tts_ms: number }
  | { type: 'final'; answer: string; spoken_text: string; grounded: boolean; citations: Citation[]; session_id: string; district: string | null; timings_ms: Record<string, unknown>; notes: string[] }
  | { type: 'error'; error: string }
  | { type: 'pong' };

export class VoiceSocket {
  private socket: WebSocket | null = null;
  private readonly handlers = new Set<(event: VoiceServerEvent) => void>();
  private heartbeat: ReturnType<typeof setInterval> | null = null;
  private reconnectAttempts = 0;
  private closedByUs = false;

  constructor(private readonly sessionId: string) {}

  get isOpen(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = `${BACKEND_URL.replace(/^http/, 'ws')}/ws/voice?session_id=${encodeURIComponent(
        this.sessionId
      )}`;
      const socket = new WebSocket(wsUrl);
      this.socket = socket;
      this.closedByUs = false;

      socket.onopen = () => {
        this.reconnectAttempts = 0;
        // Idle WS connections get culled by proxies at ~60s; ping well inside that.
        this.heartbeat = setInterval(() => this.send({ type: 'ping' }), 25_000);
        resolve();
      };

      socket.onmessage = (raw) => {
        try {
          const event = JSON.parse(raw.data) as VoiceServerEvent;
          if (event.type === 'pong') return;
          this.handlers.forEach((handler) => handler(event));
        } catch {
          /* ignore malformed frame */
        }
      };

      socket.onerror = () => reject(new Error('Voice socket failed to connect'));

      socket.onclose = () => {
        if (this.heartbeat) clearInterval(this.heartbeat);
        this.heartbeat = null;
        if (!this.closedByUs && this.reconnectAttempts < 3) {
          this.reconnectAttempts += 1;
          setTimeout(() => this.connect().catch(() => undefined), 800 * this.reconnectAttempts);
        }
      };
    });
  }

  on(handler: (event: VoiceServerEvent) => void): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  private send(payload: Record<string, unknown>): void {
    if (this.isOpen) this.socket!.send(JSON.stringify(payload));
  }

  /** Forward an interim transcript so the backend can retrieve speculatively. */
  sendPartial(text: string): void {
    this.send({ type: 'partial', text });
  }

  sendFinal(text: string, options?: { voice?: string; filters?: Record<string, unknown>; synthesize?: boolean }): void {
    this.send({ type: 'final', text, ...options });
  }

  cancel(): void {
    this.send({ type: 'cancel' });
  }

  reset(): void {
    this.send({ type: 'reset' });
  }

  close(): void {
    this.closedByUs = true;
    if (this.heartbeat) clearInterval(this.heartbeat);
    this.socket?.close();
    this.socket = null;
  }
}

// ------------------------------------------------------- Web Speech recognition
type SpeechRecognitionCtor = new () => any;

function getSpeechRecognition(): SpeechRecognitionCtor | null {
  if (typeof window === 'undefined') return null;
  const w = window as unknown as Record<string, SpeechRecognitionCtor | undefined>;
  return w.SpeechRecognition || w.webkitSpeechRecognition || null;
}

export function supportsInterimSpeech(): boolean {
  return getSpeechRecognition() !== null;
}

export interface RecognizerCallbacks {
  onPartial?: (text: string) => void;
  onFinal?: (text: string) => void;
  onError?: (message: string) => void;
  onEnd?: () => void;
}

/**
 * Interim-capable recognizer. Returns a stop() handle, or null if unsupported.
 */
export function startInterimRecognition(
  callbacks: RecognizerCallbacks,
  language = 'en-IN'
): (() => void) | null {
  const Recognition = getSpeechRecognition();
  if (!Recognition) return null;

  const recognition = new Recognition();
  recognition.lang = language;
  recognition.continuous = false;
  recognition.interimResults = true;   // the whole point — enables speculation
  recognition.maxAlternatives = 1;

  let finalText = '';
  let stopped = false;

  recognition.onresult = (event: any) => {
    let interim = '';
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const result = event.results[i];
      if (result.isFinal) finalText += result[0].transcript;
      else interim += result[0].transcript;
    }
    const combined = (finalText + interim).trim();
    if (combined) callbacks.onPartial?.(combined);
  };

  recognition.onerror = (event: any) => {
    // 'no-speech' and 'aborted' are normal end-of-turn conditions, not failures.
    if (event.error !== 'no-speech' && event.error !== 'aborted') {
      callbacks.onError?.(String(event.error));
    }
  };

  recognition.onend = () => {
    if (stopped) return;
    stopped = true;
    const text = finalText.trim();
    if (text) callbacks.onFinal?.(text);
    callbacks.onEnd?.();
  };

  try {
    recognition.start();
  } catch (error) {
    callbacks.onError?.(String(error));
    return null;
  }

  return () => {
    stopped = false;
    try {
      recognition.stop();
    } catch {
      /* already stopped */
    }
  };
}

// -------------------------------------------------- MediaRecorder + Azure STT
export interface MediaRecorderHandle {
  stop: (send?: boolean) => void;
  stream: MediaStream;
}

/**
 * Record until silence, then hand the blob to the caller as base64.
 *
 * The silence detector is a plain RMS threshold over an AnalyserNode. It is not
 * a VAD, but for a push-to-talk-style interaction it is enough, and it avoids
 * the user having to press stop.
 */
export async function recordUntilSilence(options: {
  onComplete: (base64: string) => void;
  onError?: (message: string) => void;
  onLevel?: (level: number) => void;
  silenceMs?: number;
  minMs?: number;
}): Promise<MediaRecorderHandle> {
  const { onComplete, onError, onLevel, silenceMs = 1500, minMs = 400 } = options;

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
  });

  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  source.connect(analyser);

  const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus'
    : 'audio/webm';
  const recorder = new MediaRecorder(stream, { mimeType });
  const chunks: Blob[] = [];
  const startedAt = Date.now();

  let silenceTimer: ReturnType<typeof setTimeout> | null = null;
  let finished = false;

  const teardown = () => {
    stream.getTracks().forEach((track) => track.stop());
    audioContext.close().catch(() => undefined);
    if (silenceTimer) clearTimeout(silenceTimer);
  };

  const monitor = () => {
    if (finished || recorder.state !== 'recording') return;
    const buffer = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(buffer);
    const average = buffer.reduce((sum, value) => sum + value, 0) / buffer.length;
    onLevel?.(average / 255);

    if (average < 10) {
      if (!silenceTimer) {
        silenceTimer = setTimeout(() => {
          if (Date.now() - startedAt >= minMs) stopRecording(true);
        }, silenceMs);
      }
    } else if (silenceTimer) {
      clearTimeout(silenceTimer);
      silenceTimer = null;
    }
    requestAnimationFrame(monitor);
  };

  function stopRecording(send: boolean): void {
    if (finished) return;
    finished = true;
    if (recorder.state !== 'inactive') recorder.stop();
    if (!send) teardown();
  }

  recorder.ondataavailable = (event) => {
    if (event.data.size > 0) chunks.push(event.data);
  };

  recorder.onstop = () => {
    teardown();
    if (Date.now() - startedAt < minMs) {
      onError?.('Recording was too short — please speak for at least a second.');
      return;
    }
    const blob = new Blob(chunks, { type: mimeType });
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      const base64 = result.split(',')[1];
      if (base64) onComplete(base64);
      else onError?.('Could not encode the recording.');
    };
    reader.onerror = () => onError?.('Failed to read the recording.');
    reader.readAsDataURL(blob);
  };

  recorder.start();
  monitor();

  return { stop: (send = true) => stopRecording(send), stream };
}

// ------------------------------------------------------------ audio + lip-sync
/**
 * Sequential audio queue.
 *
 * The backend streams one audio clip per sentence, so clips arrive while an
 * earlier one is still playing. Queueing keeps them in order and keeps the
 * avatar's mouth driven by exactly the clip that is audible.
 */
export interface AudioClip {
  url: string;
  cues: MouthCue[];
  /** The sentence this clip speaks — revealed in the transcript as it plays. */
  text?: string;
  /** Clip length in seconds, from the TTS response. */
  durationS?: number;
}

export class AudioQueue {
  private readonly queue: AudioClip[] = [];
  private current: HTMLAudioElement | null = null;
  private frame: number | null = null;
  private playing = false;

  constructor(
    private readonly onShape: (shape: string) => void,
    private readonly onStateChange?: (speaking: boolean) => void,
    /**
     * Fired the instant a clip begins playing, with its text and real duration.
     *
     * This is the hook that keeps captions in step with speech: the UI reveals a
     * sentence only once its audio is audible, paced over the clip's own length.
     */
    private readonly onClipStart?: (clip: AudioClip, durationS: number) => void
  ) {}

  enqueue(clip: AudioClip): void {
    this.queue.push(clip);
    if (!this.playing) void this.playNext();
  }

  private async playNext(): Promise<void> {
    const next = this.queue.shift();
    if (!next) {
      this.playing = false;
      this.onShape('X');
      this.onStateChange?.(false);
      return;
    }

    this.playing = true;
    this.onStateChange?.(true);

    const audio = new Audio(next.url);
    this.current = audio;
    this.animate(audio, next.cues);

    await new Promise<void>((resolve) => {
      let announced = false;
      const announce = () => {
        if (announced) return;
        announced = true;
        // Prefer the element's decoded duration; fall back to the server's value,
        // since `duration` can still be NaN on the first play event.
        const measured = Number.isFinite(audio.duration) && audio.duration > 0
          ? audio.duration
          : next.durationS ?? 0;
        this.onClipStart?.(next, measured);
      };

      audio.onloadedmetadata = announce;
      audio.onplay = announce;
      audio.onended = () => resolve();
      audio.onerror = () => {
        announce();   // still reveal the text if the audio failed
        resolve();
      };
      audio.play().catch(() => {
        announce();
        resolve();
      });
    });

    URL.revokeObjectURL(next.url);
    if (this.frame) cancelAnimationFrame(this.frame);
    void this.playNext();
  }

  private animate(audio: HTMLAudioElement, cues: MouthCue[]): void {
    let index = 0;
    const tick = () => {
      if (audio.paused || audio.ended) {
        this.onShape('X');
        return;
      }
      const t = audio.currentTime;
      while (index < cues.length - 1 && t >= cues[index + 1].start) index += 1;
      const cue = cues[index];
      if (cue && t >= cue.start && t <= cue.end) this.onShape(cue.value);
      this.frame = requestAnimationFrame(tick);
    };
    this.frame = requestAnimationFrame(tick);
  }

  stop(): void {
    this.queue.length = 0;
    if (this.frame) cancelAnimationFrame(this.frame);
    if (this.current) {
      this.current.pause();
      this.current = null;
    }
    this.playing = false;
    this.onShape('X');
    this.onStateChange?.(false);
  }
}
