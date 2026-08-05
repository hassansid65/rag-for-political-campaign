'use client';

/**
 * Main assistant shell.
 *
 * Layout mirrors ds-catalogue-bot: dark 3D avatar panel on the left (35%), chat
 * on the right (65%), collapsible tool sidebar, rounded input pill.
 *
 * The interaction model is the substantive difference. There are two paths:
 *
 *  * **Text** → SSE `/query`. Sources render from the first `retrieval` event,
 *    then tokens stream into the bubble.
 *  * **Voice** → WebSocket `/ws/voice`. Interim transcripts are forwarded as
 *    `partial` messages so the backend retrieves speculatively while the citizen
 *    is still speaking; audio arrives per sentence and is queued so the avatar's
 *    mouth is always driven by the clip that is actually audible.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Info,
  Megaphone,
  Mic,
  MicOff,
  Radio,
  Send,
  SlidersHorizontal,
  Trash2,
  Volume2,
  VolumeX,
  Zap,
} from 'lucide-react';
import Avatar, { type AvatarHandle } from './Avatar';
import MessageBubble from './MessageBubble';
import SourceCard from './SourceCard';
import SidePanel from './SidePanel';
import {
  api,
  base64ToAudioUrl,
  queryStream,
  type Citation,
  type MouthCue,
  type RetrieveFilters,
  type StreamEvent,
} from '@/lib/api';
import { SentenceSplitter, Typewriter, splitIntoSentences } from '@/lib/streaming';
import {
  AudioQueue,
  VoiceSocket,
  recordUntilSilence,
  startInterimRecognition,
  supportsInterimSpeech,
  type AudioClip,
  type VoiceServerEvent,
} from '@/lib/voice';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  citations?: Citation[];
  grounded?: boolean;
  streaming?: boolean;
}

function newId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [filters, setFilters] = useState<RetrieveFilters>({});
  const [showPanel, setShowPanel] = useState(false);
  const [showSources, setShowSources] = useState(true);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [detectedDistrict, setDetectedDistrict] = useState<string | null>(null);
  const [speculation, setSpeculation] = useState<string | null>(null);
  const [statusNote, setStatusNote] = useState<string | null>(null);

  const sessionId = useRef<string>('');
  const avatarRef = useRef<AvatarHandle>(null);
  const audioQueue = useRef<AudioQueue | null>(null);
  const socket = useRef<VoiceSocket | null>(null);
  const stopRecognition = useRef<(() => void) | null>(null);
  const stopRecorder = useRef<((send?: boolean) => void) | null>(null);
  const scrollAnchor = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const typewriterRef = useRef<Typewriter | null>(null);

  // Session id must be stable across renders but generated on the client only,
  // or SSR and hydration disagree.
  if (!sessionId.current) {
    sessionId.current = `web-${Math.random().toString(36).slice(2, 10)}`;
  }

  useEffect(() => {
    audioQueue.current = new AudioQueue(
      (shape) => avatarRef.current?.setMouthShape(shape),
      (speaking) => setIsSpeaking(speaking),
      // Reveal a sentence only when its audio is actually audible, paced over the
      // clip's real duration. Without this the text finishes typing 8-10s before
      // the avatar starts talking, because each TTS round-trip to Azure costs
      // 2-3s while the token stream lands in well under a second.
      (clip, durationS) => {
        if (clip.text) {
          typewriterRef.current?.pushOver(clip.text, durationS * 1000);
        }
      }
    );
    return () => audioQueue.current?.stop();
  }, []);

  useEffect(() => {
    scrollAnchor.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, liveTranscript]);

  // ------------------------------------------------------------- message utils
  const appendMessage = useCallback((message: Message) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  const patchMessage = useCallback((id: string, patch: Partial<Message>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }, []);

  // ----------------------------------------------------------------- speech out
  // Declared before the text path because `sendText` calls `speakSentence` as
  // sentences complete.
  /**
   * Synthesize one sentence and queue it for playback.
   *
   * Requests are issued the moment a sentence closes, so several can be in flight
   * at once — but playback must follow the order they were *requested*, not the
   * order they return. A short final sentence routinely synthesizes faster than a
   * long opening one, and without a sequence number the answer is spoken out of
   * order. `AudioQueue` serializes playback; this keeps the queue itself ordered.
   */
  const ttsSeq = useRef(0);
  const ttsNext = useRef(0);
  const ttsPending = useRef<Map<number, AudioClip | null>>(new Map());

  const resetTts = useCallback(() => {
    ttsSeq.current = 0;
    ttsNext.current = 0;
    ttsPending.current.clear();
  }, []);

  const drainTts = useCallback(() => {
    const pending = ttsPending.current;
    while (pending.has(ttsNext.current)) {
      const clip = pending.get(ttsNext.current);
      pending.delete(ttsNext.current);
      ttsNext.current += 1;
      if (clip) {
        audioQueue.current?.enqueue(clip);
      } else {
        // Synthesis failed for this sentence. Nothing will play, so nothing will
        // trigger the caption — reveal the text directly or the transcript stalls
        // permanently at the failed sentence.
        const stalled = typewriterRef.current;
        if (stalled) stalled.flush();
      }
    }
  }, []);

  const speakSentence = useCallback(
    async (text: string) => {
      const sentence = text.trim();
      if (!sentence) return;
      const seq = ttsSeq.current++;
      try {
        const result = await api.tts({ text: sentence });
        ttsPending.current.set(seq, {
          url: base64ToAudioUrl(result.audio),
          cues: result.lipsync.mouthCues,
          // Carried through so the caption reveals exactly what is being spoken.
          text: sentence,
          durationS: result.duration_s,
        });
      } catch {
        // Reserve the slot with null. Skipping it entirely would stall every later
        // sentence behind a sequence number that never fills.
        ttsPending.current.set(seq, null);
        setStatusNote('Text-to-speech is unavailable — check the Azure Speech key.');
      }
      drainTts();
    },
    [drainTts]
  );

  /** Speak a finished answer — the replay button on a completed bubble. */
  const speak = useCallback(
    async (text: string) => {
      resetTts();
      for (const sentence of splitIntoSentences(text)) {
        void speakSentence(sentence);
      }
    },
    [resetTts, speakSentence]
  );

  const stopSpeaking = useCallback(() => {
    audioQueue.current?.stop();
    avatarRef.current?.reset();
    resetTts();
    setIsSpeaking(false);
  }, [resetTts]);

  // ------------------------------------------------------------------ text path
  const sendText = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || loading) return;

      setInput('');
      setStatusNote(null);
      appendMessage({ id: newId(), text: trimmed, isUser: true, timestamp: new Date() });

      const assistantId = newId();
      appendMessage({
        id: assistantId,
        text: '',
        isUser: false,
        timestamp: new Date(),
        streaming: true,
      });
      setLoading(true);

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      let accumulated = '';

      // Arrival is lumpy; display should not be. Deltas feed a buffer that a rAF
      // loop reveals at a steady rate — that decoupling is what reads as
      // "writing" rather than "loading".
      const typer = new Typewriter((visible) =>
        patchMessage(assistantId, { text: visible })
      );
      typewriterRef.current = typer;

      // Speak sentence by sentence instead of waiting for the last token. First
      // audio then depends on the first sentence (~1/5 of the text), which is the
      // single biggest cut to perceived voice latency in the text path.
      const splitter = new SentenceSplitter();

      try {
        await queryStream(
          {
            query: trimmed,
            session_id: sessionId.current,
            filters: Object.keys(filters).length ? filters : undefined,
          },
          (event: StreamEvent) => {
            if (event.type === 'retrieval') {
              // Render source cards before generation finishes.
              patchMessage(assistantId, { citations: event.sources });
              const district = event.inferred_filters?.district;
              if (typeof district === 'string') setDetectedDistrict(district);
            } else if (event.type === 'delta') {
              accumulated += event.text;
              if (voiceEnabled) {
                // Do NOT drip here. With voice on, each sentence's text is
                // revealed by its own audio clip starting (see the AudioQueue
                // onClipStart hook) so caption and speech begin together.
                for (const sentence of splitter.push(accumulated)) {
                  void speakSentence(sentence);
                }
              } else {
                typer.push(event.text);
              }
            } else if (event.type === 'final') {
              patchMessage(assistantId, {
                citations: event.citations,
                grounded: event.grounded,
                streaming: false,
              });
              if (voiceEnabled) {
                splitter.push(event.answer);
                const tail = splitter.remainder();
                if (tail) void speakSentence(tail);
              } else {
                // The server's canonical text wins — it is sanitized and may
                // differ from the concatenated deltas. Keep dripping rather than
                // snapping, or the last words appear all at once.
                typer.set(event.answer);
              }
            } else if (event.type === 'error') {
              typer.flush();
              patchMessage(assistantId, {
                text: `Something went wrong: ${event.error}`,
                streaming: false,
                grounded: false,
              });
            }
          },
          controller.signal
        );
      } catch (error) {
        typer.stop();
        if ((error as Error).name !== 'AbortError') {
          patchMessage(assistantId, {
            text:
              'I could not reach the backend. Check that it is running on ' +
              `${process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'}.`,
            streaming: false,
            grounded: false,
          });
        }
      } finally {
        setLoading(false);
      }
    },
    [appendMessage, filters, loading, patchMessage, speakSentence, voiceEnabled]
  );

  // ------------------------------------------------------------- voice WS path
  //
  // The mic button was removed from the composer on request — it needs ffmpeg on
  // the backend to transcode the browser's WebM/Opus for Azure STT, and without it
  // every recording fails. Everything below is left intact and wired rather than
  // deleted, because `/ws/voice` is the only entry point to speculative retrieval
  // (partial transcripts overlapping retrieval with speech), which is a required
  // capability of this system. Re-enabling is one button calling `startVoice`.
  //
  // The outbound half — sentence-level TTS, viseme lip-sync, audio-paced captions
  // — is fully live via the text path and needs no microphone.
  const ensureSocket = useCallback(async (): Promise<VoiceSocket | null> => {
    if (socket.current?.isOpen) return socket.current;
    const ws = new VoiceSocket(sessionId.current);
    socket.current = ws;

    let assistantId = '';
    let accumulated = '';

    ws.on((event: VoiceServerEvent) => {
      switch (event.type) {
        case 'ready':
          if (event.district) setDetectedDistrict(event.district);
          if (!event.speech_configured) {
            setStatusNote('Azure Speech is not configured — replies will be text only.');
          }
          break;

        case 'speculation':
          if (event.state === 'fired') {
            setSpeculation('Searching while you speak…');
            setTimeout(() => setSpeculation(null), 1400);
          }
          if (event.stats) {
            const reused = Number(event.stats.reused ?? 0);
            if (reused > 0) {
              const saved = Math.round(Number(event.stats.estimated_saved_ms ?? 0));
              setSpeculation(`Reused pre-fetched results (saved ~${saved}ms)`);
              setTimeout(() => setSpeculation(null), 2600);
            }
          }
          break;

        case 'retrieval':
          assistantId = newId();
          accumulated = '';
          appendMessage({
            id: assistantId,
            text: '',
            isUser: false,
            timestamp: new Date(),
            citations: event.sources,
            streaming: true,
          });
          if (event.district) setDetectedDistrict(event.district);
          break;

        case 'delta':
          accumulated += event.text;
          if (assistantId) patchMessage(assistantId, { text: accumulated });
          break;

        case 'audio': {
          // One clip per sentence — queue so playback stays in order. The server
          // already sent the sentence text, so the caption is driven by playback
          // here too, exactly as in the text path.
          audioQueue.current?.enqueue({
            url: base64ToAudioUrl(event.audio),
            cues: event.lipsync.mouthCues,
            text: event.text,
            durationS: event.duration_s,
          });
          break;
        }

        case 'final':
          if (assistantId) {
            patchMessage(assistantId, {
              text: event.answer,
              citations: event.citations,
              grounded: event.grounded,
              streaming: false,
            });
          }
          setLoading(false);
          break;

        case 'error':
          setStatusNote(event.error);
          setLoading(false);
          break;

        default:
          break;
      }
    });

    try {
      await ws.connect();
      return ws;
    } catch {
      setStatusNote('Could not open the voice channel; falling back to text.');
      return null;
    }
  }, [appendMessage, patchMessage]);

  const startVoice = useCallback(async () => {
    setStatusNote(null);
    stopSpeaking();
    const ws = await ensureSocket();
    setIsRecording(true);
    setLiveTranscript('');

    const submit = (text: string) => {
      setIsRecording(false);
      setLiveTranscript('');
      const clean = text.trim();
      if (!clean) return;
      appendMessage({ id: newId(), text: clean, isUser: true, timestamp: new Date() });
      setLoading(true);
      if (ws?.isOpen) {
        ws.sendFinal(clean, {
          filters: Object.keys(filters).length ? (filters as Record<string, unknown>) : undefined,
          synthesize: voiceEnabled,
        });
      } else {
        void sendText(clean);
      }
    };

    // Preferred: interim results, which is what makes speculation possible.
    if (supportsInterimSpeech()) {
      stopRecognition.current = startInterimRecognition({
        onPartial: (text) => {
          setLiveTranscript(text);
          ws?.sendPartial(text);
        },
        onFinal: submit,
        onError: (message) => {
          setStatusNote(`Speech recognition error: ${message}`);
          setIsRecording(false);
        },
        onEnd: () => setIsRecording(false),
      });
      if (stopRecognition.current) return;
    }

    // Fallback: record → Azure STT. No interim results, so no speculation.
    try {
      const handle = await recordUntilSilence({
        onComplete: async (base64) => {
          try {
            const result = await api.stt(base64);
            if (result.success && result.text) submit(result.text);
            else {
              setIsRecording(false);
              setStatusNote(result.error || 'No speech was recognised.');
            }
          } catch (error) {
            setIsRecording(false);
            setStatusNote(error instanceof Error ? error.message : 'Transcription failed.');
          }
        },
        onError: (message) => {
          setIsRecording(false);
          setStatusNote(message);
        },
      });
      stopRecorder.current = handle.stop;
    } catch {
      setIsRecording(false);
      setStatusNote('Microphone access was denied.');
    }
  }, [appendMessage, ensureSocket, filters, sendText, stopSpeaking, voiceEnabled]);

  const stopVoice = useCallback(() => {
    stopRecognition.current?.();
    stopRecognition.current = null;
    stopRecorder.current?.(true);
    stopRecorder.current = null;
    setIsRecording(false);
  }, []);

  const clearConversation = useCallback(() => {
    abortRef.current?.abort();
    stopVoice();
    stopSpeaking();
    setMessages([]);
    setDetectedDistrict(null);
    setStatusNote(null);
    socket.current?.reset();
    void api.resetSession(sessionId.current).catch(() => undefined);
  }, [stopSpeaking, stopVoice]);

  useEffect(() => () => socket.current?.close(), []);

  // ------------------------------------------------------------------- render
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-surface-0 lg:flex-row">
      {/* ------------------------------------------------- avatar (left 35%) */}
      <div className="relative z-20 hidden h-full items-center justify-center border-r border-line bg-[#0a0c10] shadow-xl lg:flex lg:w-[35%]">
        <div className="absolute right-6 top-6 z-30 flex gap-2">
          <button
            onClick={() => {
              if (isSpeaking) stopSpeaking();
              setVoiceEnabled((value) => !value);
            }}
            className={`rounded-full border border-white/20 p-3 text-white shadow-lg backdrop-blur-xl transition-all hover:scale-105 ${
              voiceEnabled ? 'bg-brand-blue/80' : 'bg-surface-1/10 hover:bg-surface-2/20'
            }`}
            title={voiceEnabled ? 'Voice replies on' : 'Voice replies off'}
          >
            {voiceEnabled ? <Volume2 size={19} /> : <VolumeX size={19} />}
          </button>
        </div>

        <Avatar ref={avatarRef} isSpeaking={isSpeaking} isListening={isRecording} />

        <div className="absolute bottom-6 left-6 z-30 flex flex-col gap-1.5 text-white/50">
          <span className="text-[11px] font-semibold tracking-wider drop-shadow-md">
            POLITICAL CAMPAIGN ASSISTANT
          </span>
          {detectedDistrict && (
            <span className="text-[10px] font-bold uppercase tracking-wider text-teal-300/80">
              {detectedDistrict} district
            </span>
          )}
        </div>

        {speculation && (
          <div className="absolute left-1/2 top-6 z-30 -translate-x-1/2">
            <div className="flex items-center gap-1.5 rounded-full border border-white/20 bg-surface-1/10 px-3 py-1.5 backdrop-blur-xl">
              <Zap size={11} className="text-amber-300" />
              <span className="text-[10px] font-semibold tracking-wide text-white/80">
                {speculation}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* -------------------------------------------------- chat (right 65%) */}
      <div className="relative z-10 flex h-full w-full flex-1 flex-row lg:w-[65%]">
        <div
          className={`${
            showPanel ? 'w-[340px]' : 'w-0'
          } z-30 shrink-0 overflow-hidden border-r border-line bg-surface-1 shadow-lg transition-all duration-300`}
        >
          <SidePanel
            filters={filters}
            onFiltersChange={setFilters}
            showSources={showSources}
            onShowSourcesChange={setShowSources}
            detectedDistrict={detectedDistrict}
          />
        </div>

        <div className="relative flex h-full flex-1 flex-col bg-surface-0">
          {/* header */}
          <div className="pointer-events-none absolute top-0 z-20 flex w-full items-center justify-between bg-gradient-to-b from-surface-0 to-transparent px-6 py-4">
            <div className="pointer-events-auto flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-brand-blue to-cyan-400 text-white shadow-lg shadow-brand-blue/20">
                <Megaphone size={19} />
              </div>
              <div className="flex flex-col">
                <span className="text-[15px] font-bold tracking-tight text-ink">
                  Political Campaign Assistant
                </span>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-ink-faint">
                  Grounded in uploaded documents
                </span>
              </div>
            </div>
            <div className="pointer-events-auto flex gap-2">
              <button
                onClick={() => setShowPanel((value) => !value)}
                className="rounded-full border border-line bg-surface-1/80 p-2 text-ink-muted shadow-sm backdrop-blur-sm transition-colors hover:bg-surface-2 hover:text-brand-blue"
                title="Documents & filters"
              >
                <SlidersHorizontal size={17} />
              </button>
              <button
                onClick={clearConversation}
                className="rounded-full border border-line bg-surface-1/80 p-2 text-ink-muted shadow-sm backdrop-blur-sm transition-colors hover:bg-surface-2 hover:text-brand-red"
                title="Clear conversation"
              >
                <Trash2 size={17} />
              </button>
            </div>
          </div>

          {/* messages */}
          <div className="flex flex-1 flex-col overflow-y-auto px-6 pb-44 pt-24">
            {messages.length === 0 ? (
              <div className="mx-auto flex w-full max-w-lg flex-1 flex-col items-center justify-center text-center">
                <div className="mb-6 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-tr from-brand-blue to-cyan-400 text-white shadow-lg shadow-brand-blue/20">
                  <Megaphone size={30} />
                </div>
                <h2 className="mb-3 text-3xl font-bold tracking-tight text-ink">
                  How can I help you today?
                </h2>
                <p className="text-[15px] leading-relaxed text-ink-muted">
                  Ask about the manifesto, welfare schemes, your district, or the
                  candidates. Tell me where you&apos;re from and I&apos;ll keep answers local.
                </p>
              </div>
            ) : (
              <div className="mx-auto w-full max-w-3xl">
                {messages.map((message) => (
                  <div key={message.id}>
                    <MessageBubble
                      message={message.text}
                      isUser={message.isUser}
                      timestamp={message.timestamp}
                      grounded={message.grounded ?? true}
                      streaming={message.streaming}
                      onSpeak={
                        !message.isUser && message.text ? () => void speak(message.text) : undefined
                      }
                    />
                    {showSources &&
                      !message.isUser &&
                      message.citations &&
                      message.citations.length > 0 && (
                        <div className="mb-6 ml-1 mt-1 space-y-2">
                          <div className="text-[10px] font-bold uppercase tracking-wider text-ink-faint">
                            Sources
                          </div>
                          <div className="grid grid-cols-1 gap-2.5 md:grid-cols-2">
                            {message.citations.map((citation) => (
                              <SourceCard key={citation.chunk_id} source={citation} />
                            ))}
                          </div>
                        </div>
                      )}
                  </div>
                ))}

                {liveTranscript && (
                  <div className="mb-6 flex justify-end">
                    <div className="max-w-[85%] rounded-2xl rounded-tr-sm border-2 border-dashed border-brand-blue/30 bg-brand-blue/5 px-5 py-3.5">
                      <div className="mb-1 flex items-center gap-1.5">
                        <Radio size={10} className="animate-pulse text-brand-red" />
                        <span className="text-[9px] font-bold uppercase tracking-wider text-ink-faint">
                          Listening
                        </span>
                      </div>
                      <span className="text-[15px] font-medium text-ink-muted">
                        {liveTranscript}
                      </span>
                    </div>
                  </div>
                )}

                {loading && !messages.some((m) => m.streaming) && (
                  <div className="flex justify-start">
                    <div className="flex gap-2 rounded-2xl border border-line bg-surface-1 px-5 py-4 shadow-sm">
                      <div className="h-2.5 w-2.5 animate-bounce rounded-full bg-brand-blue/60" />
                      <div
                        className="h-2.5 w-2.5 animate-bounce rounded-full bg-brand-blue/80"
                        style={{ animationDelay: '0.15s' }}
                      />
                      <div
                        className="h-2.5 w-2.5 animate-bounce rounded-full bg-brand-blue"
                        style={{ animationDelay: '0.3s' }}
                      />
                    </div>
                  </div>
                )}
                <div ref={scrollAnchor} />
              </div>
            )}
          </div>

          {/* input */}
          <div className="pointer-events-none absolute bottom-0 flex w-full flex-col items-center bg-gradient-to-t from-surface-0 via-surface-0/90 to-transparent px-4 pb-7 pt-10 lg:px-8">
            {statusNote && (
              <div className="pointer-events-auto mb-2.5 flex max-w-3xl items-start gap-2 rounded-xl border border-brand-amber/30 bg-brand-amber/10 px-3.5 py-2">
                <Info size={13} className="mt-0.5 shrink-0 text-brand-amber" />
                <span className="text-[11px] font-medium leading-snug text-brand-amber">
                  {statusNote}
                </span>
              </div>
            )}

            <div className="pointer-events-auto flex w-full max-w-3xl items-center rounded-full border border-line bg-surface-1 px-2 py-2 shadow-[0_8px_30px_rgba(15,35,70,0.10)] backdrop-blur-lg">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    void sendText(input);
                  }
                }}
                placeholder="Ask about schemes, your district, candidates…"
                className="flex-1 bg-transparent py-3 pl-5 text-[15px] text-ink placeholder-ink-faint focus:outline-none"
                disabled={loading}
              />

              <div className="ml-2 flex items-center gap-1.5 pr-1">
                {isSpeaking && (
                  <button
                    onClick={stopSpeaking}
                    className="rounded-full p-2.5 text-ink-faint transition-all hover:bg-surface-2 hover:text-brand-red"
                    title="Stop speaking"
                  >
                    <VolumeX size={19} />
                  </button>
                )}
                <button
                  onClick={() => void sendText(input)}
                  disabled={loading || !input.trim()}
                  className="mx-1 rounded-full bg-brand-blue p-2.5 text-white transition-all duration-300 hover:bg-[#004c8c] hover:shadow-md hover:shadow-brand-blue/30 disabled:bg-surface-2 disabled:text-ink-faint disabled:shadow-none"
                  title="Send"
                >
                  <Send size={17} className="translate-x-[1px]" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
