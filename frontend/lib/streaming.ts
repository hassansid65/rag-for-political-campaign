/**
 * Two streaming helpers for the chat UI.
 *
 * `Typewriter` — smooths the *visual* stream.
 * `SentenceSplitter` — starts audio before the answer is finished.
 */

/**
 * Drip-feed accumulated text to the UI at a steady character rate.
 *
 * Token streams are lumpy: an LLM emits 2 characters, then 30, then 4, then
 * pauses 400 ms. Rendering each delta as it lands produces visible stutter — text
 * jumps in blocks, which reads as "loading" rather than "writing". ChatGPT's feel
 * comes from decoupling arrival from display: deltas go into a buffer, and a
 * rAF loop reveals characters at a near-constant rate.
 *
 * The rate adapts to the backlog so the display never falls far behind the model
 * — a long answer that arrives fast still finishes promptly instead of typing
 * out for twenty seconds after generation ended.
 */
export class Typewriter {
  private target = '';
  private shown = 0;
  private frame: number | null = null;
  private lastTick = 0;
  private finished = false;
  /**
   * Character index up to which a fixed reveal rate applies, and that rate.
   *
   * Used to pin text to speech: when a sentence's audio starts and we know it
   * lasts 3.2 s, that sentence's characters are revealed over exactly 3.2 s so the
   * caption tracks the voice instead of racing ahead of it.
   */
  private pacedUntil = 0;
  private pacedRate = 0;

  constructor(
    private readonly onRender: (text: string) => void,
    private readonly baseCharsPerSecond = 220
  ) {}

  /** Append newly arrived text. Safe to call at any rate. */
  push(chunk: string): void {
    this.target += chunk;
    this.start();
  }

  /**
   * Append text that should be revealed over exactly `durationMs`.
   *
   * This is what keeps text and voice together. Revealing at the model's token
   * rate finishes the paragraph seconds before the first audio clip returns from
   * Azure; revealing at the *audio's* rate means a word appears as it is spoken.
   */
  pushOver(chunk: string, durationMs: number): void {
    if (!chunk) return;
    this.target += chunk;
    this.pacedUntil = this.target.length;
    // Slight lead (1.06x) so the caption is never a frame behind the audio —
    // text trailing speech reads as lag, text a hair ahead reads as natural.
    this.pacedRate = Math.max(
      12,
      (chunk.length / Math.max(250, durationMs)) * 1000 * 1.06
    );
    this.start();
  }

  /** Replace the whole target (used when `final` arrives with canonical text). */
  set(text: string): void {
    this.target = text;
    this.start();
  }

  /** How much text is still unrevealed. */
  get backlog(): number {
    return this.target.length - this.shown;
  }

  /** Stop dripping and reveal everything immediately. */
  flush(): void {
    this.finished = true;
    this.stop();
    this.shown = this.target.length;
    this.onRender(this.target);
  }

  stop(): void {
    if (this.frame !== null) {
      cancelAnimationFrame(this.frame);
      this.frame = null;
    }
  }

  private start(): void {
    if (this.frame !== null || this.finished) return;
    this.lastTick = performance.now();
    this.frame = requestAnimationFrame(this.tick);
  }

  private readonly tick = (now: number): void => {
    const elapsed = Math.max(0, now - this.lastTick);
    this.lastTick = now;

    const backlog = this.target.length - this.shown;
    if (backlog <= 0) {
      this.frame = null;
      return;
    }

    let speed: number;
    if (this.shown < this.pacedUntil && this.pacedRate > 0) {
      // Inside an audio-paced segment: follow the voice, ignore the backlog.
      speed = this.pacedRate;
    } else {
      // Free-running (voice off, or text ahead of any queued audio). Speed up when
      // the buffer is deep, or a fast 800-character answer would still be typing
      // four seconds after the model finished.
      speed =
        backlog > 400 ? this.baseCharsPerSecond * 4
        : backlog > 150 ? this.baseCharsPerSecond * 2
        : this.baseCharsPerSecond;
    }

    const advance = Math.max(1, Math.round((speed * elapsed) / 1000));
    this.shown = Math.min(this.target.length, this.shown + advance);
    this.onRender(this.target.slice(0, this.shown));

    this.frame = requestAnimationFrame(this.tick);
  };
}

/**
 * Emit complete sentences from a growing text stream, exactly once each.
 *
 * This is what removes the dominant chunk of perceived voice latency. Synthesizing
 * the finished answer means first audio waits for the last token; synthesizing
 * sentence-by-sentence means it waits only for the first sentence — typically a
 * fifth of the text.
 *
 * A sentence is only released when punctuation is followed by whitespace *and* a
 * capital or digit, so "Rs. 44.0 lakh" and "Dr. Kesineni" are not split into
 * fragments. TTS on "Rs." alone is both wrong and audible.
 */
export class SentenceSplitter {
  private buffer = '';
  private emitted = 0;

  private static readonly BOUNDARY = /([.!?।])\s+(?=[A-Z0-9"'(])/g;
  // Abbreviations whose trailing period is never a sentence end here.
  private static readonly ABBREV = /\b(?:Rs|Dr|Sri|Smt|Shri|Mr|Mrs|Ms|Prof|No|vs|etc|approx)\.$/i;

  /** Feed the full accumulated text; returns any newly completed sentences. */
  push(fullText: string): string[] {
    this.buffer = fullText;
    const out: string[] = [];

    SentenceSplitter.BOUNDARY.lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = SentenceSplitter.BOUNDARY.exec(this.buffer)) !== null) {
      const end = match.index + 1; // include the punctuation
      if (end <= this.emitted) continue;

      const candidate = this.buffer.slice(this.emitted, end).trim();
      if (!candidate) continue;
      // Don't cut after an abbreviation's period.
      if (SentenceSplitter.ABBREV.test(candidate)) continue;
      // Too short to be worth a TTS round-trip on its own.
      if (candidate.replace(/[^a-z0-9]/gi, '').length < 12) continue;

      out.push(candidate);
      this.emitted = end;
    }
    return out;
  }

  /** Whatever is left after the stream ends. */
  remainder(): string | null {
    const tail = this.buffer.slice(this.emitted).trim();
    this.emitted = this.buffer.length;
    return tail.replace(/[^a-z0-9]/gi, '').length >= 2 ? tail : null;
  }

  reset(): void {
    this.buffer = '';
    this.emitted = 0;
  }
}

/**
 * Split a finished answer into TTS-sized sentences.
 *
 * Used for the replay button, where the whole text is already available — the
 * streaming path uses `SentenceSplitter` instead so it can emit incrementally.
 */
export function splitIntoSentences(text: string, minChars = 12): string[] {
  const flat = text.replace(/\s+/g, ' ').trim();
  if (!flat) return [];

  const parts = flat.split(/(?<=[.!?।])\s+(?=[A-Z0-9"'(])/);
  const out: string[] = [];
  for (const part of parts) {
    const piece = part.trim();
    if (!piece) continue;
    const previous = out[out.length - 1];
    // Merge a fragment (or an abbreviation that got split) into the previous one.
    if (
      previous &&
      (piece.replace(/[^a-z0-9]/gi, '').length < minChars ||
        /\b(?:Rs|Dr|Sri|Smt|Shri|Mr|Mrs|Ms|Prof|No)\.$/i.test(previous))
    ) {
      out[out.length - 1] = `${previous} ${piece}`;
    } else {
      out.push(piece);
    }
  }
  return out;
}
