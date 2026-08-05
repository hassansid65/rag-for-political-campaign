'use client';

import React from 'react';
import { AlertTriangle, Volume2 } from 'lucide-react';

interface MessageBubbleProps {
  message: string;
  isUser: boolean;
  timestamp?: Date;
  grounded?: boolean;
  streaming?: boolean;
  onSpeak?: () => void;
}

/**
 * Renders assistant text with citation markers turned into inline chips.
 *
 * The backend emits `[1]`, `[2]` markers that map to the source cards below the
 * bubble. Rendering them as visible superscript chips is what makes the grounding
 * legible — a wall of prose with hidden provenance looks the same whether it is
 * cited or invented.
 */
function renderContent(text: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  // Split on citation markers, keeping them as capture groups.
  const parts = text.split(/(\[\d{1,2}\])/g);

  parts.forEach((part, index) => {
    const marker = part.match(/^\[(\d{1,2})\]$/);
    if (marker) {
      nodes.push(
        <sup
          key={`c-${index}`}
          className="mx-0.5 inline-flex h-4 min-w-4 items-center justify-center rounded bg-brand-blue/10 px-1 text-[10px] font-bold text-brand-blue align-super"
          title={`Source ${marker[1]}`}
        >
          {marker[1]}
        </sup>
      );
      return;
    }

    // Preserve paragraph breaks; the model is instructed not to emit markdown,
    // so we deliberately do not run a markdown parser here (and never
    // dangerouslySetInnerHTML on model output).
    const lines = part.split('\n');
    lines.forEach((line, lineIndex) => {
      if (line) nodes.push(<span key={`t-${index}-${lineIndex}`}>{line}</span>);
      if (lineIndex < lines.length - 1) nodes.push(<br key={`br-${index}-${lineIndex}`} />);
    });
  });

  return nodes;
}

export default function MessageBubble({
  message,
  isUser,
  timestamp,
  grounded = true,
  streaming = false,
  onSpeak,
}: MessageBubbleProps) {
  return (
    <div className={`group mb-6 flex ${isUser ? 'justify-end' : 'justify-start'} animate-fade-up`}>
      <div className="relative max-w-[85%] lg:max-w-[78%]">
        {isUser ? (
          <div className="rounded-2xl rounded-tr-sm bg-brand-blue-dim px-5 py-4 font-medium leading-relaxed text-white shadow-lg shadow-black/5">
            {message}
          </div>
        ) : (
          <div className="relative rounded-2xl rounded-tl-sm border border-line bg-surface-1 px-6 py-5 leading-relaxed text-ink shadow-sm">
            <div className="break-words text-[15px]">
              {renderContent(message)}
              {streaming && (
                <span className="ml-0.5 inline-block h-4 w-[2px] animate-pulse bg-brand-blue align-middle" />
              )}
            </div>

            {!grounded && !streaming && (
              <div className="mt-3 flex items-start gap-2 rounded-lg border border-brand-amber/30 bg-brand-amber/10 px-3 py-2">
                <AlertTriangle size={14} className="mt-0.5 shrink-0 text-brand-amber" />
                <span className="text-[11px] font-medium leading-snug text-brand-amber">
                  No source was cited for this answer — it may fall outside the uploaded
                  campaign documents.
                </span>
              </div>
            )}

            {onSpeak && !streaming && (
              <div className="absolute -right-11 top-1 flex flex-col gap-1.5 opacity-0 transition-opacity group-hover:opacity-100">
                <button
                  onClick={onSpeak}
                  className="rounded-xl border border-line bg-surface-1 p-2 text-ink-faint shadow-sm transition-all hover:scale-105 hover:text-ink"
                  title="Read this answer aloud"
                >
                  <Volume2 size={15} />
                </button>
              </div>
            )}
          </div>
        )}

        {timestamp && (
          <div
            className={`mt-2 text-[10px] font-bold uppercase tracking-widest opacity-40 ${
              isUser ? 'pr-1 text-right' : 'pl-1'
            }`}
          >
            {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        )}
      </div>
    </div>
  );
}
