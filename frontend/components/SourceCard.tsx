'use client';

import React from 'react';
import { BookText, FileText, Hash, MapPin, Sparkles } from 'lucide-react';
import type { Category, Citation } from '@/lib/api';

const CATEGORY_LABEL: Record<Category, string> = {
  manifesto: 'Manifesto',
  district_info: 'District Info',
  candidate_profile: 'Candidate',
  scheme: 'Scheme',
  faq: 'FAQ',
  press_release: 'Press Release',
  speech: 'Speech',
  other: 'Document',
};

// Deep foreground over a pale wash of the same hue — the light-theme pairing.
// (The dark build inverted this: bright 300-weight text over a 10%-alpha wash,
// because on near-black a 700-on-50 pair collapses into one illegible value.)
const CATEGORY_TINT: Record<Category, string> = {
  manifesto: 'text-brand-blue bg-brand-blue/10',
  district_info: 'text-teal-700 bg-teal-50',
  candidate_profile: 'text-purple-700 bg-purple-50',
  scheme: 'text-emerald-700 bg-emerald-50',
  faq: 'text-brand-amber bg-brand-amber/10',
  press_release: 'text-rose-700 bg-rose-50',
  speech: 'text-indigo-700 bg-indigo-50',
  other: 'text-ink-muted bg-surface-2',
};

export default function SourceCard({ source }: { source: Citation }) {
  const tint = CATEGORY_TINT[source.category] ?? CATEGORY_TINT.other;

  return (
    <div className="group relative overflow-hidden rounded-2xl border border-line bg-surface-1 p-4 shadow-sm transition-all hover:border-brand-blue/40 hover:shadow-md">
      <div className="absolute right-0 top-0 rounded-bl-xl bg-surface-2 px-2 py-1 text-[10px] font-bold uppercase tracking-widest text-ink-faint transition-colors group-hover:bg-surface-2 group-hover:text-brand-blue">
        {source.marker}
      </div>

      <div className="mb-3 flex items-start gap-3 pr-10">
        <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${tint}`}>
          <BookText size={17} />
        </div>
        <div className="min-w-0 flex-1">
          <h4 className="truncate text-[13px] font-bold leading-tight text-ink">
            {source.source}
          </h4>
          <div className="mt-1 flex flex-wrap items-center gap-1.5">
            <span className={`rounded px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wide ${tint}`}>
              {CATEGORY_LABEL[source.category] ?? 'Document'}
            </span>
            {/* Relevance is the reranker's calibrated probability, not cosine. */}
            <span className="inline-flex items-center gap-1 text-[9px] font-bold uppercase tracking-wide text-ink-faint">
              <Sparkles size={9} />
              {(source.score * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      <p className="mb-3 line-clamp-3 text-[11.5px] leading-relaxed text-ink-muted">
        {source.snippet}
      </p>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line pt-2.5">
        {source.district && (
          <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-ink-muted">
            <MapPin size={11} className="text-ink-faint" />
            {source.district}
          </span>
        )}
        {source.section && (
          <span className="inline-flex min-w-0 items-center gap-1 text-[10px] font-semibold text-ink-muted">
            <Hash size={11} className="shrink-0 text-ink-faint" />
            <span className="truncate">{source.section}</span>
          </span>
        )}
        {source.page ? (
          <span className="inline-flex items-center gap-1 text-[10px] font-semibold text-ink-muted">
            <FileText size={11} className="text-ink-faint" />
            p.{source.page}
          </span>
        ) : null}
      </div>
    </div>
  );
}
