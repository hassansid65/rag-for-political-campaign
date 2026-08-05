'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  Check,
  CloudUpload,
  Filter,
  Gauge,
  Loader2,
  MapPin,
  Trash2,
  XCircle,
} from 'lucide-react';
import {
  api,
  type Category,
  type DocumentSummary,
  type HealthResponse,
  type RetrieveFilters,
} from '@/lib/api';

const CATEGORIES: Array<{ value: Category | 'all'; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'manifesto', label: 'Manifesto' },
  { value: 'district_info', label: 'District' },
  { value: 'candidate_profile', label: 'Candidate' },
  { value: 'scheme', label: 'Schemes' },
  { value: 'faq', label: 'FAQ' },
];

interface SidePanelProps {
  filters: RetrieveFilters;
  onFiltersChange: (filters: RetrieveFilters) => void;
  showSources: boolean;
  onShowSourcesChange: (value: boolean) => void;
  detectedDistrict: string | null;
}

export default function SidePanel({
  filters,
  onFiltersChange,
  showSources,
  onShowSourcesChange,
  detectedDistrict,
}: SidePanelProps) {
  const [districts, setDistricts] = useState<string[]>([]);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    const [docs, hp] = await Promise.allSettled([api.documents(), api.health()]);
    if (docs.status === 'fulfilled') setDocuments(docs.value.documents);
    if (hp.status === 'fulfilled') setHealth(hp.value);
  }, []);

  useEffect(() => {
    api
      .districts()
      .then((d) => setDistricts(d.districts))
      .catch(() => setDistricts([]));
    void refresh();
    // Poll health so the latency panel reflects the queries being run.
    const timer = setInterval(refresh, 15_000);
    return () => clearInterval(timer);
  }, [refresh]);

  const handleFiles = useCallback(
    async (files: FileList | File[] | null) => {
      const list = Array.from(files ?? []);
      if (list.length === 0) return;

      setUploading(true);
      setUploadMessage(null);
      try {
        const result = await api.upload(list);
        setUploadMessage({ ok: true, text: result.message });
        await refresh();
      } catch (error) {
        setUploadMessage({
          ok: false,
          text: error instanceof Error ? error.message : 'Upload failed',
        });
      } finally {
        setUploading(false);
        if (fileInput.current) fileInput.current.value = '';
      }
    },
    [refresh]
  );

  const patch = (next: Partial<RetrieveFilters>) => {
    const merged: RetrieveFilters = { ...filters, ...next };
    (Object.keys(merged) as Array<keyof RetrieveFilters>).forEach((key) => {
      if (!merged[key]) delete merged[key];
    });
    onFiltersChange(merged);
  };

  const hasFilters = Object.keys(filters).length > 0;
  const totalChunks = documents.reduce((sum, doc) => sum + doc.chunks, 0);

  return (
    <div className="flex h-full flex-col gap-7 overflow-y-auto p-6">
      {/* ------------------------------------------------------------ upload */}
      <section>
        <SectionHeading icon={<CloudUpload size={12} />} label="Campaign Documents" />
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            void handleFiles(e.dataTransfer.files);
          }}
          onClick={() => fileInput.current?.click()}
          className={`cursor-pointer rounded-2xl border-2 border-dashed p-5 text-center transition-all ${
            dragging
              ? 'border-brand-blue bg-brand-blue/10'
              : 'border-line bg-surface-1 hover:border-brand-blue/50 hover:bg-surface-2'
          }`}
        >
          <input
            ref={fileInput}
            type="file"
            multiple
            accept=".pdf,.docx,.txt,.md,.markdown,.html,.csv"
            className="hidden"
            onChange={(e) => void handleFiles(e.target.files)}
          />
          {uploading ? (
            <div className="flex flex-col items-center gap-2 text-brand-blue">
              <Loader2 size={20} className="animate-spin" />
              <span className="text-[11px] font-bold uppercase tracking-wider">
                Parsing &amp; indexing…
              </span>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-1.5">
              <CloudUpload size={20} className="text-ink-faint" />
              <span className="text-[12px] font-bold text-ink">
                Drop files or click to upload
              </span>
              <span className="text-[10px] font-medium text-ink-faint">
                PDF · DOCX · TXT · MD · HTML · CSV
              </span>
            </div>
          )}
        </div>

        {uploadMessage && (
          <div
            className={`mt-2 flex items-start gap-2 rounded-xl px-3 py-2 text-[11px] font-medium ${
              uploadMessage.ok
                ? 'bg-brand-green/10 text-brand-green'
                : 'bg-brand-red/10 text-brand-red'
            }`}
          >
            {uploadMessage.ok ? (
              <Check size={13} className="mt-0.5 shrink-0" />
            ) : (
              <XCircle size={13} className="mt-0.5 shrink-0" />
            )}
            <span className="leading-snug">{uploadMessage.text}</span>
          </div>
        )}

        {documents.length > 0 && (
          <div className="mt-3 space-y-1.5">
            <div className="flex items-center justify-between px-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-ink-faint">
                {documents.length} indexed
              </span>
              <span className="text-[10px] font-bold uppercase tracking-wider text-ink-faint">
                {totalChunks} chunks
              </span>
            </div>
            {documents.map((doc) => (
              <div
                key={doc.doc_id}
                className="group flex items-center gap-2 rounded-xl border border-line bg-surface-1 px-3 py-2"
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[11.5px] font-semibold text-ink">
                    {doc.source}
                  </div>
                  <div className="truncate text-[9.5px] font-medium uppercase tracking-wide text-ink-faint">
                    {doc.category.replace('_', ' ')} · {doc.chunks} chunks
                    {doc.districts.length > 0 && ` · ${doc.districts.slice(0, 2).join(', ')}`}
                  </div>
                </div>
                <button
                  onClick={async () => {
                    await api.deleteDocument(doc.doc_id).catch(() => undefined);
                    await refresh();
                  }}
                  className="shrink-0 rounded-lg p-1.5 text-ink-faint opacity-0 transition-all hover:bg-brand-red/10 hover:text-brand-red group-hover:opacity-100"
                  title="Remove from index"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* ----------------------------------------------------------- filters */}
      <section>
        <SectionHeading icon={<Filter size={12} />} label="Retrieval Filters" />

        {detectedDistrict && !filters.district && (
          <div className="mb-3 flex items-center gap-2 rounded-xl border border-brand-green/30 bg-brand-green/10 px-3 py-2">
            <MapPin size={13} className="shrink-0 text-brand-green" />
            <span className="text-[11px] font-semibold leading-snug text-brand-green">
              Detected from conversation:{' '}
              <span className="font-bold">{detectedDistrict}</span>
            </span>
          </div>
        )}

        <label className="mb-1.5 block text-[10px] font-bold uppercase tracking-wider text-ink-faint">
          District
        </label>
        <select
          value={filters.district || 'all'}
          onChange={(e) => patch({ district: e.target.value === 'all' ? undefined : e.target.value })}
          className="mb-4 w-full cursor-pointer appearance-none rounded-xl border border-line bg-surface-1 px-4 py-2.5 text-[12.5px] font-semibold text-ink shadow-sm transition-all focus:outline-none focus:ring-2 focus:ring-brand-blue/20"
        >
          <option value="all">All districts</option>
          {districts.map((district) => (
            <option key={district} value={district}>
              {district}
            </option>
          ))}
        </select>

        <label className="mb-1.5 block text-[10px] font-bold uppercase tracking-wider text-ink-faint">
          Document type
        </label>
        <div className="grid grid-cols-2 gap-2">
          {CATEGORIES.map((category) => {
            const active =
              filters.category === category.value ||
              (!filters.category && category.value === 'all');
            return (
              <button
                key={category.value}
                onClick={() =>
                  patch({
                    category:
                      category.value === 'all' ? undefined : (category.value as Category),
                  })
                }
                className={`rounded-xl border px-3 py-2 text-[11.5px] font-semibold transition-all ${
                  active
                    ? 'border-brand-blue bg-brand-blue text-white shadow-md shadow-brand-blue/20'
                    : 'border-line bg-surface-1 text-ink-muted hover:border-line-strong'
                }`}
              >
                {category.label}
              </button>
            );
          })}
        </div>

        <label className="mt-4 flex cursor-pointer items-center justify-between rounded-2xl border border-line bg-surface-1 p-3.5 shadow-sm transition-colors hover:bg-surface-2">
          <span className="text-[11.5px] font-bold text-ink">Show citations</span>
          <input
            type="checkbox"
            checked={showSources}
            onChange={(e) => onShowSourcesChange(e.target.checked)}
            className="h-4 w-4 rounded accent-brand-blue"
          />
        </label>

        {hasFilters && (
          <button
            onClick={() => onFiltersChange({})}
            className="mt-3 flex w-full items-center justify-center gap-2 rounded-2xl border border-brand-red/10 bg-brand-red/5 py-3 text-[11px] font-bold uppercase tracking-widest text-brand-red transition-all hover:bg-brand-red hover:text-white active:scale-95"
          >
            <XCircle size={14} />
            Reset filters
          </button>
        )}
      </section>

      {/* ------------------------------------------------------------ health */}
      {health && (
        <section>
          <SectionHeading icon={<Activity size={12} />} label="System" />
          <div className="space-y-1.5">
            {health.components.map((component) => (
              <div
                key={component.name}
                className="flex items-center gap-2.5 rounded-xl border border-line bg-surface-1 px-3 py-2"
              >
                <span
                  className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                    component.status === 'ok'
                      ? 'bg-emerald-500'
                      : component.status === 'degraded'
                        ? 'bg-brand-amber/100'
                        : component.status === 'disabled'
                          ? 'bg-ink-faint'
                          : 'bg-rose-500'
                  }`}
                />
                <div className="min-w-0 flex-1">
                  <div className="text-[11px] font-bold text-ink">
                    {component.name.replace(/_/g, ' ')}
                  </div>
                  <div className="truncate text-[9.5px] font-medium text-ink-faint">
                    {component.detail || component.status}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <LatencyPanel latency={health.latency_ms} />
        </section>
      )}
    </div>
  );
}

function SectionHeading({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="mb-3 flex items-center gap-2 text-[10px] font-bold uppercase tracking-wider text-ink-faint">
      {icon}
      <span>{label}</span>
    </div>
  );
}

/** Surfaces the p50/p95 the backend already tracks per pipeline stage. */
function LatencyPanel({ latency }: { latency: HealthResponse['latency_ms'] }) {
  const rows = [
    ['embed', 'embed.query'],
    ['vector search', 'store.search_dense'],
    ['bm25', 'store.search_sparse'],
    ['rerank', 'rerank.cascade'],
    ['llm 1st token', 'llm.first_token'],
    ['tts', 'tts.synthesize'],
  ] as const;

  const present = rows.filter(([, key]) => latency[key]?.count);
  if (present.length === 0) return null;

  return (
    <div className="mt-3 rounded-2xl border border-line bg-surface-1 p-3.5">
      <div className="mb-2.5 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-ink-faint">
        <Gauge size={11} />
        <span>Latency (p50 / p95)</span>
      </div>
      <div className="space-y-1">
        {present.map(([label, key]) => {
          const stat = latency[key];
          return (
            <div key={key} className="flex items-center justify-between gap-2">
              <span className="text-[10.5px] font-semibold text-ink-muted">{label}</span>
              <span className="font-mono text-[10px] font-bold text-ink">
                {stat.p50.toFixed(0)} / {stat.p95.toFixed(0)}
                <span className="ml-0.5 font-sans font-medium text-ink-faint">ms</span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
