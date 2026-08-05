/**
 * Backend client.
 *
 * `queryStream` is the important one: it consumes the SSE stream from
 * POST /query so the UI can render source cards from the `retrieval` event while
 * the model is still producing `delta` events, instead of blocking on the whole
 * answer. Everything else is a thin typed wrapper.
 */

export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

// ----------------------------------------------------------------------- types
export type Category =
  | 'manifesto'
  | 'district_info'
  | 'candidate_profile'
  | 'scheme'
  | 'faq'
  | 'press_release'
  | 'speech'
  | 'other';

export interface ChunkMetadata {
  doc_id: string;
  source: string;
  category: Category;
  district?: string | null;
  districts: string[];
  state?: string | null;
  topic?: string | null;
  topics: string[];
  section?: string | null;
  section_path: string[];
  page?: number | null;
  language: string;
  candidate?: string | null;
  party?: string | null;
  scheme_names: string[];
  chunk_index: number;
}

export interface RetrievedChunk {
  id: string;
  text: string;
  chunk_text?: string | null;
  metadata: ChunkMetadata;
  score: number;
  dense_score?: number | null;
  sparse_score?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  retriever: string;
}

export interface Citation {
  marker: string;
  source: string;
  category: Category;
  district?: string | null;
  section?: string | null;
  page?: number | null;
  chunk_id: string;
  score: number;
  snippet: string;
}

export interface RetrieveFilters {
  district?: string;
  category?: Category;
  topic?: string;
  source?: string;
  language?: string;
}

export interface QueryResponse {
  answer: string;
  session_id: string;
  query: string;
  effective_query: string;
  grounded: boolean;
  citations: Citation[];
  sources_used: number;
  retrieved: RetrievedChunk[];
  inferred_filters: Record<string, unknown>;
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens: number;
    cache_read_input_tokens: number;
  };
  model: string;
  cache_hit: boolean;
  timings_ms: Record<string, unknown>;
}

export interface RetrieveResponse {
  query: string;
  effective_query: string;
  rewrites: string[];
  inferred_filters: Record<string, unknown>;
  results: RetrievedChunk[];
  total_candidates: number;
  reranked: boolean;
  cache_hit: boolean;
  timings_ms: Record<string, unknown>;
}

export interface UploadedDocument {
  doc_id: string;
  source: string;
  category: Category;
  districts: string[];
  topics: string[];
  pages?: number | null;
  chars: number;
  chunks_indexed: number;
  detected_language: string;
  warnings: string[];
}

export interface DocumentSummary {
  doc_id: string;
  source: string;
  category: Category;
  districts: string[];
  topics: string[];
  chunks: number;
  ingested_at?: string | null;
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'down';
  version: string;
  environment: string;
  uptime_s: number;
  components: Array<{
    name: string;
    status: 'ok' | 'degraded' | 'down' | 'disabled';
    detail: string;
    latency_ms?: number | null;
  }>;
  collection: Record<string, unknown>;
  config: Record<string, unknown>;
  latency_ms: Record<string, { p50: number; p95: number; count: number }>;
}

export interface VoiceTurnResponse {
  text: string;
  spoken_text: string;
  session_id: string;
  audio?: string | null;
  audio_format: string;
  lipsync?: { mouthCues: MouthCue[]; metadata: Record<string, unknown> } | null;
  citations: Citation[];
  grounded: boolean;
  facialExpression: string;
  animation: string;
  timings_ms: Record<string, unknown>;
}

export interface MouthCue {
  start: number;
  end: number;
  value: string;
}

// --------------------------------------------------------------------- helpers
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BACKEND_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${response.status} — ${detail}`);
  }
  return response.json() as Promise<T>;
}

// ------------------------------------------------------------------------- api
export const api = {
  health: () => request<HealthResponse>('/health'),

  metrics: () => request<Record<string, unknown>>('/metrics'),

  districts: () => request<{ districts: string[]; total: number }>('/districts'),

  documents: () =>
    request<{ documents: DocumentSummary[]; total_documents: number; total_chunks: number }>(
      '/documents'
    ),

  deleteDocument: (docId: string) =>
    request<{ status: string; chunks_removed: number }>(`/documents/${docId}`, {
      method: 'DELETE',
    }),

  retrieve: (body: {
    query: string;
    top_k?: number;
    filters?: RetrieveFilters;
    session_id?: string;
    similarity_threshold?: number;
  }) => request<RetrieveResponse>('/retrieve', { method: 'POST', body: JSON.stringify(body) }),

  query: (body: {
    query: string;
    session_id?: string;
    top_k?: number;
    filters?: RetrieveFilters;
    include_citations?: boolean;
    voice_mode?: boolean;
  }) => request<QueryResponse>('/query', { method: 'POST', body: JSON.stringify(body) }),

  voiceTurn: (body: {
    message: string;
    session_id: string;
    voice?: string;
    filters?: RetrieveFilters;
  }) => request<VoiceTurnResponse>('/voice/turn', { method: 'POST', body: JSON.stringify(body) }),

  tts: (body: { text: string; voice?: string }) =>
    request<{
      audio: string;
      spoken_text: string;
      duration_s: number;
      lipsync: { mouthCues: MouthCue[]; metadata: Record<string, unknown> };
    }>('/voice/tts', { method: 'POST', body: JSON.stringify(body) }),

  stt: (audioBase64: string, format = 'webm') =>
    request<{ text: string; success: boolean; language?: string; error?: string }>('/voice/stt', {
      method: 'POST',
      body: JSON.stringify({ audio: audioBase64, format }),
    }),

  voices: () =>
    request<{ presets: Record<string, string>; default: string; stt_languages: string[] }>(
      '/voice/voices'
    ),

  resetSession: (sessionId: string) =>
    request<{ status: string }>(`/sessions/${sessionId}`, { method: 'DELETE' }),

  async upload(files: File[], meta?: Record<string, string>) {
    const form = new FormData();
    files.forEach((file) => form.append('files', file));
    Object.entries(meta || {}).forEach(([key, value]) => {
      if (value) form.append(key, value);
    });
    const response = await fetch(`${BACKEND_URL}/upload`, { method: 'POST', body: form });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail =
          typeof body.detail === 'string'
            ? body.detail
            : body.detail?.message || JSON.stringify(body.detail);
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    return response.json() as Promise<{
      status: string;
      documents: UploadedDocument[];
      total_chunks_indexed: number;
      message: string;
    }>;
  },
};

// -------------------------------------------------------------------- SSE query
export type StreamEvent =
  | { type: 'retrieval'; sources: Citation[]; effective_query: string; inferred_filters: Record<string, unknown>; chunks: RetrievedChunk[]; cache_hit: boolean; timings_ms: Record<string, unknown> }
  | { type: 'delta'; text: string }
  | { type: 'final'; answer: string; grounded: boolean; citations: Citation[]; session_id: string; usage: QueryResponse['usage']; model: string; timings_ms: Record<string, unknown>; notes: string[] }
  | { type: 'error'; error: string };

/**
 * Consume POST /query as Server-Sent Events.
 *
 * fetch + ReadableStream rather than EventSource because EventSource cannot issue
 * a POST, and the request body carries the query, filters, and session id.
 */
export async function queryStream(
  body: {
    query: string;
    session_id?: string;
    top_k?: number;
    filters?: RetrieveFilters;
    voice_mode?: boolean;
  },
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const response = await fetch(`${BACKEND_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Stream failed: ${response.status} ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; a frame may split across reads.
    const frames = buffer.split('\n\n');
    buffer = frames.pop() ?? '';

    for (const frame of frames) {
      let eventName = 'message';
      const dataLines: string[] = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) eventName = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
      }
      if (eventName === 'end' || dataLines.length === 0) continue;
      try {
        onEvent(JSON.parse(dataLines.join('\n')) as StreamEvent);
      } catch {
        /* skip malformed frame rather than killing the stream */
      }
    }
  }
}

// ------------------------------------------------------------------------ audio
export function base64ToAudioUrl(base64: string, mime = 'audio/wav'): string {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}
