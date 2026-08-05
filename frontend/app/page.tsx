'use client';

import dynamic from 'next/dynamic';

// The chat shell pulls in three.js and a WebGL canvas, neither of which can be
// server-rendered.
//
// `ssr: false` is only permitted inside a Client Component from Next 15 onward —
// in a Server Component it is a build error, since the server cannot skip
// rendering a child it is responsible for. Hence the 'use client' above: this
// file is a thin client boundary whose only job is to defer the heavy import.
const ChatWindow = dynamic(() => import('@/components/ChatWindow'), {
  ssr: false,
  loading: () => (
    <div className="flex h-screen items-center justify-center bg-[#0a0c10]">
      <div className="flex flex-col items-center gap-4">
        <div className="h-9 w-9 animate-spin rounded-full border-2 border-white/20 border-t-white/80" />
        <span className="text-xs font-semibold uppercase tracking-widest text-white/50">
          Starting assistant
        </span>
      </div>
    </div>
  ),
});

export default function Home() {
  return (
    <main className="h-screen">
      <ChatWindow />
    </main>
  );
}
