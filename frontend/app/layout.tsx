import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'], display: 'swap' });

export const metadata: Metadata = {
  title: 'Political Campaign Assistant',
  description:
    'Voice AI political campaign assistant with real-time retrieval over uploaded campaign documents.',
  icons: { icon: '/icon.svg' },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  // The avatar canvas fills the viewport; pinch-zoom on it is never useful.
  maximumScale: 1,
  themeColor: '#0a0c10',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
