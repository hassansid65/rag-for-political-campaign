import type { Config } from 'tailwindcss'

// Light theme on a very faint blue base.
//
// The component classes are unchanged from the dark build — everything is written
// against semantic tokens (surface-*, line, ink-*, brand-*), so switching themes
// is a palette swap here rather than a sweep through four components. That is the
// whole reason the tokens exist.
//
// The avatar panel stays dark on purpose: the GLB's spotlight rig and the
// office-window backdrop are lit for a dark surround, and a bright panel behind a
// rim-lit head reads as a cut-out.
const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // elevation — a faint blue page with white cards floating on it
        'surface-0': '#f4f8fd', // page background
        'surface-1': '#ffffff', // cards, bubbles, sidebar
        'surface-2': '#eaf1fa', // inputs, hover states
        'surface-3': '#dde8f5', // pressed / active
        line: '#dbe5f1', // hairline borders
        'line-strong': '#bfcfe3',

        // text
        ink: '#111a26',
        'ink-muted': '#55657a',
        'ink-faint': '#8593a6',

        // accent — darkened from the dark theme's #58a6ff, which is far too pale
        // on white. This clears WCAG AA against surface-0 and surface-1.
        'brand-blue': '#0b62c4',
        'brand-blue-dim': '#0a539f',
        'brand-red': '#c62828',
        'brand-green': '#177245',
        'brand-amber': '#8a5a00',
      },
      keyframes: {
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(6px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-ring': {
          '0%': { transform: 'scale(0.9)', opacity: '0.7' },
          '100%': { transform: 'scale(1.6)', opacity: '0' },
        },
      },
      animation: {
        'fade-up': 'fade-up 0.25s ease-out',
        'pulse-ring': 'pulse-ring 1.4s cubic-bezier(0.24, 0, 0.38, 1) infinite',
      },
    },
  },
  plugins: [],
}
export default config
