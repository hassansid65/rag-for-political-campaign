/** @type {import('next').NextConfig} */
const nextConfig = {
  // StrictMode double-mounts effects, which restarts the GLB load and can leave
  // two WebGL contexts fighting over the canvas.
  reactStrictMode: false,

  // Next 16 runs Turbopack by default. An empty object is a deliberate opt-in:
  // it silences the "webpack config with no turbopack config" error and states
  // that no custom bundler rules are needed.
  //
  // The previous webpack rule for .glb/.gltf was unnecessary. The avatar models
  // live in `public/` and are fetched by URL at runtime (`useGLTF('/…glb')`),
  // never imported as modules, so no loader is involved.
  turbopack: {},

  env: {
    NEXT_PUBLIC_BACKEND_URL: process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000',
  },
}

module.exports = nextConfig
