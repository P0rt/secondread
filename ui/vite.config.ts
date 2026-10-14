import { defineConfig } from 'vite';

// The backend (FastAPI on 127.0.0.1:8787) is reached through this proxy, so the browser only
// ever talks to its own origin: no CORS, and Sentry's sentry-trace/baggage headers need no
// preflight. `vite preview` gets the same proxy so a production build can be tried locally.
// SECONDREAD_API overrides the target (e.g. a mock backend on another port).
const target = process.env.SECONDREAD_API ?? 'http://127.0.0.1:8787';
const proxy = {
  '/api': { target, changeOrigin: false },
};

export default defineConfig({
  // Relative asset URLs: the static demo build works from any sub-path (e.g. GitHub Pages).
  base: './',
  server: { host: '127.0.0.1', port: 5173, strictPort: false, proxy },
  preview: { host: '127.0.0.1', port: 4173, proxy },
  build: { target: 'es2022' },
});
