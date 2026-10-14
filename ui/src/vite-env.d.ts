/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Sentry DSN for the browser project. Unset (the default) means Sentry is never loaded. */
  readonly VITE_SENTRY_DSN?: string;
  /** "1" builds the static demo: no backend calls, demo results load on start. */
  readonly VITE_DEMO_ONLY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
