# SecondRead UI

Vite + vanilla TypeScript, no framework. The backend contract is in [API.md](API.md).

```sh
npm install
npm run dev          # http://127.0.0.1:5173, /api proxied to http://127.0.0.1:8787
npm run build        # type-check + production build into dist/
npm run build:demo   # static demo build: no /api calls, loads public/demo-results.json on start
npm run preview      # serve dist/ (same /api proxy)
```

| Variable | Where | Effect |
|---|---|---|
| `SECONDREAD_API` | dev/preview server | proxy target instead of `http://127.0.0.1:8787` |
| `VITE_SENTRY_DSN` | build time | turns on `@sentry/browser` (lazy chunk); unset = never loaded |
| `VITE_DEMO_ONLY=1` | build time | what `build:demo` sets through `.env.demo` |

`/?demo` opens the demo in a normal build too.

The demo data is a 10-segment slice of the public WMT22 en-ru test set, scored by the real
backend with the base model (no LoRA yet), so the page labels it as placeholder scores.
Regenerate it with `scripts/make_demo_results.py --api http://127.0.0.1:8787` once the adapter
is loaded.
