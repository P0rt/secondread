// The only module that imports @sentry/browser. It is loaded with a dynamic import from
// sentry.ts, so it becomes its own chunk, and its static named imports let the bundler drop
// the parts of the SDK we do not use (a dynamic `import('@sentry/browser')` would keep it all).

import {
  breadcrumbsIntegration,
  browserApiErrorsIntegration,
  browserTracingIntegration,
  captureMessage,
  dedupeIntegration,
  eventFiltersIntegration,
  globalHandlersIntegration,
  httpContextIntegration,
  init,
  linkedErrorsIntegration,
  type Breadcrumb,
  type ErrorEvent,
} from '@sentry/browser';

export interface Hooks {
  beforeBreadcrumb: (crumb: Breadcrumb) => Breadcrumb | null;
  beforeSend: (event: ErrorEvent) => ErrorEvent;
  scrub: <T>(value: T) => T;
}

export function start(dsn: string, environment: string, hooks: Hooks): { captureMessage: typeof captureMessage } {
  init({
    dsn,
    environment,
    tracesSampleRate: 1.0,
    // An explicit list instead of the defaults: notably no Console integration, so console
    // output never becomes a breadcrumb in the first place.
    defaultIntegrations: false,
    integrations: [
      browserApiErrorsIntegration(),
      breadcrumbsIntegration({ dom: true, fetch: true, xhr: true, history: true, sentry: false }),
      globalHandlersIntegration(),
      linkedErrorsIntegration(),
      dedupeIntegration(),
      eventFiltersIntegration(),
      httpContextIntegration(),
      browserTracingIntegration(),
    ],
    // Same-origin /api calls only (Vite proxies them), so the backend trace joins the browser one.
    tracePropagationTargets: [/^\/api\//],
    // v11 replaced sendDefaultPii with dataCollection; everything that could carry text is off.
    dataCollection: {
      userInfo: false,
      genAI: { inputs: false, outputs: false },
      httpBodies: [],
      urlQueryParams: false,
      cookies: false,
      stackFrameVariables: false,
    },
    beforeBreadcrumb: hooks.beforeBreadcrumb,
    beforeSend: hooks.beforeSend,
    beforeSendSpan: (span) => hooks.scrub(span),
  });
  return { captureMessage };
}
