import { ErrorBoundary as SolidErrorBoundary, type ParentProps } from 'solid-js';

interface ErrorFallbackProps {
  error: Error;
  reset: () => void;
}

function ErrorFallback(props: ErrorFallbackProps) {
  return (
    <div class="error-boundary">
      <div class="error-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
      </div>
      <h3 class="error-title">Something went wrong</h3>
      <p class="error-message">{props.error.message || 'An unexpected error occurred'}</p>
      <button class="error-retry" onClick={props.reset}>
        Try again
      </button>
    </div>
  );
}

export default function ErrorBoundary(props: ParentProps) {
  return (
    <SolidErrorBoundary
      fallback={(err, reset) => <ErrorFallback error={err} reset={reset} />}
    >
      {props.children}
    </SolidErrorBoundary>
  );
}
