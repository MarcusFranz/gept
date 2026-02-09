import { onMount, onCleanup } from 'solid-js';
import { addToast } from './ToastContainer';

/**
 * SessionMonitor - Detects session expiry and shows notification
 *
 * Monitors for:
 * 1. 401 responses from API calls (intercepted via global fetch wrapper)
 * 2. Session check on visibility change (tab focus)
 */

// Flag to prevent multiple session expiry notifications
let sessionExpiredNotified = false;

// Show session expired notification and redirect
function handleSessionExpired() {
  if (sessionExpiredNotified) return;
  sessionExpiredNotified = true;

  // Store current URL for redirect after login
  const currentPath = window.location.pathname;
  if (currentPath !== '/login' && currentPath !== '/signup' && currentPath !== '/welcome') {
    try {
      sessionStorage.setItem('redirectAfterLogin', currentPath);
    } catch {
      // Ignore sessionStorage errors
    }
  }

  addToast({
    key: 'session-expired',
    type: 'warning',
    title: 'Session Expired',
    message: 'Your session has expired. Please sign in again.',
    // Match the auto-redirect window so this doesn't linger indefinitely.
    duration: 5500,
    action: {
      label: 'Sign In',
      onClick: () => {
        window.location.href = '/login';
      }
    }
  });

  // Auto-redirect after 5 seconds if user doesn't click
  setTimeout(() => {
    if (window.location.pathname !== '/login') {
      window.location.href = '/login';
    }
  }, 5000);
}

// Intercept fetch to detect 401 responses
function setupFetchInterceptor() {
  const originalFetch = window.fetch;

  window.fetch = async (...args) => {
    const response = await originalFetch(...args);

    // Check for 401 on API routes
    const url = typeof args[0] === 'string' ? args[0] : args[0] instanceof URL ? args[0].href : args[0]?.url || '';
    if (url.includes('/api/') && response.status === 401) {
      handleSessionExpired();
    }

    return response;
  };

  return () => {
    window.fetch = originalFetch;
  };
}

// Check session on visibility change
async function checkSessionOnFocus() {
  if (document.visibilityState !== 'visible') return;

  try {
    // Make a lightweight API call to check session
    const response = await fetch('/api/settings', {
      method: 'GET',
      credentials: 'include'
    });

    if (response.status === 401) {
      handleSessionExpired();
    }
  } catch {
    // Network error - don't treat as session expiry
  }
}

export default function SessionMonitor() {
  onMount(() => {
    // Reset flag when component mounts (new page load)
    sessionExpiredNotified = false;

    // Setup fetch interceptor
    const cleanupFetch = setupFetchInterceptor();

    // Check session on tab focus
    document.addEventListener('visibilitychange', checkSessionOnFocus);

    // Check for redirect after login
    try {
      const redirectPath = sessionStorage.getItem('redirectAfterLogin');
      if (redirectPath && window.location.pathname === '/') {
        sessionStorage.removeItem('redirectAfterLogin');
        // Already on home, which is fine - the middleware would have redirected if not logged in
      }
    } catch {
      // Ignore sessionStorage errors
    }

    onCleanup(() => {
      cleanupFetch();
      document.removeEventListener('visibilitychange', checkSessionOnFocus);
    });
  });

  // This component doesn't render anything
  return null;
}
