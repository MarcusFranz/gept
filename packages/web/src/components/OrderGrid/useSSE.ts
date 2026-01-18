import { createSignal, createEffect, onCleanup } from 'solid-js';
import type { SSEMessage, UpdateRecommendation } from '../../lib/types';

interface UseSSEOptions {
  isSignedIn: () => boolean;
  hasActiveOrders: () => boolean;
  onUpdate: (updates: UpdateRecommendation[]) => void;
}

export function useSSE(options: UseSSEOptions) {
  const { isSignedIn, hasActiveOrders, onUpdate } = options;

  const [connected, setConnected] = createSignal(false);
  const [reconnectAttempts, setReconnectAttempts] = createSignal(0);

  let eventSource: EventSource | null = null;
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

  const connect = () => {
    if (typeof window === 'undefined') return;
    if (!isSignedIn()) return;
    if (!hasActiveOrders()) return;
    if (eventSource?.readyState === EventSource.OPEN) return;

    // Close any existing connection
    disconnect();

    try {
      eventSource = new EventSource('/api/sse/alerts');

      eventSource.onopen = () => {
        setConnected(true);
        setReconnectAttempts(0);
      };

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as SSEMessage;
          if (data.type === 'update' && data.updates) {
            onUpdate(data.updates);
          }
        } catch (err) {
          console.error('[SSE] Parse error:', err);
        }
      };

      eventSource.onerror = () => {
        setConnected(false);

        // Auto-reconnect with exponential backoff
        const attempts = reconnectAttempts();
        if (attempts < 5) {
          const delay = Math.min(1000 * Math.pow(2, attempts), 30000);
          setReconnectAttempts(attempts + 1);
          reconnectTimeout = setTimeout(connect, delay);
        }
      };
    } catch (err) {
      console.error('[SSE] Connection error:', err);
    }
  };

  const disconnect = () => {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout);
      reconnectTimeout = null;
    }

    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }

    setConnected(false);
  };

  // Connect/disconnect based on conditions
  createEffect(() => {
    if (isSignedIn() && hasActiveOrders()) {
      connect();
    } else {
      disconnect();
    }
  });

  // Handle visibility changes - disconnect when hidden, reconnect when visible
  createEffect(() => {
    if (typeof document === 'undefined') return;

    const handleVisibilityChange = () => {
      if (document.hidden) {
        disconnect();
      } else if (isSignedIn() && hasActiveOrders()) {
        connect();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    onCleanup(() => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    });
  });

  // Cleanup on unmount
  onCleanup(disconnect);

  return {
    connected,
    reconnectAttempts,
    connect,
    disconnect,
  };
}
