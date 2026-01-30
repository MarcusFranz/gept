import { createSignal, onMount, onCleanup } from 'solid-js';
import type { UpdateRecommendation, SSEMessage } from './types';

export function useAlerts() {
  const [alerts, setAlerts] = createSignal<Map<string, UpdateRecommendation>>(new Map());
  const [isConnected, setIsConnected] = createSignal(false);

  let eventSource: EventSource | null = null;
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  let reconnectDelay = 1000; // Start at 1s, max 30s

  const connect = () => {
    // Don't connect if already connected
    if (eventSource?.readyState === EventSource.OPEN || eventSource?.readyState === EventSource.CONNECTING) return;

    eventSource = new EventSource('/api/events/stream');

    eventSource.onopen = () => {
      setIsConnected(true);
      reconnectDelay = 1000; // Reset backoff
    };

    eventSource.onmessage = (event) => {
      try {
        const message: SSEMessage = JSON.parse(event.data);

        if (message.type === 'ALERT' && message.data) {
          setAlerts(prev => {
            const next = new Map(prev);
            next.set(message.data!.tradeId, message.data!);
            return next;
          });
        } else if (message.type === 'CONNECTED') {
          setIsConnected(true);
        }
      } catch (err) {
        console.error('Failed to parse SSE message:', err);
      }
    };

    eventSource.onerror = () => {
      setIsConnected(false);
      eventSource?.close();
      eventSource = null;
      // Reconnect with exponential backoff
      reconnectTimeout = setTimeout(() => {
        reconnectDelay = Math.min(reconnectDelay * 2, 30000);
        connect();
      }, reconnectDelay);
    };
  };

  const dismissAlert = (tradeId: string) => {
    setAlerts(prev => {
      const next = new Map(prev);
      next.delete(tradeId);
      return next;
    });
  };

  const clearAlert = (tradeId: string) => dismissAlert(tradeId);

  onMount(() => {
    connect();
  });

  onCleanup(() => {
    if (reconnectTimeout) clearTimeout(reconnectTimeout);
    eventSource?.close();
  });

  return { alerts, isConnected, dismissAlert, clearAlert };
}
