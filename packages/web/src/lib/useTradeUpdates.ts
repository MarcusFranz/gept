import { createSignal, createEffect, onCleanup } from 'solid-js';
import type { ActiveTrade } from './db';
import type { UpdateRecommendation, UpdateCheckResponse } from './types';

const DEFAULT_INTERVAL = 30_000; // 30s
const MIN_INTERVAL = 15_000;
const MAX_INTERVAL = 60_000;

export function useTradeUpdates(trades: () => ActiveTrade[]) {
  const [updates, setUpdates] = createSignal<Map<string, UpdateRecommendation>>(new Map());
  let timer: ReturnType<typeof setTimeout> | null = null;
  let polling = false;

  const poll = async () => {
    if (polling) return;
    polling = true;

    try {
      const res = await fetch('/api/trades/updates');
      if (!res.ok) return;

      const json = await res.json();
      if (!json.success || !json.data) return;

      const data = json.data as UpdateCheckResponse;

      if (data.updates.length > 0) {
        setUpdates(prev => {
          const next = new Map(prev);
          for (const u of data.updates) {
            next.set(u.tradeId, u);
          }
          return next;
        });
      }

      // Schedule next poll using engine's suggested interval
      scheduleNext(data.nextCheckIn);
    } catch {
      // Silently back off on failure
      scheduleNext(60);
    } finally {
      polling = false;
    }
  };

  const scheduleNext = (seconds: number) => {
    if (timer) clearTimeout(timer);
    const ms = Math.max(MIN_INTERVAL, Math.min(seconds * 1000, MAX_INTERVAL));
    timer = setTimeout(poll, ms);
  };

  // Start/stop polling reactively based on trade count
  createEffect(() => {
    const activeTrades = trades();
    if (timer) clearTimeout(timer);

    if (activeTrades.length > 0) {
      // Initial poll after a short delay (let SSE connect first)
      timer = setTimeout(poll, 5_000);
    }
  });

  onCleanup(() => {
    if (timer) clearTimeout(timer);
  });

  const dismissUpdate = (tradeId: string) => {
    setUpdates(prev => {
      const next = new Map(prev);
      next.delete(tradeId);
      return next;
    });
  };

  return { updates, dismissUpdate };
}
