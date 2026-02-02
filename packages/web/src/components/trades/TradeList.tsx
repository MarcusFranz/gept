// packages/web/src/components/trades/TradeList.tsx
import { createSignal, For, Show, onCleanup } from 'solid-js';
import type { ActiveTrade } from '../../lib/db';
import { toTradeViewModel, type TradeViewModel, type Guidance } from '../../lib/trade-types';
import type { UpdateRecommendation } from '../../lib/types';
import { TradeCard } from './TradeCard';
import { TradeDetail } from './TradeDetail';
import { addToast, removeToast } from '../ToastContainer';

interface TradeListProps {
  initialTrades: ActiveTrade[];
  onNavigateToOpportunities: () => void;
  alerts?: Map<string, UpdateRecommendation>;
  onAcceptAlert?: (tradeId: string, newSellPrice: number) => void;
  onDismissAlert?: (tradeId: string) => void;
}

export function TradeList(props: TradeListProps) {
  const [trades, setTrades] = createSignal<TradeViewModel[]>(
    props.initialTrades.map(toTradeViewModel)
  );
  const [expandedId, setExpandedId] = createSignal<string | null>(null);
  const [error, setError] = createSignal<string | null>(null);

  // Refresh trades from server
  const refreshTrades = async () => {
    try {
      const res = await fetch('/api/trades/active');
      if (!res.ok) throw new Error('Failed to load trades');
      const data = await res.json();
      if (data.success) {
        setTrades(data.data.map(toTradeViewModel));
      }
      setError(null);
    } catch (err) {
      console.error('Failed to refresh trades:', err);
      setError('Failed to load trades. Please try again.');
    }
  };

  // Handle check-in for a trade
  const handleCheckIn = async (tradeId: string, progress: number): Promise<{ guidance?: Guidance }> => {
    try {
      const res = await fetch(`/api/trades/active/${tradeId}/check-in`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ progress })
      });
      if (!res.ok) throw new Error('Failed to save progress');
      const data = await res.json();

      if (data.success) {
        setTrades(prev => prev.map(t =>
          t.id === tradeId
            ? { ...t, progress, lastCheckIn: new Date() }
            : t
        ));
        setError(null);
      }

      return { guidance: data.guidance };
    } catch (err) {
      console.error('Check-in failed:', err);
      setError('Failed to save progress. Please try again.');
      return {};
    }
  };

  // Handle advancing to next phase
  const handleAdvance = async (tradeId: string) => {
    try {
      const res = await fetch(`/api/trades/active/${tradeId}/advance`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error('Failed to advance trade');
      const data = await res.json();

      if (data.success) {
        if (data.message === 'Trade completed') {
          setTrades(prev => prev.filter(t => t.id !== tradeId));
          setExpandedId(null);
        } else {
          await refreshTrades();
        }
        setError(null);
      }
    } catch (err) {
      console.error('Advance failed:', err);
      setError('Failed to advance trade. Please try again.');
    }
  };

  // Track pending cancellations: tradeId → { timer, toastId, trade, index }
  const pendingCancels = new Map<string, { timer: ReturnType<typeof setTimeout>; toastId: string; trade: TradeViewModel; index: number }>();

  const UNDO_WINDOW_MS = 6000;

  // Commit a pending cancel to the backend
  const commitCancel = async (tradeId: string) => {
    pendingCancels.delete(tradeId);
    try {
      const res = await fetch(`/api/trades/active/${tradeId}`, {
        method: 'DELETE'
      });
      if (!res.ok) throw new Error('Failed to cancel trade');
      setError(null);
    } catch (err) {
      console.error('Cancel failed:', err);
      // Restore the trade since backend failed
      await refreshTrades();
      setError('Failed to cancel trade. Please try again.');
    }
  };

  // Handle cancel — optimistic removal + undo toast
  const handleCancel = (tradeId: string) => {
    const currentTrades = trades();
    const index = currentTrades.findIndex(t => t.id === tradeId);
    if (index === -1) return;
    const trade = currentTrades[index];

    // Optimistically remove from UI
    setTrades(prev => prev.filter(t => t.id !== tradeId));
    setExpandedId(null);

    // Schedule the actual DELETE
    const timer = setTimeout(() => {
      const pending = pendingCancels.get(tradeId);
      if (pending) {
        removeToast(pending.toastId);
        commitCancel(tradeId);
      }
    }, UNDO_WINDOW_MS);

    // Show undo toast
    const toastId = addToast({
      type: 'info',
      title: 'Trade cancelled',
      message: trade.itemName,
      duration: UNDO_WINDOW_MS,
      action: {
        label: 'Undo',
        onClick: () => {
          const pending = pendingCancels.get(tradeId);
          if (pending) {
            clearTimeout(pending.timer);
            removeToast(pending.toastId);
            pendingCancels.delete(tradeId);
            // Restore to original position
            setTrades(prev => {
              const restored = [...prev];
              const insertAt = Math.min(pending.index, restored.length);
              restored.splice(insertAt, 0, trade);
              return restored;
            });
          }
        }
      }
    });

    pendingCancels.set(tradeId, { timer, toastId, trade, index });
  };

  // Cleanup pending timers on unmount
  onCleanup(() => {
    for (const [tradeId, { timer }] of pendingCancels) {
      clearTimeout(timer);
      commitCancel(tradeId);
    }
  });

  return (
    <div class="trade-list">
      <header class="trade-list-header">
        <h1>My Trades</h1>
      </header>

      <Show when={error()}>
        <div class="trade-list-error">
          {error()}
          <button onClick={() => setError(null)}>×</button>
        </div>
      </Show>

      <Show
        when={trades().length > 0}
        fallback={
          <div class="trade-list-empty">
            <svg class="trade-list-empty-icon" viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M8 36V16l8-8h16l8 8v20a4 4 0 0 1-4 4H12a4 4 0 0 1-4-4z" />
              <polyline points="16 28 24 20 32 28" />
              <line x1="24" y1="20" x2="24" y2="36" />
            </svg>
            <p class="trade-list-empty-title">No active trades</p>
            <p class="trade-list-empty-subtitle">Browse opportunities and add your first trade to get started.</p>
            <button
              class="trade-list-cta"
              onClick={() => props.onNavigateToOpportunities()}
            >
              Find an opportunity
            </button>
          </div>
        }
      >
        <div class="trade-list-items">
          <For each={trades()}>
            {(trade) => (
              <Show
                when={expandedId() === trade.id}
                fallback={
                  <TradeCard
                    trade={trade}
                    onClick={() => setExpandedId(trade.id)}
                    onCancel={() => handleCancel(trade.id)}
                    alert={props.alerts?.get(trade.id)}
                  />
                }
              >
                <TradeDetail
                  trade={trade}
                  onCheckIn={(progress) => handleCheckIn(trade.id, progress)}
                  onAdvance={() => handleAdvance(trade.id)}
                  onCancel={() => handleCancel(trade.id)}
                  onClose={() => setExpandedId(null)}
                  alert={props.alerts?.get(trade.id)}
                  onAcceptAlert={() => {
                    const alert = props.alerts?.get(trade.id);
                    const newPrice = alert?.newSellPrice ?? alert?.adjustedSellPrice;
                    if (newPrice) {
                      props.onAcceptAlert?.(trade.id, newPrice);
                    }
                  }}
                  onDismissAlert={() => props.onDismissAlert?.(trade.id)}
                  onAcknowledgePrice={() => {
                    const suggestedPrice = trade.suggestedSellPrice;
                    if (suggestedPrice) {
                      props.onAcceptAlert?.(trade.id, suggestedPrice);
                    }
                  }}
                />
              </Show>
            )}
          </For>
        </div>
      </Show>

      <style>{`
        .trade-list {
          max-width: 720px;
          margin: 0 auto;
          padding: 1.5rem 1rem 5rem;
        }

        .trade-list-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1.5rem;
        }

        .trade-list-header h1 {
          margin: 0;
          font-size: var(--font-size-2xl);
          letter-spacing: -0.01em;
        }

        .trade-list-empty {
          text-align: center;
          padding: 3rem 1rem;
          color: var(--text-secondary);
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-lg);
          box-shadow: var(--shadow-sm);
          backdrop-filter: blur(10px);
        }

        .trade-list-empty-icon {
          width: 48px;
          height: 48px;
          color: var(--text-muted);
          margin-bottom: 0.75rem;
        }

        .trade-list-empty-title {
          font-size: var(--font-size-lg);
          font-weight: 600;
          color: var(--text-primary);
          margin: 0 0 0.25rem;
        }

        .trade-list-empty-subtitle {
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
          margin: 0;
        }

        .trade-list-cta {
          margin-top: 1rem;
          padding: 0.75rem 1.5rem;
          background: linear-gradient(135deg, var(--accent) 0%, var(--action) 100%);
          color: var(--btn-text-dark);
          border: none;
          border-radius: var(--radius-lg);
          font-weight: 600;
          cursor: pointer;
          box-shadow: 0 10px 22px var(--accent-glow);
        }

        .trade-list-cta:hover {
          filter: brightness(1.05);
        }

        .trade-list-items {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .trade-list-error {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.75rem 1rem;
          background: linear-gradient(145deg, rgba(255, 107, 107, 0.2), rgba(255, 107, 107, 0.06));
          color: var(--danger);
          border: 1px solid rgba(255, 107, 107, 0.4);
          border-radius: var(--radius-md);
          margin-bottom: 1rem;
          font-size: var(--font-size-sm);
        }

        .trade-list-error button {
          background: none;
          border: none;
          color: inherit;
          cursor: pointer;
          font-size: 1.25rem;
          padding: 0 0.25rem;
        }
      `}</style>
    </div>
  );
}
