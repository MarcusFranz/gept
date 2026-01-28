// packages/web/src/components/trades/TradeList.tsx
import { createSignal, For, Show } from 'solid-js';
import type { ActiveTrade } from '../../lib/db';
import { toTradeViewModel, type TradeViewModel, type Guidance } from '../../lib/trade-types';
import { TradeCard } from './TradeCard';
import { TradeDetail } from './TradeDetail';

interface TradeListProps {
  initialTrades: ActiveTrade[];
  availableCapital: number;
  totalCapital: number;
  onNavigateToOpportunities: () => void;
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

  // Handle cancel
  const handleCancel = async (tradeId: string) => {
    if (!confirm('Are you sure you want to cancel this trade?')) return;

    try {
      const res = await fetch(`/api/trades/active/${tradeId}`, {
        method: 'DELETE'
      });
      if (!res.ok) throw new Error('Failed to cancel trade');

      setTrades(prev => prev.filter(t => t.id !== tradeId));
      setExpandedId(null);
      setError(null);
    } catch (err) {
      console.error('Cancel failed:', err);
      setError('Failed to cancel trade. Please try again.');
    }
  };

  const formatGold = (amount: number) => {
    if (amount >= 1_000_000) {
      return (amount / 1_000_000).toFixed(1) + 'M';
    }
    return Math.round(amount / 1_000) + 'K';
  };

  return (
    <div class="trade-list">
      <header class="trade-list-header">
        <h1>My Trades</h1>
        <span class="trade-list-capital">
          Available: {formatGold(props.availableCapital)} / {formatGold(props.totalCapital)}
        </span>
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
            <p>No active trades.</p>
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
                  />
                }
              >
                <TradeDetail
                  trade={trade}
                  onCheckIn={(progress) => handleCheckIn(trade.id, progress)}
                  onAdvance={() => handleAdvance(trade.id)}
                  onCancel={() => handleCancel(trade.id)}
                  onClose={() => setExpandedId(null)}
                />
              </Show>
            )}
          </For>
        </div>
      </Show>

      <style>{`
        .trade-list {
          max-width: 600px;
          margin: 0 auto;
          padding: 1rem;
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
        }

        .trade-list-capital {
          color: var(--text-secondary);
          font-size: var(--font-size-sm);
        }

        .trade-list-empty {
          text-align: center;
          padding: 3rem 1rem;
          color: var(--text-secondary);
        }

        .trade-list-cta {
          margin-top: 1rem;
          padding: 0.75rem 1.5rem;
          background: var(--accent);
          color: var(--btn-text-dark);
          border: none;
          border-radius: var(--radius-lg);
          font-weight: 600;
          cursor: pointer;
        }

        .trade-list-cta:hover {
          background: var(--accent-hover);
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
          background: var(--danger-light);
          color: var(--danger);
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
