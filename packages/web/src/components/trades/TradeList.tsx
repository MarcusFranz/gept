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

  // Refresh trades from server
  const refreshTrades = async () => {
    try {
      const res = await fetch('/api/trades/active');
      const data = await res.json();
      if (data.success) {
        setTrades(data.data.map(toTradeViewModel));
      }
    } catch (error) {
      console.error('Failed to refresh trades:', error);
    }
  };

  // Handle check-in for a trade
  const handleCheckIn = async (tradeId: string, progress: number): Promise<{ guidance?: Guidance }> => {
    const res = await fetch(`/api/trades/active/${tradeId}/check-in`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ progress })
    });
    const data = await res.json();

    if (data.success) {
      // Update local state
      setTrades(prev => prev.map(t =>
        t.id === tradeId
          ? { ...t, progress, lastCheckIn: new Date() }
          : t
      ));
    }

    return { guidance: data.guidance };
  };

  // Handle advancing to next phase
  const handleAdvance = async (tradeId: string) => {
    const res = await fetch(`/api/trades/active/${tradeId}/advance`, {
      method: 'POST'
    });
    const data = await res.json();

    if (data.success) {
      if (data.message === 'Trade completed') {
        // Remove from list
        setTrades(prev => prev.filter(t => t.id !== tradeId));
        setExpandedId(null);
      } else {
        // Refresh to get updated phase
        await refreshTrades();
      }
    }
  };

  // Handle cancel
  const handleCancel = async (tradeId: string) => {
    if (!confirm('Are you sure you want to cancel this trade?')) return;

    const res = await fetch(`/api/trades/active/${tradeId}`, {
      method: 'DELETE'
    });

    if (res.ok) {
      setTrades(prev => prev.filter(t => t.id !== tradeId));
      setExpandedId(null);
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
          font-size: 1.5rem;
        }

        .trade-list-capital {
          color: var(--text-secondary, #aaa);
          font-size: 0.875rem;
        }

        .trade-list-empty {
          text-align: center;
          padding: 3rem 1rem;
          color: var(--text-secondary, #aaa);
        }

        .trade-list-cta {
          margin-top: 1rem;
          padding: 0.75rem 1.5rem;
          background: var(--accent, #4f46e5);
          color: white;
          border: none;
          border-radius: 8px;
          font-weight: 600;
          cursor: pointer;
        }

        .trade-list-cta:hover {
          background: var(--accent-hover, #4338ca);
        }

        .trade-list-items {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }
      `}</style>
    </div>
  );
}
