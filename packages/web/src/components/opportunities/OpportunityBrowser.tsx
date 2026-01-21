// packages/web/src/components/opportunities/OpportunityBrowser.tsx
import { createSignal, createEffect, For, Show, on } from 'solid-js';
import type { Opportunity, OpportunityFilters } from '../../lib/trade-types';
import { FILTER_STORAGE_KEY } from '../../lib/trade-types';
import { FilterBar } from './FilterBar';
import { OpportunityCard } from './OpportunityCard';

interface OpportunityBrowserProps {
  availableCapital: number;
  totalCapital: number;
  onTradeAdded: () => void;
}

export function OpportunityBrowser(props: OpportunityBrowserProps) {
  const [opportunities, setOpportunities] = createSignal<Opportunity[]>([]);
  const [filters, setFilters] = createSignal<OpportunityFilters>({});
  const [loading, setLoading] = createSignal(true);
  const [loadingMore, setLoadingMore] = createSignal(false);
  const [expandedId, setExpandedId] = createSignal<string | null>(null);
  const [addingId, setAddingId] = createSignal<string | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [hasMore, setHasMore] = createSignal(false);
  const [total, setTotal] = createSignal(0);

  // Load saved filters on mount
  createEffect(() => {
    try {
      const saved = localStorage.getItem(FILTER_STORAGE_KEY);
      if (saved) {
        setFilters(JSON.parse(saved));
      }
    } catch {}
  });

  // Fetch opportunities with server-side filtering
  const fetchOpportunities = async (append = false) => {
    if (append) {
      setLoadingMore(true);
    } else {
      setLoading(true);
      setOpportunities([]);
    }
    setError(null);

    try {
      const f = filters();
      const res = await fetch('/api/opportunities', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profitMin: f.profitMin,
          profitMax: f.profitMax,
          timeMin: f.timeMin,
          timeMax: f.timeMax,
          confidence: f.confidence,
          capitalMax: f.capitalMax || props.availableCapital,
          categories: f.categories,
          limit: 30,
          offset: append ? opportunities().length : 0
        })
      });

      if (!res.ok) throw new Error('Failed to load opportunities');
      const data = await res.json();

      if (data.success) {
        if (append) {
          setOpportunities(prev => [...prev, ...data.data.items]);
        } else {
          setOpportunities(data.data.items);
        }
        setHasMore(data.data.hasMore);
        setTotal(data.data.total);
      } else {
        setError(data.error || 'Failed to load opportunities');
      }
    } catch (err) {
      setError('Failed to load opportunities');
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  };

  // Initial fetch
  createEffect(() => {
    fetchOpportunities();
  });

  // Re-fetch when filters change (with debounce effect)
  createEffect(on(filters, () => {
    fetchOpportunities();
  }, { defer: true }));

  // Add to trades
  const handleAddToTrades = async (opp: Opportunity) => {
    setAddingId(opp.id);

    try {
      const res = await fetch('/api/trades/active', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          itemId: opp.itemId,
          itemName: opp.item,
          buyPrice: opp.buyPrice,
          sellPrice: opp.sellPrice,
          quantity: opp.quantity,
          recId: opp.id,
          expectedHours: opp.expectedHours
        })
      });

      if (!res.ok) throw new Error('Failed to add trade');
      const data = await res.json();

      if (data.success) {
        setOpportunities(prev => prev.filter(o => o.id !== opp.id));
        setExpandedId(null);
        props.onTradeAdded();
      } else {
        setError(data.error || 'Failed to add trade');
      }
    } catch {
      setError('Failed to add trade');
    } finally {
      setAddingId(null);
    }
  };

  const formatGold = (amount: number) => {
    if (amount >= 1_000_000) {
      return (amount / 1_000_000).toFixed(1) + 'M';
    }
    return Math.round(amount / 1_000) + 'K';
  };

  return (
    <div class="opportunity-browser">
      <header class="opportunity-browser-header">
        <h1>Opportunities</h1>
        <span class="opportunity-browser-capital">
          Available: {formatGold(props.availableCapital)}
        </span>
      </header>

      <FilterBar
        filters={filters()}
        onChange={setFilters}
        availableCapital={props.availableCapital}
      />

      <Show when={error()}>
        <div class="opportunity-browser-error">
          {error()}
          <button onClick={() => fetchOpportunities()}>Retry</button>
        </div>
      </Show>

      <Show
        when={!loading()}
        fallback={<div class="opportunity-browser-loading">Loading opportunities...</div>}
      >
        <Show
          when={opportunities().length > 0}
          fallback={
            <div class="opportunity-browser-empty">
              <p>No opportunities match your filters.</p>
              <button onClick={() => setFilters({})}>Clear filters</button>
            </div>
          }
        >
          <div class="opportunity-browser-count">
            Showing {opportunities().length} of {total()} opportunities
          </div>

          <div class="opportunity-browser-list">
            <For each={opportunities()}>
              {(opp) => (
                <OpportunityCard
                  opportunity={opp}
                  expanded={expandedId() === opp.id}
                  onClick={() => setExpandedId(expandedId() === opp.id ? null : opp.id)}
                  onAddToTrades={() => handleAddToTrades(opp)}
                  loading={addingId() === opp.id}
                />
              )}
            </For>
          </div>
        </Show>
      </Show>

      <Show when={hasMore() && !loading()}>
        <button
          class="opportunity-browser-load-more"
          onClick={() => fetchOpportunities(true)}
          disabled={loadingMore()}
        >
          {loadingMore() ? 'Loading...' : 'Load more opportunities'}
        </button>
      </Show>

      <style>{`
        .opportunity-browser {
          max-width: 600px;
          margin: 0 auto;
          padding: 1rem;
        }

        .opportunity-browser-count {
          font-size: 0.875rem;
          color: var(--text-secondary, #aaa);
          margin-bottom: 0.75rem;
        }

        .opportunity-browser-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }

        .opportunity-browser-header h1 {
          margin: 0;
          font-size: 1.5rem;
        }

        .opportunity-browser-capital {
          color: var(--text-secondary, #aaa);
          font-size: 0.875rem;
        }

        .opportunity-browser-error {
          background: var(--error-bg, #3d2020);
          color: var(--error, #ef4444);
          padding: 1rem;
          border-radius: 8px;
          margin-bottom: 1rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .opportunity-browser-error button {
          background: var(--error, #ef4444);
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 4px;
          cursor: pointer;
        }

        .opportunity-browser-loading,
        .opportunity-browser-empty {
          text-align: center;
          padding: 3rem 1rem;
          color: var(--text-secondary, #aaa);
        }

        .opportunity-browser-empty button {
          margin-top: 1rem;
          background: var(--accent, #4f46e5);
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 4px;
          cursor: pointer;
        }

        .opportunity-browser-list {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .opportunity-browser-load-more {
          width: 100%;
          margin-top: 1rem;
          padding: 0.75rem;
          background: var(--surface-2, #1a1a2e);
          color: var(--text-secondary, #aaa);
          border: 1px solid var(--border, #333);
          border-radius: 8px;
          cursor: pointer;
        }

        .opportunity-browser-load-more:hover {
          border-color: var(--accent, #4f46e5);
          color: var(--text-primary, #fff);
        }
      `}</style>
    </div>
  );
}
