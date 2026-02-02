// packages/web/src/components/opportunities/OpportunityBrowser.tsx
import { createSignal, createEffect, createMemo, For, Show, on } from 'solid-js';
import type { Opportunity, OpportunityFilters } from '../../lib/trade-types';
import { FILTER_STORAGE_KEY } from '../../lib/trade-types';
import { FilterBar } from './FilterBar';
import { OpportunityCard } from './OpportunityCard';
import { addToast } from '../ToastContainer';

interface OpportunityBrowserProps {
  activeTradeItemIds: number[];
  onTradeAdded: () => void;
  onNavigateToTrades: () => void;
}

const WIKI_WARNING_DISMISSED_KEY = 'gept:wiki-warning-dismissed';

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
  const [wikiWarningDismissed, setWikiWarningDismissed] = createSignal(
    typeof localStorage !== 'undefined' && localStorage.getItem(WIKI_WARNING_DISMISSED_KEY) === '1'
  );

  // Load saved filters on mount
  createEffect(() => {
    try {
      const saved = localStorage.getItem(FILTER_STORAGE_KEY);
      if (saved) {
        setFilters(JSON.parse(saved));
      }
    } catch (err) { console.warn('[OpportunityBrowser] Failed to load saved filters:', err); }
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
      // Convert minimum confidence level to the array of levels the engine accepts
      const confidenceMap: Record<string, string[]> = {
        high: ['high'],
        medium: ['medium', 'high'],
        low: ['low', 'medium', 'high'],
      };
      const confidenceArr = f.confidence ? confidenceMap[f.confidence] : undefined;
      const res = await fetch('/api/opportunities', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profitMin: f.profitMin,
          profitMax: f.profitMax,
          timeMax: f.timeMax,
          confidence: confidenceArr,
          capitalMax: f.capitalMax,
          category: f.category,
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

  // Fetch when filters change (including initial load from localStorage)
  // The defer: false ensures initial fetch happens immediately
  createEffect(on(filters, () => {
    fetchOpportunities();
  }));

  // Filter out opportunities for items the user already has as active trades
  const visibleOpportunities = createMemo(() =>
    opportunities().filter(o => !props.activeTradeItemIds.includes(o.itemId))
  );

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
          expectedHours: opp.expectedHours,
          modelId: opp.modelId,
          confidence: opp.confidence,
          fillProbability: opp.fillProbability,
          expectedProfit: opp.expectedProfit
        })
      });

      if (!res.ok) throw new Error('Failed to add trade');
      const data = await res.json();

      if (data.success) {
        setOpportunities(prev => prev.filter(o => o.id !== opp.id));
        setExpandedId(null);
        props.onTradeAdded();

        addToast({
          type: 'success',
          title: 'Trade added',
          message: `${opp.item} added to your trades`,
          duration: 5000,
          action: {
            label: 'View Trades',
            onClick: () => props.onNavigateToTrades()
          }
        });
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
      </header>

      <FilterBar
        filters={filters()}
        onChange={setFilters}
      />

      <Show when={!wikiWarningDismissed()}>
        <div class="opportunity-browser-warning">
          <div class="opportunity-browser-warning-content">
            <span class="opportunity-browser-warning-icon">⚠️</span>
            <p>
              Always verify trades on the{' '}
              <a href="https://prices.runescape.wiki/osrs/" target="_blank" rel="noopener noreferrer">
                OSRS Wiki price charts
              </a>{' '}
              before committing GP. Predictions are estimates, not guarantees.
            </p>
          </div>
          <button
            class="opportunity-browser-warning-dismiss"
            onClick={() => {
              setWikiWarningDismissed(true);
              localStorage.setItem(WIKI_WARNING_DISMISSED_KEY, '1');
            }}
            aria-label="Dismiss warning"
          >
            ✕
          </button>
        </div>
      </Show>

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
          when={visibleOpportunities().length > 0}
          fallback={
            <div class="opportunity-browser-empty">
              <p>No opportunities match your filters.</p>
              <button onClick={() => setFilters({})}>Clear filters</button>
            </div>
          }
        >
          <div class="opportunity-browser-count">
            Showing {visibleOpportunities().length} of {total()} opportunities
          </div>

          <div class="opportunity-browser-list">
            <For each={visibleOpportunities()}>
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
          {loadingMore() ? 'Loading...' : `Load more (${total() - visibleOpportunities().length} remaining)`}
        </button>
      </Show>

      <style>{`
        .opportunity-browser {
          max-width: 720px;
          margin: 0 auto;
          padding: 1.5rem 1rem 5rem;
        }

        .opportunity-browser-count {
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
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
          font-size: var(--font-size-2xl);
        }

        .opportunity-browser-error {
          background: linear-gradient(145deg, rgba(255, 107, 107, 0.2), rgba(255, 107, 107, 0.06));
          color: var(--danger);
          border: 1px solid rgba(255, 107, 107, 0.4);
          padding: 1rem;
          border-radius: var(--radius-lg);
          margin-bottom: 1rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .opportunity-browser-error button {
          background: linear-gradient(135deg, var(--danger) 0%, #ff9a9a 100%);
          color: var(--btn-text-dark);
          border: none;
          padding: 0.5rem 1rem;
          border-radius: var(--radius-md);
          cursor: pointer;
        }

        .opportunity-browser-loading,
        .opportunity-browser-empty {
          text-align: center;
          padding: 3rem 1rem;
          color: var(--text-secondary);
        }

        .opportunity-browser-empty button {
          margin-top: 1rem;
          background: linear-gradient(135deg, var(--accent) 0%, var(--action) 100%);
          color: var(--btn-text-dark);
          border: none;
          padding: 0.5rem 1rem;
          border-radius: var(--radius-md);
          cursor: pointer;
          box-shadow: 0 10px 22px var(--accent-glow);
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
          background: var(--glass-bg);
          color: var(--text-secondary);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-lg);
          cursor: pointer;
          backdrop-filter: blur(10px);
        }

        .opportunity-browser-load-more:hover {
          border-color: var(--accent);
          color: var(--text-primary);
        }

        .opportunity-browser-warning {
          display: flex;
          align-items: flex-start;
          gap: 0.75rem;
          background: linear-gradient(145deg, rgba(255, 209, 102, 0.2), rgba(255, 209, 102, 0.06));
          border: 1px solid rgba(255, 209, 102, 0.5);
          border-radius: var(--radius-lg);
          padding: 0.75rem 1rem;
          margin-bottom: 1rem;
          box-shadow: var(--shadow-sm);
          backdrop-filter: blur(10px);
        }

        .opportunity-browser-warning-content {
          display: flex;
          align-items: flex-start;
          gap: 0.5rem;
          flex: 1;
          min-width: 0;
        }

        .opportunity-browser-warning-icon {
          flex-shrink: 0;
          font-size: var(--font-size-md);
          line-height: 1.5;
        }

        .opportunity-browser-warning-content p {
          margin: 0;
          font-size: var(--font-size-sm);
          color: var(--text-primary);
          line-height: 1.5;
        }

        .opportunity-browser-warning-content a {
          color: var(--accent);
          text-decoration: underline;
        }

        .opportunity-browser-warning-dismiss {
          flex-shrink: 0;
          background: none;
          border: none;
          color: var(--text-secondary);
          cursor: pointer;
          padding: 0;
          font-size: var(--font-size-sm);
          line-height: 1.5;
        }

        .opportunity-browser-warning-dismiss:hover {
          color: var(--text-primary);
        }
      `}</style>
    </div>
  );
}
