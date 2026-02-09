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
const isDev = typeof import.meta !== 'undefined' && (import.meta as any).env?.DEV;

const buildMockOpportunities = (): Opportunity[] => ([
  {
    id: '9QK4M7A',
    itemId: 560,
    item: 'Death rune',
    buyPrice: 196,
    sellPrice: 212,
    quantity: 7000,
    capitalRequired: 1372000,
    expectedProfit: 112000,
    expectedHours: 2.3,
    confidence: 'high',
    fillProbability: 0.36,
    volume24h: 240000,
    trend: 'up',
    category: 'runes',
    whyChips: [
      { icon: '🎯', label: 'High confidence', type: 'positive' },
      { icon: '⚡', label: 'Fast fill', type: 'positive' },
      { icon: '📈', label: 'Trending up', type: 'positive' },
      { icon: '🔥', label: 'High volume', type: 'positive' },
    ],
  },
  {
    id: 'V7X2L9C',
    itemId: 563,
    item: 'Law rune',
    buyPrice: 252,
    sellPrice: 270,
    quantity: 6000,
    capitalRequired: 1512000,
    expectedProfit: 108000,
    expectedHours: 3.8,
    confidence: 'medium',
    fillProbability: 0.28,
    volume24h: 190000,
    trend: 'up',
    category: 'runes',
    whyChips: [
      { icon: '🎯', label: 'Med confidence', type: 'neutral' },
      { icon: '📊', label: 'Good volume', type: 'neutral' },
      { icon: '⚡', label: 'Fast fill', type: 'positive' },
    ],
  },
  {
    id: 'P4N8Z1H',
    itemId: 6686,
    item: 'Saradomin brew(4)',
    buyPrice: 7420,
    sellPrice: 7700,
    quantity: 500,
    capitalRequired: 3710000,
    expectedProfit: 140000,
    expectedHours: 4.6,
    confidence: 'medium',
    fillProbability: 0.22,
    volume24h: 68000,
    trend: 'down',
    category: 'potions',
    whyChips: [
      { icon: '📊', label: 'Good volume', type: 'neutral' },
      { icon: '📉', label: 'Trending down', type: 'negative' },
      { icon: '🕐', label: 'Longer hold', type: 'neutral' },
    ],
  },
  {
    id: 'T6R3W8M',
    itemId: 995,
    item: 'Dragon bones',
    buyPrice: 2840,
    sellPrice: 3000,
    quantity: 1200,
    capitalRequired: 3408000,
    expectedProfit: 192000,
    expectedHours: 2.0,
    confidence: 'high',
    fillProbability: 0.34,
    volume24h: 150000,
    trend: 'up',
    category: 'bones',
    whyChips: [
      { icon: '🎯', label: 'High confidence', type: 'positive' },
      { icon: '⚡', label: 'Fast fill', type: 'positive' },
      { icon: '📈', label: 'Trending up', type: 'positive' },
    ],
  },
  {
    id: 'K5D2J7Q',
    itemId: 5295,
    item: 'Ranarr weed',
    buyPrice: 7350,
    sellPrice: 7640,
    quantity: 350,
    capitalRequired: 2572500,
    expectedProfit: 101500,
    expectedHours: 3.2,
    confidence: 'medium',
    fillProbability: 0.26,
    volume24h: 52000,
    trend: 'up',
    category: 'herbs',
    whyChips: [
      { icon: '📊', label: 'Good volume', type: 'neutral' },
      { icon: '⏱', label: 'Quick flip', type: 'positive' },
    ],
  },
  {
    id: 'B8S4Y2F',
    itemId: 245,
    item: 'Zamorak wine',
    buyPrice: 720,
    sellPrice: 780,
    quantity: 3000,
    capitalRequired: 2160000,
    expectedProfit: 180000,
    expectedHours: 1.6,
    confidence: 'high',
    fillProbability: 0.4,
    volume24h: 98000,
    trend: 'up',
    category: 'ingredients',
    whyChips: [
      { icon: '🎯', label: 'High confidence', type: 'positive' },
      { icon: '⚡', label: 'Fast fill', type: 'positive' },
      { icon: '📈', label: 'Trending up', type: 'positive' },
    ],
  },
]);

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
  const [isBeta, setIsBeta] = createSignal(false);

  const injectMockOpportunities = () => {
    const items = buildMockOpportunities();
    setOpportunities(items);
    setHasMore(false);
    setTotal(items.length);
    setIsBeta(false);
    setError(null);
  };

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

      if (!res.ok) {
        if (isDev) {
          injectMockOpportunities();
          return;
        }
        throw new Error('Failed to load opportunities');
      }
      const data = await res.json();

      if (data.success) {
        if (isDev && (!data.data?.items || data.data.items.length === 0)) {
          injectMockOpportunities();
          return;
        }
        if (append) {
          setOpportunities(prev => [...prev, ...data.data.items]);
        } else {
          setOpportunities(data.data.items);
        }
        setHasMore(data.data.hasMore);
        setTotal(data.data.total);
        if (data.isBeta !== undefined) setIsBeta(data.isBeta);
      } else {
        if (isDev) {
          injectMockOpportunities();
          return;
        }
        setError(data.error || 'Failed to load opportunities');
      }
    } catch (err) {
      if (isDev) {
        injectMockOpportunities();
      } else {
        setError('Failed to load opportunities');
      }
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
  const handleAddToTrades = async (opp: Opportunity, quantityOverride?: number) => {
    setAddingId(opp.id);
    const quantity = quantityOverride ?? opp.quantity;

    try {
      const res = await fetch('/api/trades/active', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          itemId: opp.itemId,
          itemName: opp.item,
          buyPrice: opp.buyPrice,
          sellPrice: opp.sellPrice,
          offsetPct: opp.offsetPct ?? undefined,
          quantity,
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

      <FilterBar
        filters={filters()}
        onChange={setFilters}
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
                  onAddToTrades={(qty) => handleAddToTrades(opp, qty)}
                  loading={addingId() === opp.id}
                  isBeta={isBeta()}
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
          padding: 1rem;
          padding-bottom: 5rem;
        }

        .opportunity-browser-count {
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
          margin-bottom: 0.75rem;
        }

        .opportunity-browser-error {
          background: var(--surface-2);
          color: var(--danger);
          padding: 1rem;
          border-radius: var(--radius-lg);
          border: 1px solid color-mix(in srgb, var(--danger) 35%, transparent);
          margin-bottom: 1rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .opportunity-browser-error button {
          background: var(--danger);
          color: var(--btn-text-light);
          border: 1px solid color-mix(in srgb, var(--danger) 65%, transparent);
          padding: 0.5rem 1rem;
          border-radius: var(--radius-full);
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
          background: var(--action);
          color: var(--btn-text-dark);
          border: none;
          padding: 0.5rem 1rem;
          border-radius: var(--radius-full);
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
          background: var(--surface-1);
          color: var(--text-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          cursor: pointer;
        }

        .opportunity-browser-load-more:hover {
          border-color: var(--accent);
          color: var(--text-primary);
        }

        .opportunity-browser-warning {
          display: flex;
          align-items: flex-start;
          gap: 0.75rem;
          background: var(--surface-2);
          border: 1px solid color-mix(in srgb, var(--warning) 40%, transparent);
          border-radius: var(--radius-lg);
          padding: 0.75rem 1rem;
          margin-bottom: 0.65rem;
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
