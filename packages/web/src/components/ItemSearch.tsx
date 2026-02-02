import { createSignal, createEffect, onCleanup, Show, For } from 'solid-js';
import type { ItemSearchResult, Recommendation } from '../lib/types';
import { formatGold, formatPercent } from '../lib/types';

export default function ItemSearch() {
  const [query, setQuery] = createSignal('');
  const [results, setResults] = createSignal<ItemSearchResult[]>([]);
  const [selectedItem, setSelectedItem] = createSignal<ItemSearchResult | null>(null);
  const [recommendation, setRecommendation] = createSignal<Recommendation | null>(null);
  const [loading, setLoading] = createSignal(false);
  const [searchLoading, setSearchLoading] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [showDropdown, setShowDropdown] = createSignal(false);

  let searchTimeout: ReturnType<typeof setTimeout>;

  const searchItems = async (searchQuery: string) => {
    if (searchQuery.length < 2) {
      setResults([]);
      return;
    }

    setSearchLoading(true);
    try {
      const response = await fetch(`/api/items/search?q=${encodeURIComponent(searchQuery)}`);
      const data = await response.json();
      if (data.success) {
        setResults(data.data || []);
      }
    } catch (err) {
      console.error('Search error:', err);
    } finally {
      setSearchLoading(false);
    }
  };

  createEffect(() => {
    clearTimeout(searchTimeout);
    const q = query();
    if (q.length >= 2) {
      searchTimeout = setTimeout(() => searchItems(q), 300);
    } else {
      setResults([]);
    }
    onCleanup(() => clearTimeout(searchTimeout));
  });

  const selectItem = async (item: ItemSearchResult) => {
    setSelectedItem(item);
    setQuery(item.name);
    setShowDropdown(false);
    setLoading(true);
    setError(null);
    setRecommendation(null);

    try {
      const response = await fetch(`/api/items/${item.id}`);
      const data = await response.json();
      if (data.success) {
        setRecommendation(data.data);
      } else if (data.notRecommended) {
        setError('This item is not currently recommended for flipping.');
      } else {
        setError(data.error || 'Failed to get item details');
      }
    } catch (err) {
      setError('Failed to fetch item details');
    } finally {
      setLoading(false);
    }
  };

  const clearSelection = () => {
    setSelectedItem(null);
    setRecommendation(null);
    setQuery('');
    setError(null);
  };

  return (
    <div class="item-search">
      <div class="search-container">
        <div class="search-input-wrapper">
          <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"/>
            <path d="M21 21l-4.35-4.35"/>
          </svg>
          <input
            type="text"
            class="input search-input"
            placeholder="Search items (e.g., 'AGS' or 'Armadyl Godsword')"
            value={query()}
            onInput={(e) => {
              setQuery(e.currentTarget.value);
              setShowDropdown(true);
            }}
            onFocus={() => setShowDropdown(true)}
          />
          <Show when={query().length > 0}>
            <button class="search-clear" onClick={clearSelection}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M18 6L6 18M6 6l12 12"/>
              </svg>
            </button>
          </Show>
        </div>

        <Show when={showDropdown() && results().length > 0}>
          <div class="search-dropdown">
            <For each={results()}>
              {(item) => (
                <button
                  class="search-result"
                  onClick={() => selectItem(item)}
                >
                  <span class="search-result-name">{item.name}</span>
                  <Show when={item.acronym}>
                    <span class="search-result-acronym">({item.acronym})</span>
                  </Show>
                </button>
              )}
            </For>
          </div>
        </Show>

        <Show when={searchLoading()}>
          <div class="search-dropdown">
            <div class="search-loading">
              <div class="spinner"></div>
              <span>Searching...</span>
            </div>
          </div>
        </Show>
      </div>

      <Show when={loading()}>
        <div class="item-loading card">
          <div class="spinner"></div>
          <p class="text-secondary mt-3">Loading item details...</p>
        </div>
      </Show>

      <Show when={error()}>
        <div class="item-error card">
          <p class="text-center text-secondary">{error()}</p>
        </div>
      </Show>

      <Show when={recommendation()}>
        <div class="item-details card">
          <div class="item-header">
            <h2 class="item-name">{recommendation()!.item}</h2>
            <span class="badge badge-success">Recommended</span>
          </div>

          <div class="item-prices">
            <div class="item-price">
              <span class="item-price-label">Buy Price</span>
              <span class="item-price-value font-mono">
                {formatGold(recommendation()!.buyPrice, 'gp')}
              </span>
            </div>
            <div class="item-price">
              <span class="item-price-label">Sell Price</span>
              <span class="item-price-value font-mono">
                {formatGold(recommendation()!.sellPrice, 'gp')}
              </span>
            </div>
          </div>

          <div class="item-stats grid-4">
            <div class="item-stat">
              <span class="item-stat-value text-success font-mono">
                +{formatGold(recommendation()!.expectedProfit)}
              </span>
              <span class="item-stat-label">Expected Profit</span>
            </div>
            <div class="item-stat">
              <span class="item-stat-value font-mono">
                {formatGold(recommendation()!.capitalRequired)}
              </span>
              <span class="item-stat-label">Capital Required</span>
            </div>
            <div class="item-stat">
              <span class="item-stat-value font-mono">
                {formatPercent(recommendation()!.confidence)}
              </span>
              <span class="item-stat-label">Confidence</span>
            </div>
            <div class="item-stat">
              <span class="item-stat-value font-mono">
                {recommendation()!.expectedHours.toFixed(1)}h
              </span>
              <span class="item-stat-label">Expected Time</span>
            </div>
          </div>

          <div class="item-info">
            <div class="item-info-row">
              <span class="item-info-label">24h Volume</span>
              <span class="item-info-value">{recommendation()!.volume24h.toLocaleString()}</span>
            </div>
            <div class="item-info-row">
              <span class="item-info-label">Fill Probability</span>
              <span class="item-info-value">{formatPercent(recommendation()!.fillProbability)}</span>
            </div>
            <div class="item-info-row">
              <span class="item-info-label">Trend</span>
              <span class={`item-info-value ${recommendation()!.trend === 'up' ? 'text-success' : recommendation()!.trend === 'down' ? 'text-danger' : ''}`}>
                {recommendation()!.trend === 'up' ? '↑' : recommendation()!.trend === 'down' ? '↓' : '→'} {recommendation()!.trend}
              </span>
            </div>
          </div>

          <div class="item-reason">
            <p class="text-secondary text-sm">{recommendation()!.reason}</p>
          </div>

          <div class="item-actions">
            <button class="btn btn-primary">
              Track Trade
            </button>
            <button class="btn btn-secondary">
              Report
            </button>
          </div>
        </div>
      </Show>

      <style>{`
        .item-search {
          max-width: 600px;
          margin: 0 auto;
        }

        .search-container {
          position: relative;
          margin-bottom: var(--space-4);
        }

        .search-input-wrapper {
          position: relative;
        }

        .search-icon {
          position: absolute;
          left: var(--space-3);
          top: 50%;
          transform: translateY(-50%);
          width: 20px;
          height: 20px;
          color: var(--text-muted);
        }

        .search-input {
          padding-left: calc(var(--space-3) * 2 + 20px);
          padding-right: calc(var(--space-3) + 24px);
        }

        .search-clear {
          position: absolute;
          right: var(--space-2);
          top: 50%;
          transform: translateY(-50%);
          width: 28px;
          height: 28px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: none;
          border: none;
          color: var(--text-muted);
          cursor: pointer;
          border-radius: var(--radius-full);
          transition: all var(--transition-fast);
        }

        .search-clear:hover {
          background-color: var(--hover-bg);
          color: var(--text-primary);
        }

        .search-clear svg {
          width: 16px;
          height: 16px;
        }

        .search-dropdown {
          position: absolute;
          top: 100%;
          left: 0;
          right: 0;
          margin-top: var(--space-2);
          background-color: var(--surface-2);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-md);
          max-height: 300px;
          overflow-y: auto;
          z-index: 50;
          box-shadow: var(--shadow-lg);
          backdrop-filter: blur(12px);
        }

        .search-result {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          width: 100%;
          padding: var(--space-3) var(--space-4);
          text-align: left;
          background: none;
          border: none;
          color: var(--text-primary);
          cursor: pointer;
          transition: background-color var(--transition-fast);
        }

        .search-result:hover {
          background-color: var(--hover-bg);
        }

        .search-result-acronym {
          color: var(--text-muted);
          font-size: var(--font-size-sm);
        }

        .search-loading {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          padding: var(--space-3) var(--space-4);
          color: var(--text-secondary);
        }

        .item-loading, .item-error {
          text-align: center;
          padding: var(--space-6);
        }

        .item-loading .spinner {
          margin: 0 auto;
        }

        .item-details {
          margin-top: var(--space-4);
        }

        .item-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: var(--space-4);
        }

        .item-name {
          font-size: var(--font-size-xl);
        }

        .item-prices {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: var(--space-4);
          padding: var(--space-4);
          background-color: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-md);
          margin-bottom: var(--space-4);
        }

        .item-price {
          text-align: center;
        }

        .item-price-label {
          display: block;
          font-size: var(--font-size-xs);
          color: var(--text-muted);
          margin-bottom: var(--space-1);
        }

        .item-price-value {
          font-size: var(--font-size-lg);
          font-weight: 600;
        }

        .item-stats {
          margin-bottom: var(--space-4);
        }

        .item-stat {
          text-align: center;
          padding: var(--space-3);
          background-color: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-md);
        }

        .item-stat-value {
          display: block;
          font-size: var(--font-size-base);
          font-weight: 600;
          margin-bottom: var(--space-1);
        }

        .item-stat-label {
          font-size: var(--font-size-xs);
          color: var(--text-muted);
        }

        .item-info {
          padding: var(--space-3);
          background-color: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-md);
          margin-bottom: var(--space-4);
        }

        .item-info-row {
          display: flex;
          justify-content: space-between;
          padding: var(--space-2) 0;
        }

        .item-info-row:not(:last-child) {
          border-bottom: 1px solid var(--border);
        }

        .item-info-label {
          color: var(--text-secondary);
          font-size: var(--font-size-sm);
        }

        .item-info-value {
          font-weight: 500;
          font-size: var(--font-size-sm);
        }

        .item-reason {
          padding: var(--space-3);
          background-color: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-md);
          margin-bottom: var(--space-4);
        }

        .item-actions {
          display: flex;
          gap: var(--space-2);
          justify-content: flex-end;
        }

        @media (max-width: 640px) {
          .item-stats {
            grid-template-columns: 1fr 1fr;
          }
        }
      `}</style>
    </div>
  );
}
