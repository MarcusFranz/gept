import { createSignal, createEffect, Show, For } from 'solid-js';
import type { TradeHistory } from '../lib/db';
import { formatGold } from '../lib/types';

type FilterType = 'all' | 'wins' | 'losses' | 'cancelled';

export default function TradeHistoryView() {
  const [trades, setTrades] = createSignal<TradeHistory[]>([]);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [filter, setFilter] = createSignal<FilterType>('all');
  const [page, setPage] = createSignal(1);
  const [totalPages, setTotalPages] = createSignal(1);
  const [totalCount, setTotalCount] = createSignal(0);
  const pageSize = 10;

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        filter: filter(),
        page: page().toString(),
        limit: pageSize.toString()
      });

      const response = await fetch(`/api/trades/history?${params}`);
      const data = await response.json();

      if (data.success) {
        setTrades(data.data || []);
        setTotalCount(data.total || 0);
        setTotalPages(Math.ceil((data.total || 0) / pageSize));
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('Failed to load trade history');
    } finally {
      setLoading(false);
    }
  };

  createEffect(() => {
    // Re-fetch when filter or page changes
    const _ = filter();
    const __ = page();
    fetchHistory();
  });

  const changeFilter = (newFilter: FilterType) => {
    setFilter(newFilter);
    setPage(1);
  };

  const stats = () => {
    const all = trades();
    const totalProfit = all.reduce((sum, t) => sum + t.profit, 0);
    const wins = all.filter(t => t.profit > 0 && t.status === 'completed').length;
    const losses = all.filter(t => t.profit < 0 && t.status === 'completed').length;
    return { totalProfit, wins, losses };
  };

  return (
    <div class="trade-history">
      <div class="history-header">
        <h2>Trade History</h2>
        <p class="text-secondary text-sm">{totalCount()} trades total</p>
      </div>

      <div class="tabs">
        <button
          class={`tab ${filter() === 'all' ? 'active' : ''}`}
          onClick={() => changeFilter('all')}
        >
          All
        </button>
        <button
          class={`tab ${filter() === 'wins' ? 'active' : ''}`}
          onClick={() => changeFilter('wins')}
        >
          Wins
        </button>
        <button
          class={`tab ${filter() === 'losses' ? 'active' : ''}`}
          onClick={() => changeFilter('losses')}
        >
          Losses
        </button>
        <button
          class={`tab ${filter() === 'cancelled' ? 'active' : ''}`}
          onClick={() => changeFilter('cancelled')}
        >
          Cancelled
        </button>
      </div>

      <Show when={loading()}>
        <div class="history-loading">
          <div class="spinner"></div>
        </div>
      </Show>

      <Show when={error()}>
        <div class="history-error card">
          <p class="text-danger">{error()}</p>
          <button class="btn btn-secondary btn-sm mt-2" onClick={fetchHistory}>
            Try Again
          </button>
        </div>
      </Show>

      <Show when={!loading() && !error() && trades().length === 0}>
        <div class="empty-state card">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 8v4l3 3"/>
            <circle cx="12" cy="12" r="10"/>
          </svg>
          <p>No trades found</p>
          <Show when={filter() !== 'all'}>
            <button class="btn btn-secondary btn-sm mt-2" onClick={() => changeFilter('all')}>
              View All
            </button>
          </Show>
        </div>
      </Show>

      <Show when={!loading() && !error() && trades().length > 0}>
        <div class="history-table-wrapper">
          <table class="table">
            <thead>
              <tr>
                <th>Item</th>
                <th>Buy</th>
                <th>Sell</th>
                <th>Qty</th>
                <th>Profit</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              <For each={trades()}>
                {(trade) => (
                  <tr class={trade.status === 'cancelled' ? 'cancelled' : ''}>
                    <td>
                      <span class="trade-item-name">{trade.item_name}</span>
                      <Show when={trade.status === 'cancelled'}>
                        <span class="badge badge-warning ml-2">Cancelled</span>
                      </Show>
                    </td>
                    <td class="font-mono">
                      {trade.buy_price ? formatGold(trade.buy_price, 'gp') : '-'}
                    </td>
                    <td class="font-mono">
                      {trade.sell_price ? formatGold(trade.sell_price, 'gp') : '-'}
                    </td>
                    <td class="font-mono">
                      {trade.quantity?.toLocaleString() || '-'}
                    </td>
                    <td class={`font-mono ${trade.profit >= 0 ? 'text-success' : 'text-danger'}`}>
                      {trade.profit >= 0 ? '+' : ''}{formatGold(trade.profit)}
                    </td>
                    <td class="text-muted">
                      {new Date(trade.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                )}
              </For>
            </tbody>
          </table>
        </div>

        <Show when={totalPages() > 1}>
          <div class="pagination">
            <button
              class="btn btn-secondary btn-sm"
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page() === 1}
            >
              Previous
            </button>
            <span class="pagination-info">
              Page {page()} of {totalPages()}
            </span>
            <button
              class="btn btn-secondary btn-sm"
              onClick={() => setPage(p => Math.min(totalPages(), p + 1))}
              disabled={page() === totalPages()}
            >
              Next
            </button>
          </div>
        </Show>
      </Show>

      <style>{`
        .trade-history {
          max-width: 900px;
          margin: 0 auto;
        }

        .history-header {
          margin-bottom: var(--space-4);
        }

        .history-loading, .history-error {
          text-align: center;
          padding: var(--space-6);
        }

        .history-loading .spinner {
          margin: 0 auto;
        }

        .history-table-wrapper {
          overflow-x: auto;
          background-color: var(--bg-secondary);
          border-radius: var(--radius-lg);
          border: 1px solid var(--border);
        }

        .table {
          min-width: 600px;
        }

        .table tbody tr.cancelled {
          opacity: 0.6;
        }

        .trade-item-name {
          font-weight: 500;
        }

        @media (max-width: 640px) {
          .table th, .table td {
            padding: var(--space-2);
            font-size: var(--font-size-sm);
          }
        }
      `}</style>
    </div>
  );
}
