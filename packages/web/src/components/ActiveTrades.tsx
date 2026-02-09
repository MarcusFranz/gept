import { createSignal, createEffect, Show, For } from 'solid-js';
import type { ActiveTrade } from '../lib/db';
import { formatGold } from '../lib/types';
import { calculateFlipProfit } from '../lib/ge-tax';

interface ActiveTradesProps {
  initialTrades?: ActiveTrade[];
}

export default function ActiveTrades(props: ActiveTradesProps) {
  const [trades, setTrades] = createSignal<ActiveTrade[]>(props.initialTrades || []);
  const [loading, setLoading] = createSignal(!props.initialTrades);
  const [error, setError] = createSignal<string | null>(null);
  const [selectedTrade, setSelectedTrade] = createSignal<ActiveTrade | null>(null);
  const [showCompleteModal, setShowCompleteModal] = createSignal(false);
  const [completionData, setCompletionData] = createSignal({
    sellPrice: 0,
    quantity: 0,
    profit: 0,
    notes: ''
  });
  const [processing, setProcessing] = createSignal(false);

  const computeProfit = (trade: ActiveTrade, sellPrice: number, quantity: number) =>
    calculateFlipProfit(trade.buy_price, sellPrice, quantity);

  const fetchTrades = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/trades/active');
      const data = await response.json();
      if (data.success) {
        setTrades(data.data || []);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('Failed to load trades');
    } finally {
      setLoading(false);
    }
  };

  createEffect(() => {
    if (!props.initialTrades) {
      fetchTrades();
    }
  });

  const tiedCapital = () => {
    return trades().reduce((sum, t) => sum + t.buy_price * t.quantity, 0);
  };

  const openCompleteModal = (trade: ActiveTrade) => {
    setSelectedTrade(trade);
    const sellPrice = trade.sell_price;
    const quantity = trade.quantity;
    setCompletionData({
      sellPrice,
      quantity,
      profit: computeProfit(trade, sellPrice, quantity),
      notes: ''
    });
    setShowCompleteModal(true);
  };

  const completeTrade = async () => {
    const trade = selectedTrade();
    if (!trade) return;

    setProcessing(true);
    try {
      const response = await fetch(`/api/trades/active/${trade.id}/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(completionData())
      });

      const data = await response.json();
      if (data.success) {
        setTrades(trades().filter(t => t.id !== trade.id));
        setShowCompleteModal(false);
        setSelectedTrade(null);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('Failed to complete trade');
    } finally {
      setProcessing(false);
    }
  };

  const cancelTrade = async (trade: ActiveTrade) => {
    if (!confirm(`Cancel trade for ${trade.item_name}?`)) return;

    try {
      const response = await fetch(`/api/trades/active/${trade.id}`, {
        method: 'DELETE'
      });

      const data = await response.json();
      if (data.success) {
        setTrades(trades().filter(t => t.id !== trade.id));
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('Failed to cancel trade');
    }
  };

  return (
    <div class="active-trades">
      <div class="trades-header">
        <div>
          <h2>Active Trades</h2>
          <p class="text-secondary text-sm">
            {trades().length} of 8 slots used
          </p>
        </div>
        <div class="trades-capital">
          <span class="trades-capital-label">Tied Capital</span>
          <span class="trades-capital-value font-mono">{formatGold(tiedCapital())}</span>
        </div>
      </div>

      <Show when={loading()}>
        <div class="trades-loading">
          <div class="spinner"></div>
        </div>
      </Show>

      <Show when={error()}>
        <div class="trades-error">
          <p class="text-danger">{error()}</p>
          <button class="btn btn-secondary btn-sm mt-2" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      </Show>

      <Show when={!loading() && trades().length === 0}>
        <div class="empty-state card">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="3" y="3" width="18" height="18" rx="2"/>
            <path d="M3 9h18M9 21V9"/>
          </svg>
          <p>No active trades</p>
          <p class="text-sm text-muted">Track trades from the Flips page</p>
        </div>
      </Show>

      <Show when={!loading() && trades().length > 0}>
        <div class="trades-list">
          <For each={trades()}>
            {(trade) => (
              <div class="trade-card card">
                <div class="trade-header">
                  <h3 class="trade-item-name">{trade.item_name}</h3>
                  <span class="trade-time text-muted text-sm">
                    {new Date(trade.created_at).toLocaleDateString()}
                  </span>
                </div>

                <div class="trade-prices">
                  <div class="trade-price">
                    <span class="trade-price-label">Buy</span>
                    <span class="trade-price-value font-mono">
                      {formatGold(trade.buy_price, 'gp')}
                    </span>
                  </div>
                  <div class="trade-price-arrow">→</div>
                  <div class="trade-price">
                    <span class="trade-price-label">Sell Target</span>
                    <span class="trade-price-value font-mono">
                      {formatGold(trade.sell_price, 'gp')}
                    </span>
                  </div>
                </div>

                <div class="trade-info">
                  <div class="trade-info-item">
                    <span class="trade-info-label">Quantity</span>
                    <span class="trade-info-value font-mono">{trade.quantity.toLocaleString()}</span>
                  </div>
                  <div class="trade-info-item">
                    <span class="trade-info-label">Capital</span>
                    <span class="trade-info-value font-mono">
                      {formatGold(trade.buy_price * trade.quantity)}
                    </span>
                  </div>
                  <div class="trade-info-item">
                    <span class="trade-info-label">Target Profit</span>
                    <span class="trade-info-value font-mono text-success">
                      +{formatGold(computeProfit(trade, trade.sell_price, trade.quantity))}
                    </span>
                  </div>
                </div>

                <div class="trade-actions">
                  <button
                    class="btn btn-ghost btn-sm"
                    onClick={() => cancelTrade(trade)}
                  >
                    Cancel
                  </button>
                  <button
                    class="btn btn-success btn-sm"
                    onClick={() => openCompleteModal(trade)}
                  >
                    Complete
                  </button>
                </div>
              </div>
            )}
          </For>
        </div>
      </Show>

      {/* Complete Trade Modal */}
      <Show when={showCompleteModal() && selectedTrade()}>
        <div class="modal-overlay" onClick={() => setShowCompleteModal(false)}>
          <div class="modal" onClick={(e) => e.stopPropagation()}>
            <div class="modal-header">
              <h3>Complete Trade</h3>
              <button
                class="btn btn-ghost btn-icon btn-sm"
                onClick={() => setShowCompleteModal(false)}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
                  <path d="M18 6L6 18M6 6l12 12"/>
                </svg>
              </button>
            </div>
            <div class="modal-body">
              <p class="mb-4">
                <strong>{selectedTrade()!.item_name}</strong>
              </p>

              <div class="form-group">
                <label class="label">Actual Sell Price</label>
                <input
                  type="number"
                  class="input"
                  value={completionData().sellPrice}
                  onInput={(e) => {
                    const nextSellPrice = parseInt(e.currentTarget.value) || 0;
                    const trade = selectedTrade();
                    if (!trade) return;
                    setCompletionData((prev) => ({
                      ...prev,
                      sellPrice: nextSellPrice,
                      profit: computeProfit(trade, nextSellPrice, prev.quantity),
                    }));
                  }}
                />
              </div>

              <div class="form-group">
                <label class="label">Quantity Sold</label>
                <input
                  type="number"
                  class="input"
                  value={completionData().quantity}
                  onInput={(e) => {
                    const nextQty = parseInt(e.currentTarget.value) || 0;
                    const trade = selectedTrade();
                    if (!trade) return;
                    setCompletionData((prev) => ({
                      ...prev,
                      quantity: nextQty,
                      profit: computeProfit(trade, prev.sellPrice, nextQty),
                    }));
                  }}
                />
              </div>

              <div class="form-group">
                <label class="label">Calculated Profit</label>
                <div class={`profit-display font-mono ${completionData().profit >= 0 ? 'text-success' : 'text-danger'}`}>
                  {completionData().profit >= 0 ? '+' : ''}{formatGold(completionData().profit)}
                </div>
              </div>

              <div class="form-group">
                <label class="label">Notes (optional)</label>
                <textarea
                  class="input"
                  rows="2"
                  value={completionData().notes}
                  onInput={(e) => setCompletionData({
                    ...completionData(),
                    notes: e.currentTarget.value
                  })}
                  placeholder="Any notes about this trade..."
                />
              </div>
            </div>
            <div class="modal-footer">
              <button
                class="btn btn-secondary"
                onClick={() => setShowCompleteModal(false)}
              >
                Cancel
              </button>
              <button
                class="btn btn-success"
                onClick={completeTrade}
                disabled={processing()}
              >
                {processing() ? 'Completing...' : 'Complete Trade'}
              </button>
            </div>
          </div>
        </div>
      </Show>

      <style>{`
        .active-trades {
          max-width: 800px;
          margin: 0 auto;
        }

        .trades-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          margin-bottom: var(--space-4);
        }

        .trades-capital {
          text-align: right;
        }

        .trades-capital-label {
          display: block;
          font-size: var(--font-size-xs);
          color: var(--text-muted);
        }

        .trades-capital-value {
          font-size: var(--font-size-lg);
          font-weight: 600;
        }

        .trades-loading {
          display: flex;
          justify-content: center;
          padding: var(--space-6);
        }

        .trades-error {
          text-align: center;
          padding: var(--space-4);
        }

        .trades-list {
          display: grid;
          gap: var(--space-4);
        }

        .trade-card {
          transition: transform var(--transition-fast);
        }

        .trade-card:hover {
          transform: translateY(-2px);
        }

        .trade-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: var(--space-3);
        }

        .trade-item-name {
          font-size: var(--font-size-lg);
        }

        .trade-prices {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: var(--space-4);
          padding: var(--space-3);
          background-color: var(--bg-tertiary);
          border-radius: var(--radius-md);
          margin-bottom: var(--space-3);
        }

        .trade-price {
          text-align: center;
        }

        .trade-price-label {
          display: block;
          font-size: var(--font-size-xs);
          color: var(--text-muted);
        }

        .trade-price-value {
          font-size: var(--font-size-base);
          font-weight: 600;
        }

        .trade-price-arrow {
          color: var(--text-muted);
        }

        .trade-info {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: var(--space-3);
          margin-bottom: var(--space-3);
        }

        .trade-info-item {
          text-align: center;
        }

        .trade-info-label {
          display: block;
          font-size: var(--font-size-xs);
          color: var(--text-muted);
        }

        .trade-info-value {
          font-weight: 500;
        }

        .trade-actions {
          display: flex;
          justify-content: flex-end;
          gap: var(--space-2);
          padding-top: var(--space-3);
          border-top: 1px solid var(--border);
        }

        .profit-display {
          font-size: var(--font-size-xl);
          font-weight: 700;
          padding: var(--space-3);
          background-color: var(--bg-tertiary);
          border-radius: var(--radius-md);
          text-align: center;
        }

        @media (max-width: 640px) {
          .trade-info {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
