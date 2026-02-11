// packages/web/src/components/trades/TradeList.tsx
import { createEffect, createSignal, For, Show, onCleanup, untrack } from 'solid-js';
import type { ActiveTrade } from '../../lib/db';
import { toTradeViewModel, type TradeViewModel } from '../../lib/trade-types';
import type { UpdateRecommendation } from '../../lib/types';
import { TradeCard } from './TradeCard';
import { TradeDetail } from './TradeDetail';
import { addToast, removeToast } from '../ToastContainer';

type CancelReason = 'changed_mind' | 'did_not_fill';

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
  const [cancelModal, setCancelModal] = createSignal<{
    tradeId: string;
    trade: TradeViewModel;
  } | null>(null);
  const [cancelReason, setCancelReason] = createSignal<CancelReason>('changed_mind');

  // Keep internal view-model list in sync with parent refreshes.
  // TradeList owns some local UI state (expanded/cancel/undo), but the trade
  // source-of-truth lives in the parent, which refetches after actions like
  // "Revise price". Without this, the UI can look like the trade never updated.
  createEffect(() => {
    const next = props.initialTrades.map(toTradeViewModel);
    setTrades(next);

    const expanded = untrack(() => expandedId());
    if (expanded && !next.some(t => t.id === expanded)) {
      setExpandedId(null);
    }
  });

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

  // Handle switching from buying -> selling, optionally with partial quantity.
  const handleAdvanceToSelling = async (tradeId: string, filledQuantity?: number) => {
    try {
      if (filledQuantity !== undefined) {
        const qty = Math.max(1, Math.floor(filledQuantity));
        const patchRes = await fetch(`/api/trades/active/${tradeId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ quantity: qty })
        });
        if (!patchRes.ok) throw new Error('Failed to save filled amount');
      }

      const res = await fetch(`/api/trades/active/${tradeId}/advance`, {
        method: 'POST'
      });
      if (!res.ok) throw new Error('Failed to switch to selling');
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
      console.error('Advance to selling failed:', err);
      setError('Failed to update order fill. Please try again.');
    }
  };

  // Handle completing the selling phase with chosen sale price.
  const handleCompleteSale = async (tradeId: string, sellPrice: number) => {
    try {
      const res = await fetch(`/api/trades/active/${tradeId}/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sellPrice })
      });
      if (!res.ok) throw new Error('Failed to complete trade');
      const data = await res.json();

      if (data.success) {
        setTrades(prev => prev.filter(t => t.id !== tradeId));
        setExpandedId(null);
        setError(null);
      }
    } catch (err) {
      console.error('Complete sale failed:', err);
      setError('Failed to complete sale. Please try again.');
    }
  };

  // Track pending cancellations: tradeId → { timer, toastId, trade, index, reason }
  const pendingCancels = new Map<string, { timer: ReturnType<typeof setTimeout>; toastId: string; trade: TradeViewModel; index: number; reason: CancelReason }>();

  const UNDO_WINDOW_MS = 6000;

  // Commit a pending cancel to the backend
  const commitCancel = async (tradeId: string, reason: CancelReason) => {
    pendingCancels.delete(tradeId);
    try {
      const res = await fetch(`/api/trades/active/${tradeId}?reason=${encodeURIComponent(reason)}`, {
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

  // Start cancel — optimistic removal + undo toast
  const startCancel = (tradeId: string, reason: CancelReason) => {
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
        commitCancel(tradeId, pending.reason);
      }
    }, UNDO_WINDOW_MS);

    // Show undo toast
    const toastId = addToast({
      type: 'info',
      title: 'Trade cancelled',
      message: `${trade.itemName}${reason === 'did_not_fill' ? ' (did not fill)' : ''}`,
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

    pendingCancels.set(tradeId, { timer, toastId, trade, index, reason });
  };

  const promptCancel = (tradeId: string) => {
    const currentTrades = trades();
    const index = currentTrades.findIndex(t => t.id === tradeId);
    if (index === -1) return;
    const trade = currentTrades[index];
    setCancelReason('changed_mind');
    setCancelModal({ tradeId, trade });
  };

  const startCollapse = () => {
    setExpandedId(null);
  };

  // Cleanup pending timers on unmount
  onCleanup(() => {
    for (const [tradeId, { timer, reason }] of pendingCancels) {
      clearTimeout(timer);
      commitCancel(tradeId, reason);
    }
  });

  return (
    <div class="trade-list">
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
            {(trade) => {
              const isAlert = Boolean(props.alerts?.get(trade.id) || trade.suggestedSellPrice);
              return (
              <div class={`trade-item ${expandedId() === trade.id ? 'is-expanded' : ''} ${isAlert ? 'trade-item-alert' : ''}`}>
                <div class="trade-item-card">
                  <TradeCard
                    trade={trade}
                    expanded={expandedId() === trade.id}
                    onClick={() => {
                      if (expandedId() === trade.id) {
                        startCollapse();
                        return;
                      }
                      setExpandedId(trade.id);
	                    }}
	                    onCancel={() => promptCancel(trade.id)}
	                    alert={props.alerts?.get(trade.id)}
	                  />
                </div>
                <div class={`trade-detail-wrap ${expandedId() === trade.id ? 'is-expanded' : ''}`}>
                  <Show when={expandedId() === trade.id}>
                    <TradeDetail
	                      trade={trade}
	                      onAdvanceToSelling={(filledQuantity) => handleAdvanceToSelling(trade.id, filledQuantity)}
	                      onCompleteSale={(sellPrice) => handleCompleteSale(trade.id, sellPrice)}
	                      onCancel={() => promptCancel(trade.id)}
	                      onClose={() => startCollapse()}
	                      showHeader={false}
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
                </div>
              </div>
            )}}
          </For>
        </div>
      </Show>

      <Show when={cancelModal()}>
        <div
          class="trade-cancel-backdrop"
          role="presentation"
          onClick={() => setCancelModal(null)}
        >
          <div
            class="trade-cancel-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Cancel trade"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 class="trade-cancel-title">Cancel trade?</h3>
            <p class="trade-cancel-subtitle">
              Why are you cancelling <span class="trade-cancel-item">{cancelModal()!.trade.itemName}</span>?
            </p>

            <div class="trade-cancel-options">
              <label class={`trade-cancel-option ${cancelReason() === 'did_not_fill' ? 'is-selected' : ''}`}>
                <input
                  type="radio"
                  name="cancel-reason"
                  value="did_not_fill"
                  checked={cancelReason() === 'did_not_fill'}
                  onChange={() => setCancelReason('did_not_fill')}
                />
                <span class="trade-cancel-option-text">
                  <span class="trade-cancel-option-title">Did not fill</span>
                  <span class="trade-cancel-option-help">Helps us tune prices to improve fills.</span>
                </span>
              </label>

              <label class={`trade-cancel-option ${cancelReason() === 'changed_mind' ? 'is-selected' : ''}`}>
                <input
                  type="radio"
                  name="cancel-reason"
                  value="changed_mind"
                  checked={cancelReason() === 'changed_mind'}
                  onChange={() => setCancelReason('changed_mind')}
                />
                <span class="trade-cancel-option-text">
                  <span class="trade-cancel-option-title">Changed my mind</span>
                  <span class="trade-cancel-option-help">We will not treat this as a pricing issue.</span>
                </span>
              </label>
            </div>

            <div class="trade-cancel-actions">
              <button
                class="trade-cancel-btn trade-cancel-btn-secondary"
                onClick={() => setCancelModal(null)}
              >
                Keep trade
              </button>
              <button
                class="trade-cancel-btn trade-cancel-btn-danger"
                onClick={() => {
                  const modal = cancelModal();
                  if (!modal) return;
                  const reason = cancelReason();
                  setCancelModal(null);
                  startCancel(modal.tradeId, reason);
                }}
              >
                Cancel trade
              </button>
            </div>
          </div>
        </div>
      </Show>

      <style>{`
        .trade-list {
          max-width: 720px;
          margin: 0 auto;
          padding: 1rem;
          padding-bottom: 5rem;
        }

        .trade-list-empty {
          text-align: center;
          padding: 3rem 1rem;
          color: var(--text-secondary);
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
          background: var(--action);
          color: var(--btn-text-dark);
          border: none;
          border-radius: var(--radius-full);
          font-weight: 600;
          cursor: pointer;
          box-shadow: 0 18px 36px -26px rgba(168, 240, 8, 0.6);
        }

        .trade-list-cta:hover {
          background: var(--action-hover);
        }

        .trade-list-items {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }

        .trade-item {
          display: flex;
          flex-direction: column;
          gap: 0;
          position: relative;
        }

        .trade-item.is-expanded.trade-item-alert::before {
          content: '';
          position: absolute;
          inset: 0;
          border-radius: var(--radius-xl);
          border: 1px solid color-mix(in srgb, var(--warning) 60%, transparent);
          pointer-events: none;
          z-index: 3;
        }

        .trade-item-card {
          z-index: 2;
        }

        .trade-detail-wrap {
          margin-top: -1px;
        }

        .trade-detail-wrap.is-expanded .trade-detail {
          border-color: var(--border-light);
        }

        .trade-item.is-expanded .trade-detail {
          border-top-left-radius: 0;
          border-top-right-radius: 0;
          border-top: none;
          margin-top: -1px;
        }

        .trade-list-error {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.75rem 1rem;
          background: var(--surface-2);
          color: var(--danger);
          border-radius: var(--radius-lg);
          border: 1px solid color-mix(in srgb, var(--danger) 35%, transparent);
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

        .trade-cancel-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.55);
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 1.25rem;
          z-index: 50;
        }

        .trade-cancel-modal {
          width: min(520px, 100%);
          background: var(--surface-1);
          border: 1px solid var(--border);
          border-radius: var(--radius-xl);
          box-shadow: 0 26px 70px -40px rgba(0, 0, 0, 0.75);
          padding: 1.2rem 1.2rem 1rem;
        }

        .trade-cancel-title {
          margin: 0 0 0.25rem;
          font-size: var(--font-size-lg);
          font-weight: 700;
          color: var(--text-primary);
          letter-spacing: -0.01em;
        }

        .trade-cancel-subtitle {
          margin: 0 0 1rem;
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
          line-height: 1.35;
        }

        .trade-cancel-item {
          color: var(--text-primary);
          font-weight: 600;
        }

        .trade-cancel-options {
          display: grid;
          gap: 0.65rem;
          margin-bottom: 1rem;
        }

        .trade-cancel-option {
          display: grid;
          grid-template-columns: 18px 1fr;
          gap: 0.75rem;
          align-items: start;
          padding: 0.85rem 0.9rem;
          border-radius: var(--radius-lg);
          border: 1px solid var(--border);
          background: color-mix(in srgb, var(--surface-2) 55%, transparent);
          cursor: pointer;
          transition: border-color 0.2s ease, background 0.2s ease, transform 0.2s ease;
        }

        .trade-cancel-option:hover {
          transform: translateY(-1px);
          border-color: var(--border-light);
        }

        .trade-cancel-option.is-selected {
          border-color: color-mix(in srgb, var(--action) 60%, var(--border));
          background: color-mix(in srgb, var(--action) 10%, var(--surface-2));
        }

        .trade-cancel-option input {
          margin-top: 2px;
          accent-color: var(--action);
        }

        .trade-cancel-option-text {
          display: grid;
          gap: 0.15rem;
        }

        .trade-cancel-option-title {
          font-weight: 700;
          color: var(--text-primary);
          font-size: var(--font-size-sm);
        }

        .trade-cancel-option-help {
          font-size: 0.85rem;
          color: var(--text-secondary);
        }

        .trade-cancel-actions {
          display: flex;
          gap: 0.6rem;
          justify-content: flex-end;
        }

        .trade-cancel-btn {
          border-radius: var(--radius-full);
          padding: 0.7rem 1rem;
          font-weight: 650;
          border: 1px solid transparent;
          cursor: pointer;
          transition: transform 0.2s ease, background 0.2s ease, border-color 0.2s ease;
        }

        .trade-cancel-btn:active {
          transform: translateY(1px);
        }

        .trade-cancel-btn-secondary {
          background: transparent;
          border-color: var(--border);
          color: var(--text-primary);
        }

        .trade-cancel-btn-secondary:hover {
          border-color: var(--border-light);
          background: color-mix(in srgb, var(--surface-2) 80%, transparent);
        }

        .trade-cancel-btn-danger {
          background: color-mix(in srgb, var(--danger) 88%, black);
          color: white;
        }

        .trade-cancel-btn-danger:hover {
          background: color-mix(in srgb, var(--danger) 95%, black);
        }
      `}</style>
    </div>
  );
}
