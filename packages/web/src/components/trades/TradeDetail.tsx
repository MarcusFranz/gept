// packages/web/src/components/trades/TradeDetail.tsx
import { createSignal, onCleanup, onMount, Show } from 'solid-js';
import type { TradeViewModel, Guidance } from '../../lib/trade-types';
import type { UpdateRecommendation } from '../../lib/types';
import { CheckInBar } from './CheckInBar';
import { GuidancePrompt } from './GuidancePrompt';
import { Sparkline } from '../Sparkline';
import { fetchPriceHistoryWithFallback, type PriceHistoryData } from '../../lib/price-history';

interface TradeDetailProps {
  trade: TradeViewModel;
  onCheckIn: (progress: number) => Promise<{ guidance?: Guidance }>;
  onAdvance: () => Promise<void>;
  onCancel: () => void;
  onClose: () => void;
  alert?: UpdateRecommendation;
  onAcceptAlert?: () => void;
  onDismissAlert?: () => void;
  onAcknowledgePrice?: () => void;
  showHeader?: boolean;
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export function TradeDetail(props: TradeDetailProps) {
  const [priceHistory, setPriceHistory] = createSignal<PriceHistoryData | null>(null);
  const [historyLoading, setHistoryLoading] = createSignal(true);

  onMount(() => {
    let disposed = false;
    onCleanup(() => {
      disposed = true;
    });

    setHistoryLoading(true);
    (async () => {
      try {
        let data = await fetchPriceHistoryWithFallback(props.trade.itemId);

        // Retry once after 2s if the first attempt fails (transient engine errors)
        if (!data && !disposed) {
          await sleep(2000);
          data = await fetchPriceHistoryWithFallback(props.trade.itemId);
        }

        // Dev-only: force the Death rune sparkline to "feel slow" so the
        // placeholder animation is easy to see in action.
        if (import.meta.env.DEV && props.trade.itemId === 560) {
          await sleep(5000);
        }

        if (!disposed && data) setPriceHistory(data);
      } finally {
        if (!disposed) setHistoryLoading(false);
      }
    })();
  });
  const [loading, setLoading] = createSignal(false);
  const [guidance, setGuidance] = createSignal<Guidance | undefined>(undefined);
  const [pendingProgress, setPendingProgress] = createSignal<number | undefined>(undefined);
  const [closing, setClosing] = createSignal(false);

  const formatGold = (amount: number) => {
    if (amount >= 1_000_000) {
      return (amount / 1_000_000).toFixed(2) + 'M';
    } else if (amount >= 1_000) {
      return (amount / 1_000).toFixed(1) + 'K';
    }
    return amount.toLocaleString() + ' gp';
  };

  // Exact GP price for GE offers - no abbreviation
  const formatExactGold = (amount: number) => {
    return amount.toLocaleString() + ' gp';
  };

  const handleProgressChange = async (progress: number) => {
    setPendingProgress(progress);
  };

  const handleDone = async () => {
    setLoading(true);
    try {
      // If there's pending progress, submit it first
      if (pendingProgress() !== undefined) {
        await props.onCheckIn(pendingProgress()!);
      }
      // Advance to next phase
      await props.onAdvance();
    } finally {
      setLoading(false);
    }
  };

  const handleCheckIn = async () => {
    if (pendingProgress() === undefined) return;

    setLoading(true);
    try {
      const result = await props.onCheckIn(pendingProgress()!);
      if (result.guidance) {
        setGuidance(result.guidance);
      }
      setPendingProgress(undefined);
    } finally {
      setLoading(false);
    }
  };

  const handleGuidanceAccept = async () => {
    setLoading(true);
    try {
      // Guidance actions (relist/exit/sell_now) will call additional API endpoints
      // when implemented. For now, accepting guidance just dismisses the prompt.
      // The parent component will handle the actual action in a future phase.
      setGuidance(undefined);
    } finally {
      setLoading(false);
    }
  };

  // Use getter functions for reactive prop access
  const actualBuy = () => props.trade.actualBuyPrice || props.trade.buyPrice;
  const actualSell = () => props.trade.actualSellPrice || props.trade.sellPrice;

  // Price alerts can come from two sources:
  // - Real-time alert payloads (SSE / polling), which include a suggested price.
  // - Persisted suggested_sell_price on the trade (durable across refreshes).
  //
  // We intentionally render a single integrated UI (strike-through + new price)
  // instead of a separate notification-style banner.
  const alertSuggestedSell = () => {
    const a = props.alert;
    if (!a) return null;
    const v =
      a.type === 'ADJUST_PRICE'
        ? a.newSellPrice
        : a.type === 'SELL_NOW'
          ? a.adjustedSellPrice
          : undefined;
    return typeof v === 'number' && Number.isFinite(v) ? v : null;
  };

  const suggestedSell = () => alertSuggestedSell() ?? props.trade.suggestedSellPrice;
  const hasSuggestedSell = () => suggestedSell() != null;

  const applySuggestedSell = () => {
    // Prefer the alert-backed accept flow when it has an explicit suggested price,
    // otherwise fall back to the persisted suggestion flow.
    if (alertSuggestedSell() != null) {
      props.onAcceptAlert?.();
      return;
    }
    props.onAcknowledgePrice?.();
  };

  const dismissSuggestedSell = () => {
    props.onDismissAlert?.();
  };

  const suggestedDir = () => {
    const suggested = suggestedSell();
    if (suggested == null) return 'neutral' as const;
    const original = actualSell();
    if (suggested > original) return 'up' as const;
    if (suggested < original) return 'down' as const;
    return 'neutral' as const;
  };

  const handleClose = () => {
    if (closing()) return;
    setClosing(true);
    setTimeout(() => {
      props.onClose();
    }, 320);
  };

  return (
    <div class={`trade-detail ${closing() ? 'is-closing' : 'is-open'}`}>
      <Show when={props.showHeader !== false}>
        <div class="trade-detail-header">
          <h3 class="trade-detail-title">{props.trade.itemName}</h3>
          <div class={`trade-detail-actions ${closing() ? 'is-closing' : 'is-open'}`}>
            <span class={`trade-detail-phase phase-${props.trade.phase}`}>
              {props.trade.phase.toUpperCase()}
            </span>
            <button class="trade-detail-close" onClick={handleClose} aria-label="Close trade detail">×</button>
          </div>
        </div>
      </Show>

      <div class="trade-detail-prices-exact">
        <div class="price-box buy-price">
          <span class="price-label">Buy at</span>
          <span class="price-value">{formatExactGold(actualBuy())}</span>
        </div>
        <div class="price-arrow">→</div>
        <div class="price-box sell-price">
          <span class="price-label">Sell at</span>
          <Show
            when={hasSuggestedSell()}
            fallback={<span class="price-value">{formatExactGold(actualSell())}</span>}
          >
            <span class="price-value price-strikethrough">{formatExactGold(actualSell())}</span>
            <span class={`price-value price-suggested ${suggestedDir() === 'up' ? 'price-suggested-up' : suggestedDir() === 'down' ? 'price-suggested-down' : ''}`}>
              {formatExactGold(suggestedSell()!)}
            </span>
            <button
              class="price-acknowledge"
              onClick={(e) => {
                e.stopPropagation();
                applySuggestedSell();
              }}
              aria-label="Apply suggested price"
              title="Apply suggested price"
            >
              ✓
            </button>
          </Show>
        </div>
      </div>

      <div class="trade-detail-sparkline">
        <Sparkline
          highs={priceHistory()?.highs ?? []}
          lows={priceHistory()?.lows ?? []}
          loading={historyLoading() && !priceHistory()}
          width={280}
          height={56}
        />
      </div>

      <Show when={props.alert?.reason && hasSuggestedSell()}>
        <div class="trade-detail-alert-inline">
          <p class="trade-detail-alert-reason">{props.alert!.reason}</p>
          <button
            class="trade-detail-alert-dismiss"
            onClick={(e) => {
              e.stopPropagation();
              dismissSuggestedSell();
            }}
            disabled={loading()}
          >
            Dismiss
          </button>
        </div>
      </Show>

      {guidance() ? (
        <GuidancePrompt
          guidance={guidance()!}
          onAccept={handleGuidanceAccept}
          onDismiss={() => setGuidance(undefined)}
          loading={loading()}
        />
      ) : (
        <CheckInBar
          progress={pendingProgress() ?? props.trade.progress}
          phase={props.trade.phase}
          onProgressChange={handleProgressChange}
          onDone={handleDone}
          disabled={loading()}
        />
      )}

      {pendingProgress() !== undefined && pendingProgress() !== props.trade.progress && !guidance() && (
        <button
          class="trade-detail-submit"
          onClick={handleCheckIn}
          disabled={loading()}
        >
          {loading() ? 'Saving...' : 'Save progress'}
        </button>
      )}

      <hr class="trade-detail-divider" />

      <button
        class="trade-detail-cancel"
        onClick={() => props.onCancel()}
        disabled={loading()}
      >
        Cancel trade
      </button>

      <style>{`
        .trade-detail {
          background: var(--surface-2);
          border: 1px solid var(--border);
          border-radius: var(--radius-xl);
          padding: 1.1rem;
          position: relative;
          box-shadow: none;
        }

        .trade-detail-alert-inline {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.75rem;
          margin-top: 0.85rem;
          padding: 0.75rem 0.9rem;
          border-radius: var(--radius-lg);
          background: color-mix(in srgb, var(--surface-1) 88%, var(--warning) 12%);
          border: 1px solid color-mix(in srgb, var(--warning) 35%, transparent);
        }

        .trade-detail-alert-reason {
          margin: 0;
          font-size: 0.92rem;
          line-height: 1.35;
          color: var(--text);
          opacity: 0.9;
        }

        .trade-detail-alert-dismiss {
          background: transparent;
          border: 1px solid color-mix(in srgb, var(--border) 70%, transparent);
          color: var(--text);
          opacity: 0.85;
          padding: 0.4rem 0.6rem;
          border-radius: 999px;
          font-weight: 600;
          cursor: pointer;
          transition: opacity 0.2s ease, border-color 0.2s ease;
          white-space: nowrap;
        }

        .trade-detail-alert-dismiss:hover:not(:disabled) {
          opacity: 1;
          border-color: color-mix(in srgb, var(--border-light) 70%, transparent);
        }

        .trade-detail-alert-dismiss:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .trade-detail.is-open {
          animation: none;
        }

        .trade-detail.is-closing {
          animation: none;
        }

        .trade-detail-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
          gap: 0.75rem;
        }

        .trade-detail-title {
          margin: 0;
          font-size: var(--font-size-lg);
        }

        .trade-detail-actions {
          display: flex;
          align-items: center;
          gap: 0.35rem;
          flex-shrink: 0;
        }

        .trade-detail-phase {
          font-size: var(--font-size-xs);
          font-weight: 700;
          padding: 0.25rem 0.75rem;
          border-radius: var(--radius-full);
          transition: none;
        }

        .phase-buying {
          background: var(--phase-buy-light);
          color: var(--phase-buy);
        }

        .phase-selling {
          background: var(--phase-sell-light);
          color: var(--phase-sell);
        }

        .trade-detail-close {
          background: var(--surface-2);
          border: 1px solid var(--border);
          color: var(--text-muted);
          font-size: 1.05rem;
          cursor: pointer;
          padding: 0.2rem 0.55rem;
          line-height: 1;
          border-radius: var(--radius-full);
          opacity: 1;
          transform: none;
          transition: color var(--transition-fast), border-color var(--transition-fast), background var(--transition-fast);
        }

        .trade-detail-close:hover {
          color: var(--text-primary);
          border-color: var(--border-light);
        }

        .trade-detail-actions.is-open .trade-detail-phase,
        .trade-detail-actions.is-open .trade-detail-close,
        .trade-detail-actions.is-closing .trade-detail-phase,
        .trade-detail-actions.is-closing .trade-detail-close {
          animation: none;
        }

        .trade-detail-prices-exact {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.5rem;
          margin: 0.85rem 0;
          padding: 0.85rem;
          background: var(--surface-2);
          border-radius: var(--radius-lg);
          border: 1px solid var(--border);
        }

        .price-box {
          display: flex;
          flex-direction: column;
          align-items: center;
          flex: 1;
        }

        .price-label {
          font-size: var(--font-size-xs);
          text-transform: uppercase;
          color: var(--text-muted);
          margin-bottom: 0.25rem;
        }

        .price-value {
          font-size: var(--font-size-sm);
          font-weight: 600;
          font-family: var(--font-mono);
        }

        .buy-price .price-value {
          color: var(--gold);
        }

        .sell-price .price-value {
          color: var(--success);
        }

        .price-strikethrough {
          text-decoration: line-through;
          opacity: 0.5;
          color: var(--text-muted) !important;
          font-size: var(--font-size-xs);
        }

        .price-suggested {
          color: var(--text-primary) !important;
        }

        .price-suggested-up {
          color: var(--success) !important;
        }

        .price-suggested-down {
          color: var(--danger) !important;
        }

        .price-acknowledge {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 22px;
          height: 22px;
          padding: 0;
          margin-left: 0.25rem;
          background: var(--success-light);
          color: var(--success);
          border: 1px solid color-mix(in srgb, var(--success) 70%, transparent);
          border-radius: var(--radius-full);
          font-size: 0.75rem;
          font-weight: 700;
          cursor: pointer;
          transition: background var(--transition-fast);
        }

        .price-acknowledge:hover {
          background: var(--success);
          color: var(--bg-primary);
        }

        .price-arrow {
          color: var(--text-muted);
          font-size: 1rem;
        }

        .trade-detail-sparkline {
          margin: 0.85rem 0 1.05rem;
          display: flex;
          justify-content: center;
          padding: 0.75rem 0.75rem;
        }

        .trade-detail-profit {
          margin-top: 0.25rem;
          color: var(--text-secondary);
          font-size: var(--font-size-sm);
        }

        .trade-detail-profit strong {
          color: var(--success);
        }

        .trade-detail-divider {
          border: none;
          border-top: 1px solid var(--border);
          margin: 1rem 0;
        }

        .trade-detail-submit {
          width: 100%;
          padding: 0.7rem;
          background: var(--action);
          color: var(--btn-text-dark);
          border: 1px solid color-mix(in srgb, var(--action) 75%, #000);
          border-radius: var(--radius-full);
          font-weight: 600;
          cursor: pointer;
          margin-top: 0.5rem;
          transition: transform 0.4s var(--ease-hero), box-shadow 0.4s var(--ease-hero), background 0.3s ease;
        }

        .trade-detail-submit:hover:not(:disabled) {
          background: var(--action-hover);
          transform: translateY(-2px) scale(1.01);
          box-shadow: 0 20px 34px -22px rgba(168, 240, 8, 0.55);
        }

        .trade-detail-submit:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .trade-detail-cancel {
          width: 100%;
          padding: 0.6rem;
          background: var(--surface-2);
          color: var(--danger);
          border: 1px solid color-mix(in srgb, var(--danger) 65%, transparent);
          border-radius: var(--radius-full);
          cursor: pointer;
          margin-top: 1rem;
          transition: background var(--transition-fast), color var(--transition-fast);
        }

        .trade-detail-cancel:hover:not(:disabled) {
          background: var(--danger-light);
        }

        .trade-detail-cancel:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}
