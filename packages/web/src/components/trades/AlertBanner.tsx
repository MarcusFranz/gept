// packages/web/src/components/trades/AlertBanner.tsx
import { Show } from 'solid-js';
import type { UpdateRecommendation } from '../../lib/types';

interface AlertBannerProps {
  alert: UpdateRecommendation;
  onAccept: () => void;
  onDismiss: () => void;
  loading?: boolean;
  currentSellPrice?: number;
}

export function AlertBanner(props: AlertBannerProps) {
  const isDanger = () => props.alert.type === 'SELL_NOW';

  const formatPrice = (price: number) => {
    if (price >= 1_000_000) {
      return (price / 1_000_000).toFixed(2) + 'M';
    } else if (price >= 1_000) {
      return (price / 1_000).toFixed(1) + 'K';
    }
    return price.toLocaleString();
  };

  const getHeader = () => {
    if (props.alert.type === 'SELL_NOW') return 'Sell price at risk';
    if (props.alert.type === 'ADJUST_PRICE') return 'Consider repricing';
    return 'Trade alert';
  };

  const getIcon = () => {
    if (props.alert.type === 'SELL_NOW') return '\u26A0\uFE0F';
    return '\u26A0\uFE0F';
  };

  const getBody = () => {
    if (props.alert.type === 'ADJUST_PRICE' && props.alert.newSellPrice) {
      return props.alert.reason;
    }
    if (props.alert.type === 'SELL_NOW' && props.alert.adjustedSellPrice) {
      return `Prices have dropped below your buy price. Consider selling now at ${formatPrice(props.alert.adjustedSellPrice)}gp to limit loss.`;
    }
    return props.alert.reason;
  };

  const getAcceptLabel = () => {
    if (props.alert.type === 'ADJUST_PRICE') return 'Revise price';
    if (props.alert.type === 'SELL_NOW') return 'Sell now';
    return 'Accept';
  };

  const getDismissLabel = () => {
    if (props.alert.type === 'SELL_NOW') return 'Keep waiting';
    return 'Dismiss';
  };

  const getPriceDeltaDirection = (from: number, to: number) => {
    if (!Number.isFinite(from) || !Number.isFinite(to)) return 'neutral' as const;
    if (to > from) return 'up' as const;
    if (to < from) return 'down' as const;
    return 'neutral' as const;
  };

  const getSuggestedSell = () => {
    if (props.alert.type === 'ADJUST_PRICE') return props.alert.newSellPrice;
    if (props.alert.type === 'SELL_NOW') return props.alert.adjustedSellPrice;
    return undefined;
  };

  const getOriginalSell = () => {
    if (props.alert.type === 'ADJUST_PRICE') return props.alert.originalSellPrice ?? props.currentSellPrice;
    if (props.alert.type === 'SELL_NOW') return props.currentSellPrice;
    return undefined;
  };

  return (
    <div class={`alert-banner ${isDanger() ? 'alert-banner-danger' : ''}`}>
      <div class="alert-banner-header">
        <span class="alert-banner-icon">{getIcon()}</span>
        <span class={`alert-banner-title ${isDanger() ? 'alert-banner-title-danger' : ''}`}>
          {getHeader()}
        </span>
      </div>

      <p class="alert-banner-body">{getBody()}</p>

      <Show when={getSuggestedSell() !== undefined && getOriginalSell() !== undefined}>
        {() => {
          const original = getOriginalSell()!;
          const suggested = getSuggestedSell()!;
          const dir = getPriceDeltaDirection(original, suggested);
          return (
            <div class="alert-banner-price">
              <div class="alert-banner-price-stack">
                <span class="alert-banner-price-original">{formatPrice(original)}gp</span>
                <span class={`alert-banner-price-suggested ${dir === 'up' ? 'is-up' : dir === 'down' ? 'is-down' : ''}`}>
                  {formatPrice(suggested)}gp
                  <button
                    class="alert-banner-price-accept"
                    onClick={() => props.onAccept()}
                    disabled={props.loading}
                    aria-label="Apply suggested price"
                    title="Apply suggested price"
                  >
                    ✓
                  </button>
                </span>
              </div>
            </div>
          );
        }}
      </Show>

      <Show when={props.alert.type === 'ADJUST_PRICE' && props.alert.profitDelta !== undefined}>
        <p class="alert-banner-impact">
          Profit impact: <strong class="alert-banner-delta">{formatPrice(props.alert.profitDelta!)}</strong>
        </p>
      </Show>

      <div class="alert-banner-actions">
        <button
          class={`alert-banner-btn alert-banner-btn-primary ${isDanger() ? 'alert-banner-btn-danger' : ''}`}
          onClick={() => props.onAccept()}
          disabled={props.loading}
        >
          {props.loading ? 'Processing...' : getAcceptLabel()}
        </button>
        <button
          class="alert-banner-btn alert-banner-btn-secondary"
          onClick={() => props.onDismiss()}
          disabled={props.loading}
        >
          {getDismissLabel()}
        </button>
      </div>

      <style>{`
        .alert-banner {
          background: var(--surface-2);
          border: 1px solid color-mix(in srgb, var(--warning) 40%, transparent);
          border-radius: var(--radius-lg);
          padding: 1rem;
          margin: 1rem 0;
          box-shadow: var(--shadow-sm);
        }

        .alert-banner-danger {
          border-color: color-mix(in srgb, var(--danger) 40%, transparent);
        }

        .alert-banner-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.5rem;
        }

        .alert-banner-icon {
          font-size: 1.25rem;
        }

        .alert-banner-title {
          font-weight: 600;
          color: var(--warning);
        }

        .alert-banner-title-danger {
          color: var(--danger);
        }

        .alert-banner-body {
          margin: 0.5rem 0;
          color: var(--text-primary);
          font-size: var(--font-size-sm);
        }

        .alert-banner-price {
          margin-top: 0.5rem;
          padding: 0.75rem 0.85rem;
          border-radius: var(--radius-lg);
          background: color-mix(in srgb, var(--surface-1) 80%, transparent);
          border: 1px solid var(--border);
        }

        .alert-banner-price-stack {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          gap: 0.2rem;
          font-family: var(--font-mono);
        }

        .alert-banner-price-original {
          text-decoration: line-through;
          color: var(--text-muted);
          opacity: 0.75;
          font-size: var(--font-size-xs);
        }

        .alert-banner-price-suggested {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          font-weight: 700;
          color: var(--text-primary);
        }

        .alert-banner-price-suggested.is-up {
          color: var(--success);
        }

        .alert-banner-price-suggested.is-down {
          color: var(--danger);
        }

        .alert-banner-price-accept {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 22px;
          height: 22px;
          padding: 0;
          background: var(--success-light);
          color: var(--success);
          border: 1px solid color-mix(in srgb, var(--success) 70%, transparent);
          border-radius: var(--radius-full);
          font-size: 0.75rem;
          font-weight: 800;
          cursor: pointer;
          transition: background var(--transition-fast), color var(--transition-fast);
        }

        .alert-banner-price-accept:hover:not(:disabled) {
          background: var(--success);
          color: var(--bg-primary);
        }

        .alert-banner-price-accept:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .alert-banner-impact {
          margin: 0.25rem 0 0.5rem;
          color: var(--text-secondary);
          font-size: var(--font-size-sm);
        }

        .alert-banner-delta {
          color: var(--danger);
        }

        .alert-banner-actions {
          display: flex;
          gap: 0.75rem;
          margin-top: 1rem;
        }

        .alert-banner-btn {
          flex: 1;
          padding: 0.65rem 1rem;
          border: 1px solid transparent;
          border-radius: var(--radius-full);
          font-weight: 600;
          cursor: pointer;
          transition: background var(--transition-fast);
        }

        .alert-banner-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .alert-banner-btn-primary {
          background: var(--warning);
          color: var(--btn-text-dark);
          border-color: color-mix(in srgb, var(--warning) 65%, transparent);
        }

        .alert-banner-btn-primary:hover:not(:disabled) {
          filter: brightness(0.9);
        }

        .alert-banner-btn-danger {
          background: var(--danger);
          color: var(--btn-text-dark);
          border-color: color-mix(in srgb, var(--danger) 65%, transparent);
        }

        .alert-banner-btn-danger:hover:not(:disabled) {
          filter: brightness(0.9);
        }

        .alert-banner-btn-secondary {
          background: var(--surface-1);
          color: var(--text-primary);
          border: 1px solid var(--border);
        }

        .alert-banner-btn-secondary:hover:not(:disabled) {
          background: var(--surface-2);
        }
      `}</style>
    </div>
  );
}
