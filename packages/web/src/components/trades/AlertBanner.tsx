// packages/web/src/components/trades/AlertBanner.tsx
import { Show } from 'solid-js';
import type { UpdateRecommendation } from '../../lib/types';

interface AlertBannerProps {
  alert: UpdateRecommendation;
  onAccept: () => void;
  onDismiss: () => void;
  loading?: boolean;
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
      const original = props.alert.originalSellPrice;
      const suggested = props.alert.newSellPrice;
      const fromText = (typeof original === 'number' && original > 0)
        ? ` (from ${formatPrice(original)}gp)`
        : '';
      return `${props.alert.reason} Suggested sell price: ${formatPrice(suggested)}gp${fromText} to improve fill probability.`;
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

  return (
    <div class={`alert-banner ${isDanger() ? 'alert-banner-danger' : ''}`}>
      <div class="alert-banner-header">
        <span class="alert-banner-icon">{getIcon()}</span>
        <span class={`alert-banner-title ${isDanger() ? 'alert-banner-title-danger' : ''}`}>
          {getHeader()}
        </span>
      </div>

      <p class="alert-banner-body">{getBody()}</p>

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
