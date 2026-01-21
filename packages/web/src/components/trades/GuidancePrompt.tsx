// packages/web/src/components/trades/GuidancePrompt.tsx
import type { Guidance } from '../../lib/trade-types';

interface GuidancePromptProps {
  guidance: Guidance;
  onAccept: () => void;
  onDismiss: () => void;
  loading?: boolean;
}

export function GuidancePrompt(props: GuidancePromptProps) {
  const getActionLabel = () => {
    switch (props.guidance.action) {
      case 'relist': return 'Relist';
      case 'exit': return 'Exit trade';
      case 'sell_now': return 'Sell now';
      default: return 'OK';
    }
  };

  const getDismissLabel = () => {
    switch (props.guidance.action) {
      case 'relist': return 'Keep waiting';
      case 'exit': return 'Keep waiting';
      case 'sell_now': return 'Stick to plan';
      default: return 'Dismiss';
    }
  };

  const formatPrice = (price: number) => {
    if (price >= 1_000_000) {
      return (price / 1_000_000).toFixed(2) + 'M';
    } else if (price >= 1_000) {
      return (price / 1_000).toFixed(1) + 'K';
    }
    return price.toLocaleString();
  };

  return (
    <div class="guidance-prompt">
      <div class="guidance-header">
        <span class="guidance-icon">⚠️</span>
        <span class="guidance-title">
          {props.guidance.action === 'relist' && 'Filling slower than expected'}
          {props.guidance.action === 'exit' && 'Consider exiting this trade'}
          {props.guidance.action === 'sell_now' && 'Opportunity to sell now'}
          {props.guidance.action === 'hold' && 'On track'}
        </span>
      </div>

      {props.guidance.params?.newPrice && (
        <p class="guidance-recommendation">
          Relist at <strong>{formatPrice(props.guidance.params.newPrice)}</strong>
          {props.guidance.params.priceDelta && (
            <span class="guidance-delta">
              {' '}(+{formatPrice(props.guidance.params.priceDelta)} from your order)
            </span>
          )}
          {props.guidance.params.expectedSpeedup && (
            <span class="guidance-speedup">
              , should fill {props.guidance.params.expectedSpeedup}
            </span>
          )}
        </p>
      )}

      {!props.guidance.params?.newPrice && props.guidance.reason && (
        <p class="guidance-reason">{props.guidance.reason}</p>
      )}

      <div class="guidance-actions">
        <button
          class="guidance-btn guidance-btn-primary"
          onClick={() => props.onAccept()}
          disabled={props.loading}
        >
          {props.loading ? 'Processing...' : getActionLabel()}
        </button>
        <button
          class="guidance-btn guidance-btn-secondary"
          onClick={() => props.onDismiss()}
          disabled={props.loading}
        >
          {getDismissLabel()}
        </button>
      </div>

      <style>{`
        .guidance-prompt {
          background: var(--warning-bg, #2d2a1f);
          border: 1px solid var(--warning-border, #b59f3b);
          border-radius: 8px;
          padding: 1rem;
          margin: 1rem 0;
        }

        .guidance-header {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.5rem;
        }

        .guidance-icon {
          font-size: 1.25rem;
        }

        .guidance-title {
          font-weight: 600;
          color: var(--warning-text, #e5c767);
        }

        .guidance-recommendation {
          margin: 0.5rem 0;
          color: var(--text-primary, #fff);
        }

        .guidance-delta {
          color: var(--text-secondary, #aaa);
          font-size: 0.875rem;
        }

        .guidance-speedup {
          color: var(--success, #22c55e);
          font-size: 0.875rem;
        }

        .guidance-reason {
          margin: 0.5rem 0;
          color: var(--text-secondary, #aaa);
          font-size: 0.875rem;
        }

        .guidance-actions {
          display: flex;
          gap: 0.75rem;
          margin-top: 1rem;
        }

        .guidance-btn {
          flex: 1;
          padding: 0.625rem 1rem;
          border: none;
          border-radius: 6px;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.15s;
        }

        .guidance-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .guidance-btn-primary {
          background: var(--warning, #eab308);
          color: black;
        }

        .guidance-btn-primary:hover:not(:disabled) {
          background: var(--warning-hover, #ca9a06);
        }

        .guidance-btn-secondary {
          background: var(--surface-3, #333);
          color: var(--text-primary, #fff);
        }

        .guidance-btn-secondary:hover:not(:disabled) {
          background: var(--surface-4, #444);
        }
      `}</style>
    </div>
  );
}
