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
          background: var(--surface-2);
          border: 1px solid color-mix(in srgb, var(--warning) 40%, transparent);
          border-radius: var(--radius-lg);
          padding: 1rem;
          margin: 1rem 0;
          box-shadow: var(--shadow-sm);
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
          color: var(--warning);
        }

        .guidance-recommendation {
          margin: 0.5rem 0;
          color: var(--text-primary);
        }

        .guidance-delta {
          color: var(--text-secondary);
          font-size: var(--font-size-sm);
        }

        .guidance-speedup {
          color: var(--success);
          font-size: var(--font-size-sm);
        }

        .guidance-reason {
          margin: 0.5rem 0;
          color: var(--text-secondary);
          font-size: var(--font-size-sm);
        }

        .guidance-actions {
          display: flex;
          gap: 0.75rem;
          margin-top: 1rem;
        }

        .guidance-btn {
          flex: 1;
          padding: 0.65rem 1rem;
          border: 1px solid transparent;
          border-radius: var(--radius-full);
          font-weight: 600;
          cursor: pointer;
          transition: background var(--transition-fast);
        }

        .guidance-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .guidance-btn-primary {
          background: var(--warning);
          color: var(--btn-text-dark);
          border-color: color-mix(in srgb, var(--warning) 65%, transparent);
        }

        .guidance-btn-primary:hover:not(:disabled) {
          filter: brightness(0.9);
        }

        .guidance-btn-secondary {
          background: var(--surface-1);
          color: var(--text-primary);
          border: 1px solid var(--border);
        }

        .guidance-btn-secondary:hover:not(:disabled) {
          background: var(--surface-2);
        }
      `}</style>
    </div>
  );
}
