// packages/web/src/components/trades/TradeCard.tsx
import type { TradeViewModel } from '../../lib/trade-types';

interface TradeCardProps {
  trade: TradeViewModel;
  onClick: () => void;
  expanded?: boolean;
}

export function TradeCard(props: TradeCardProps) {
  const formatGold = (amount: number) => {
    if (amount >= 1_000_000) {
      return (amount / 1_000_000).toFixed(1) + 'M';
    } else if (amount >= 1_000) {
      return Math.round(amount / 1_000) + 'K';
    }
    return amount.toLocaleString();
  };

  const getStatusBadge = () => {
    switch (props.trade.status) {
      case 'check_in':
        return <span class="status-badge status-check-in">Check-in</span>;
      case 'needs_attention':
        return <span class="status-badge status-attention">Needs attention</span>;
      case 'ready':
        return <span class="status-badge status-ready">Ready</span>;
      default:
        return null;
    }
  };

  return (
    <div
      class={`trade-card ${props.expanded ? 'trade-card-expanded' : ''}`}
      onClick={() => props.onClick()}
      role="button"
      tabIndex={0}
    >
      <div class="trade-card-header">
        <div class="trade-card-item">
          <span class="trade-card-name">{props.trade.itemName}</span>
        </div>
        <span class={`trade-card-phase phase-${props.trade.phase}`}>
          {props.trade.phase.toUpperCase()}
        </span>
      </div>

      <div class="trade-card-progress">
        <div class="trade-card-bar">
          <div
            class="trade-card-bar-fill"
            style={{ width: `${props.trade.progress}%` }}
          />
        </div>
        <span class="trade-card-profit">+{formatGold(props.trade.targetProfit)}</span>
      </div>

      <div class="trade-card-footer">
        <span class="trade-card-time">
          {Math.round((Date.now() - props.trade.createdAt.getTime()) / (1000 * 60 * 60))}h in trade
        </span>
        {getStatusBadge()}
      </div>

      <style>{`
        .trade-card {
          background: var(--surface-2, #1a1a2e);
          border: 1px solid var(--border, #333);
          border-radius: 8px;
          padding: 0.875rem;
          cursor: pointer;
          transition: border-color 0.15s, transform 0.15s;
        }

        .trade-card:hover {
          border-color: var(--accent, #4f46e5);
        }

        .trade-card-expanded {
          border-color: var(--accent, #4f46e5);
          background: var(--surface-3, #252540);
        }

        .trade-card-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .trade-card-name {
          font-weight: 600;
          color: var(--text-primary, #fff);
        }

        .trade-card-phase {
          font-size: 0.75rem;
          font-weight: 700;
          padding: 0.125rem 0.5rem;
          border-radius: 4px;
        }

        .phase-buying {
          background: var(--info-bg, #1e3a5f);
          color: var(--info, #60a5fa);
        }

        .phase-selling {
          background: var(--success-bg, #1a3d2e);
          color: var(--success, #22c55e);
        }

        .trade-card-progress {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          margin-bottom: 0.5rem;
        }

        .trade-card-bar {
          flex: 1;
          height: 6px;
          background: var(--surface-4, #333);
          border-radius: 3px;
          overflow: hidden;
        }

        .trade-card-bar-fill {
          height: 100%;
          background: var(--accent, #4f46e5);
          transition: width 0.3s ease;
        }

        .trade-card-profit {
          font-weight: 600;
          color: var(--success, #22c55e);
          font-size: 0.875rem;
          white-space: nowrap;
        }

        .trade-card-footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .trade-card-time {
          font-size: 0.75rem;
          color: var(--text-tertiary, #666);
        }

        .status-badge {
          font-size: 0.75rem;
          font-weight: 600;
          padding: 0.125rem 0.5rem;
          border-radius: 4px;
        }

        .status-check-in {
          background: var(--warning-bg, #3d3520);
          color: var(--warning, #eab308);
        }

        .status-attention {
          background: var(--error-bg, #3d2020);
          color: var(--error, #ef4444);
        }

        .status-ready {
          background: var(--success-bg, #1a3d2e);
          color: var(--success, #22c55e);
        }
      `}</style>
    </div>
  );
}
