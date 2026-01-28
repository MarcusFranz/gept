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
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          padding: 0.875rem;
          cursor: pointer;
          transition: border-color var(--transition-fast);
        }

        .trade-card:hover {
          border-color: var(--accent);
        }

        .trade-card-expanded {
          border-color: var(--accent);
          background: var(--bg-tertiary);
        }

        .trade-card-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .trade-card-name {
          font-weight: 600;
          color: var(--text-primary);
        }

        .trade-card-phase {
          font-size: var(--font-size-xs);
          font-weight: 700;
          padding: 0.125rem 0.5rem;
          border-radius: var(--radius-sm);
        }

        .phase-buying {
          background: var(--gold-light);
          color: var(--gold);
        }

        .phase-selling {
          background: var(--success-light);
          color: var(--success);
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
          background: var(--bg-tertiary);
          border-radius: var(--radius-sm);
          overflow: hidden;
        }

        .trade-card-bar-fill {
          height: 100%;
          background: var(--accent);
          transition: width 0.3s ease;
        }

        .trade-card-profit {
          font-weight: 600;
          color: var(--success);
          font-size: var(--font-size-sm);
          white-space: nowrap;
        }

        .trade-card-footer {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .trade-card-time {
          font-size: var(--font-size-xs);
          color: var(--text-muted);
        }

        .status-badge {
          font-size: var(--font-size-xs);
          font-weight: 600;
          padding: 0.125rem 0.5rem;
          border-radius: var(--radius-sm);
        }

        .status-check-in {
          background: var(--warning-light);
          color: var(--warning);
        }

        .status-attention {
          background: var(--danger-light);
          color: var(--danger);
        }

        .status-ready {
          background: var(--success-light);
          color: var(--success);
        }
      `}</style>
    </div>
  );
}
