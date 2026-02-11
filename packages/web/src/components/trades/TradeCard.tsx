// packages/web/src/components/trades/TradeCard.tsx
import type { TradeViewModel } from '../../lib/trade-types';
import type { UpdateRecommendation } from '../../lib/types';
import { calculateFlipProfit } from '../../lib/ge-tax';
import Tooltip from '../Tooltip';

interface TradeCardProps {
  trade: TradeViewModel;
  onClick: () => void;
  onCancel: () => void;
  expanded?: boolean;
  alert?: UpdateRecommendation;
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

  // Reflect alert/suggested prices in the profit shown on the card.
  // Otherwise the "PRED. PROFIT" number can look stale while the user is
  // being told to revise their sell price.
  const effectiveSellPrice = () => {
    if (props.alert?.type === 'ADJUST_PRICE') {
      return props.alert.newSellPrice ?? props.trade.suggestedSellPrice ?? props.trade.sellPrice;
    }
    if (props.alert?.type === 'SELL_NOW') {
      return props.alert.adjustedSellPrice ?? props.trade.sellPrice;
    }
    return props.trade.suggestedSellPrice ?? props.trade.sellPrice;
  };

  const originalProfit = () =>
    calculateFlipProfit(props.trade.buyPrice, props.trade.sellPrice, props.trade.quantity);
  const effectiveProfit = () =>
    calculateFlipProfit(props.trade.buyPrice, effectiveSellPrice(), props.trade.quantity);
  const profitDir = () => {
    const orig = originalProfit();
    const next = effectiveProfit();
    if (next > orig) return 'up' as const;
    if (next < orig) return 'down' as const;
    return 'neutral' as const;
  };

  const timeInTradeMinutes = () =>
    Math.max(0, Math.round((Date.now() - props.trade.createdAt.getTime()) / (1000 * 60)));

  const isOverdue = () => {
    const expectedHours = props.trade.expectedHours;
    if (!Number.isFinite(expectedHours) || expectedHours <= 0) return false;
    return timeInTradeMinutes() > expectedHours * 60;
  };

  const timeInTradeText = () => {
    const mins = timeInTradeMinutes();
    if (props.expanded) {
      if (mins >= 60) return `${Math.floor(mins / 60)}h ${mins % 60}m`;
      return `${mins}m`;
    }
    return `${Math.round(mins / 60)}h`;
  };

  const getStatusBadge = () => {
    switch (props.trade.status) {
      case 'check_in':
        return (
          <Tooltip text="Tap to mark your order as filled or partially filled" position="bottom" delay={200}>
            <span class="status-badge status-check-in">Check-in</span>
          </Tooltip>
        );
      case 'needs_attention':
        return <span class="status-badge status-attention">Needs attention</span>;
      case 'ready':
        return <span class="status-badge status-ready">Ready</span>;
      default:
        return <span class="status-badge status-on-track">On track</span>;
    }
  };

  return (
    <div
      class={`trade-card ${props.expanded ? 'trade-card-expanded' : ''} ${(props.alert || props.trade.suggestedSellPrice) ? 'trade-card-alert' : ''}`}
      onClick={() => props.onClick()}
      role="button"
      tabIndex={0}
    >
      <div class="trade-card-header">
        <div class="trade-card-item">
          <span class="trade-card-name">{props.trade.itemName}</span>
        </div>
        <div class="trade-card-center">
          {(props.alert || props.trade.suggestedSellPrice) ? (
            <span class="status-badge status-alert">Price alert</span>
          ) : isOverdue() ? (
            <Tooltip text="This trade has been open longer than the expected completion time." position="bottom" delay={200}>
              <span class="status-badge status-overdue">Overdue</span>
            </Tooltip>
          ) : (
            getStatusBadge()
          )}
        </div>
        <div class="trade-card-profit">
          <span class="trade-card-kicker">PRED. PROFIT</span>
          {effectiveSellPrice() !== props.trade.sellPrice ? (
            <span class={`trade-card-profit-value is-split ${profitDir() === 'up' ? 'is-up' : profitDir() === 'down' ? 'is-down' : ''}`}>
              <span class="trade-card-profit-old">
                {(originalProfit() >= 0 ? '+' : '-')}{formatGold(Math.abs(originalProfit()))}
              </span>
              <span class="trade-card-profit-new">
                {(effectiveProfit() >= 0 ? '+' : '-')}{formatGold(Math.abs(effectiveProfit()))}
              </span>
            </span>
          ) : (
            <span class="trade-card-profit-value">
              {(effectiveProfit() >= 0 ? '+' : '-')}{formatGold(Math.abs(effectiveProfit()))}
            </span>
          )}
        </div>
      </div>

      <div class="trade-card-footer">
        <div class="trade-card-footer-labels" aria-hidden="true">
          <span>QTY</span>
          <span>TIME IN TRADE</span>
          <span>STATUS</span>
        </div>
        <div class="trade-card-footer-values">
          <span class="trade-card-qty-value">{props.trade.quantity.toLocaleString()}</span>
          <span class="trade-card-time-value">{timeInTradeText()}</span>
          <span class={`trade-card-phase-pill ${props.trade.phase === 'buying' ? 'badge-success' : 'badge-warning'}`}>
            {props.trade.phase === 'buying' ? 'Buying' : 'Selling'}
          </span>
        </div>
      </div>

      <style>{`
        .trade-card {
          position: relative;
          background: var(--surface-1);
          border: 1px solid var(--border);
          border-radius: var(--radius-xl);
          padding: 0.95rem;
          cursor: pointer;
          transition: border-color 0.3s ease, box-shadow 0.6s var(--ease-hero), transform 0.6s var(--ease-hero), background 0.3s ease;
          box-shadow: var(--shadow-sm);
        }

        .trade-card::before {
          display: none;
        }

        .trade-card::after {
          display: none;
        }

        .trade-card > * {
          position: relative;
          z-index: 1;
        }

        .trade-card:hover {
          border-color: var(--border-light);
          transform: translateY(-4px) scale(1.01);
          box-shadow: var(--shadow-md);
        }

        .trade-card:focus {
          outline: none;
        }

        .trade-card-expanded {
          border-color: var(--border-light);
          background: var(--surface-2);
          border-bottom-left-radius: 0;
          border-bottom-right-radius: 0;
          border-bottom: none;
          box-shadow: none;
          transform: translateY(0) scale(1);
        }

        .trade-card-expanded:hover {
          transform: translateY(0) scale(1);
          box-shadow: none;
          border-color: var(--border-light);
        }

        .trade-card-expanded::before,
        .trade-card-expanded::after {
          opacity: 0;
        }

        .trade-card-header {
          display: grid;
          grid-template-columns: 1fr auto 1fr;
          align-items: start;
          gap: 0.75rem;
          margin-bottom: 0.75rem;
        }

        .trade-card-item {
          grid-column: 1;
          justify-self: start;
          min-width: 0;
          display: flex;
          flex-direction: column;
          gap: 0.3rem;
        }

        .trade-card-center {
          grid-column: 2;
          justify-self: center;
          display: flex;
          align-items: center;
          min-width: 0;
        }

        .trade-card-name {
          display: block;
          font-weight: 600;
          color: var(--text-primary);
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }

        .trade-card-phase-pill {
          display: inline-block;
          font-size: var(--font-size-xs);
          font-weight: 600;
          padding: 0.2rem 0.55rem;
          border-radius: var(--radius-full);
          cursor: default;
          white-space: nowrap;
          line-height: 1.2;
        }

        .trade-card-profit {
          grid-column: 3;
          justify-self: end;
          display: flex;
          flex-direction: column;
          align-items: flex-end;
          gap: 0.125rem;
          text-align: right;
          line-height: 1.05;
          min-width: 0;
        }

        .trade-card-kicker {
          display: block;
          font-size: 0.6rem;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          color: var(--text-muted);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          max-width: 100%;
        }

        .trade-card-profit-value {
          font-weight: 700;
          color: var(--success);
          font-size: var(--font-size-sm);
          white-space: nowrap;
        }

        .trade-card-profit-value.is-split {
          display: inline-flex;
          flex-direction: column;
          align-items: flex-end;
          gap: 0.1rem;
          color: var(--text-primary);
        }

        .trade-card-profit-old {
          text-decoration: line-through;
          color: var(--text-muted);
          opacity: 0.75;
          font-size: 0.7rem;
          font-weight: 600;
        }

        .trade-card-profit-new {
          font-size: var(--font-size-sm);
          font-weight: 800;
        }

        .trade-card-profit-value.is-split.is-up .trade-card-profit-new {
          color: var(--success);
        }

        .trade-card-profit-value.is-split.is-down .trade-card-profit-new {
          color: var(--danger);
        }

        .trade-card-footer {
          display: flex;
          flex-direction: column;
          align-items: stretch;
          gap: 0.2rem;
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
        }

        .trade-card-footer-labels,
        .trade-card-footer-values {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          align-items: center;
          gap: 0.5rem;
        }

        .trade-card-footer-labels {
          font-size: 0.62rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: var(--text-muted);
        }

        .trade-card-footer-values {
          font-weight: 600;
          color: var(--text-primary);
          min-width: 0;
        }

        .trade-card-footer-labels > *:nth-child(1),
        .trade-card-footer-values > *:nth-child(1) {
          justify-self: start;
        }

        .trade-card-footer-labels > *:nth-child(2),
        .trade-card-footer-values > *:nth-child(2) {
          justify-self: center;
        }

        .trade-card-footer-labels > *:nth-child(3),
        .trade-card-footer-values > *:nth-child(3) {
          justify-self: end;
        }

        .trade-card-qty-value {
          font-family: var(--font-mono);
          white-space: nowrap;
        }

        .trade-card-time-value {
          white-space: nowrap;
        }

        .status-badge {
          display: inline-flex;
          align-items: center;
          font-size: var(--font-size-xs);
          font-weight: 600;
          padding: 0.2rem 0.55rem;
          border-radius: var(--radius-full);
          white-space: nowrap;
        }

        .status-on-track {
          background: var(--success-light);
          color: var(--success);
        }

        .status-overdue {
          background: var(--warning-light);
          color: var(--warning);
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

        .status-alert {
          background: var(--danger-light);
          color: var(--danger);
        }

        .trade-card-alert {
          border-color: color-mix(in srgb, var(--warning) 60%, transparent);
        }

        .trade-card-alert:hover {
          border-color: color-mix(in srgb, var(--warning) 60%, transparent);
        }

        .trade-card-alert.trade-card-expanded {
          border-color: color-mix(in srgb, var(--warning) 60%, transparent);
        }
      `}</style>
    </div>
  );
}
