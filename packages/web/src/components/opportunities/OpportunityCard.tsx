// packages/web/src/components/opportunities/OpportunityCard.tsx
import { Show, For } from 'solid-js';
import type { Opportunity } from '../../lib/trade-types';
import Tooltip from '../Tooltip';

interface OpportunityCardProps {
  opportunity: Opportunity;
  onClick: () => void;
  expanded?: boolean;
  onAddToTrades: () => void;
  loading?: boolean;
}

export function OpportunityCard(props: OpportunityCardProps) {
  const opp = () => props.opportunity;

  const formatGold = (amount: number) => {
    if (amount >= 1_000_000) {
      return (amount / 1_000_000).toFixed(1) + 'M';
    } else if (amount >= 1_000) {
      return Math.round(amount / 1_000) + 'K';
    }
    return amount.toLocaleString();
  };

  // Exact GP price for GE offers - no abbreviation
  const formatExactGold = (amount: number) => {
    return amount.toLocaleString() + ' gp';
  };

  const getChipColorClass = (chip: { label: string; type: string }) => {
    const label = chip.label.toLowerCase();
    if (label.includes('confidence')) return 'chip-confidence';
    if (label.includes('volume')) return 'chip-volume';
    if (label.includes('fill') || label.includes('quick')) return 'chip-speed';
    if (label.includes('trending up')) return 'chip-trend-up';
    if (label.includes('trending down')) return 'chip-trend-down';
    if (label.includes('hold') || label.includes('longer')) return 'chip-time';
    // Fallback to type-based
    if (chip.type === 'positive') return 'chip-trend-up';
    if (chip.type === 'negative') return 'chip-trend-down';
    return 'chip-time';
  };

  const formatHours = (hours: number) => {
    if (hours < 1) return `~${Math.round(hours * 60)}min`;
    return `~${hours.toFixed(1)}h`;
  };

  return (
    <div
      class={`opportunity-card ${props.expanded ? 'opportunity-card-expanded' : ''}`}
      onClick={() => !props.expanded && props.onClick()}
    >
      <div class="opportunity-card-header">
        <span class="opportunity-card-name">{opp().item}</span>
        <span class="opportunity-card-profit">+{formatGold(opp().expectedProfit)}</span>
      </div>

      <div class="opportunity-card-meta">
        <span>Buy {formatGold(opp().buyPrice)} → Sell {formatGold(opp().sellPrice)}</span>
        <span>·</span>
        <span>{formatHours(opp().expectedHours)} hold</span>
        <span>·</span>
        <Tooltip text="Model prediction confidence based on historical accuracy">
          <span class={`confidence-badge badge-${(opp().confidence ?? 'medium') === 'high' ? 'success' : (opp().confidence ?? 'medium') === 'medium' ? 'warning' : 'danger'}`}>
            {(opp().confidence ?? 'medium').charAt(0).toUpperCase() + (opp().confidence ?? 'medium').slice(1)}
          </span>
        </Tooltip>
      </div>

      <Show when={props.expanded}>
        <div class="opportunity-card-details">
          <div class="opportunity-card-prices">
            <div class="price-box buy-price">
              <span class="price-label">Buy at</span>
              <span class="price-value">{formatExactGold(opp().buyPrice)}</span>
            </div>
            <div class="price-arrow">→</div>
            <div class="price-box sell-price">
              <span class="price-label">Sell at</span>
              <span class="price-value">{formatExactGold(opp().sellPrice)}</span>
            </div>
          </div>
          <div class="opportunity-card-row">
            <span>Quantity:</span>
            <span>{opp().quantity.toLocaleString()}</span>
          </div>
          <div class="opportunity-card-row">
            <span>Capital:</span>
            <span>{formatGold(opp().capitalRequired)}</span>
          </div>
          <div class="opportunity-card-row">
            <span>Expected profit:</span>
            <span class="profit-value">+{formatGold(opp().expectedProfit)}</span>
          </div>

          <Show when={opp().whyChips?.length}>
            <div class="opportunity-card-chips">
              <For each={opp().whyChips}>
                {(chip) => (
                  <span class={`chip ${getChipColorClass(chip)}`}>
                    {chip.icon} {chip.label}
                  </span>
                )}
              </For>
            </div>
          </Show>

          <Show when={opp().volume24h}>
            <div class="opportunity-card-volume">
              24h Volume: {opp().volume24h?.toLocaleString()}
            </div>
          </Show>

          <button
            class="opportunity-card-cta"
            onClick={(e) => {
              e.stopPropagation();
              props.onAddToTrades();
            }}
            disabled={props.loading}
          >
            {props.loading ? 'Adding...' : 'Add to trades'}
          </button>

          <button
            class="opportunity-card-close"
            onClick={(e) => {
              e.stopPropagation();
              props.onClick(); // Toggle off
            }}
          >
            Close
          </button>
        </div>
      </Show>

      <style>{`
        .opportunity-card {
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          padding: 0.875rem;
          cursor: pointer;
          transition: border-color var(--transition-fast);
        }

        .opportunity-card:hover {
          border-color: var(--accent);
        }

        .opportunity-card-expanded {
          border-color: var(--accent);
          cursor: default;
        }

        .opportunity-card-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .opportunity-card-name {
          font-weight: 600;
          color: var(--text-primary);
        }

        .opportunity-card-profit {
          font-weight: 700;
          color: var(--success);
        }

        .opportunity-card-meta {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-top: 0.25rem;
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
        }

        .confidence-badge {
          font-size: var(--font-size-xs);
          font-weight: 600;
          padding: 0.0625rem 0.375rem;
          border-radius: var(--radius-sm);
          cursor: default;
        }

        .opportunity-card-details {
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid var(--border);
        }

        .opportunity-card-prices {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.5rem;
          margin-bottom: 1rem;
          padding: 0.75rem;
          background: var(--bg-tertiary);
          border-radius: var(--radius-md);
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

        .price-arrow {
          color: var(--text-muted);
          font-size: 1rem;
        }

        .opportunity-card-row {
          display: flex;
          justify-content: space-between;
          font-size: var(--font-size-sm);
          margin-bottom: 0.25rem;
        }

        .profit-value {
          color: var(--success);
          font-weight: 600;
        }

        .opportunity-card-chips {
          display: flex;
          flex-wrap: wrap;
          gap: 0.5rem;
          margin: 1rem 0;
        }

        .chip {
          font-size: var(--font-size-xs);
          padding: 0.25rem 0.5rem;
          border-radius: var(--radius-sm);
          white-space: nowrap;
        }

        .chip-confidence {
          background: var(--chip-confidence-light);
          color: var(--chip-confidence);
        }

        .chip-volume {
          background: var(--chip-volume-light);
          color: var(--chip-volume);
        }

        .chip-speed {
          background: var(--chip-speed-light);
          color: var(--chip-speed);
        }

        .chip-time {
          background: var(--chip-time-light);
          color: var(--chip-time);
        }

        .chip-trend-up {
          background: var(--success-light);
          color: var(--success);
        }

        .chip-trend-down {
          background: var(--danger-light);
          color: var(--danger);
        }

        .opportunity-card-volume {
          font-size: var(--font-size-xs);
          color: var(--text-muted);
          margin-bottom: 1rem;
        }

        .opportunity-card-cta {
          width: 100%;
          padding: 0.75rem;
          background: var(--action);
          color: var(--btn-text-dark);
          border: none;
          border-radius: var(--radius-md);
          font-weight: 600;
          cursor: pointer;
          margin-top: 1rem;
        }

        .opportunity-card-cta:hover:not(:disabled) {
          background: var(--action-hover);
        }

        .opportunity-card-cta:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .opportunity-card-close {
          width: 100%;
          padding: 0.5rem;
          background: transparent;
          color: var(--text-secondary);
          border: none;
          cursor: pointer;
          margin-top: 0.5rem;
        }

        .opportunity-card-close:hover {
          color: var(--text-primary);
        }
      `}</style>
    </div>
  );
}
