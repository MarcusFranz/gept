// packages/web/src/components/opportunities/OpportunityCard.tsx
import { Show, For, createSignal, createEffect } from 'solid-js';
import type { Opportunity } from '../../lib/trade-types';
import Tooltip from '../Tooltip';
import { Sparkline } from '../Sparkline';
import { fetchPriceHistoryWithFallback, type PriceHistoryData } from '../../lib/price-history';

interface OpportunityCardProps {
  opportunity: Opportunity;
  onClick: () => void;
  expanded?: boolean;
  onAddToTrades: (quantity: number) => void;
  loading?: boolean;
  isBeta?: boolean;
}

export function OpportunityCard(props: OpportunityCardProps) {
  const opp = () => props.opportunity;
  const [priceHistory, setPriceHistory] = createSignal<PriceHistoryData | null>(null);
  const [historyLoading, setHistoryLoading] = createSignal(false);
  const [historyAttempted, setHistoryAttempted] = createSignal(false);
  const [quantity, setQuantity] = createSignal(opp().quantity);

  // Lazily fetch price history only when the card is expanded.
  createEffect(() => {
    if (!props.expanded) return;
    if (priceHistory() || historyLoading() || historyAttempted()) return;

    setHistoryLoading(true);
    fetchPriceHistoryWithFallback(opp().itemId)
      .then(data => {
        if (data) setPriceHistory(data);
      })
      .finally(() => {
        setHistoryLoading(false);
        setHistoryAttempted(true);
      });
  });

  createEffect(() => {
    setQuantity(opp().quantity);
  });

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

  const stepSize = () => {
    const base = opp().quantity;
    if (base >= 5000) return 500;
    if (base >= 1000) return 100;
    if (base >= 200) return 25;
    if (base >= 50) return 5;
    return 1;
  };

  const setQuantitySafe = (value: number) => {
    const safe = Math.max(1, Math.floor(value));
    setQuantity(safe);
  };

  const unitProfit = () => {
    const base = opp().quantity || 1;
    return opp().expectedProfit / base;
  };

  const adjustedProfit = () => Math.round(unitProfit() * quantity());
  const adjustedCapital = () => Math.round(opp().buyPrice * quantity());
  const volumeText = () => {
    const volume = opp().volume24h;
    return volume && volume > 0 ? formatGold(volume) : '—';
  };
  const volumeTooltip = () => {
    const volume = opp().volume24h;
    return volume && volume > 0
      ? `${volume.toLocaleString()} 24h volume`
      : '24h volume unavailable';
  };

  return (
    <div
      class={`opportunity-card ${props.expanded ? 'opportunity-card-expanded' : ''}`}
      onClick={() => props.onClick()}
    >
      <div class="opportunity-card-header">
        <span class="opportunity-card-name">
          {opp().item}
          <Show when={props.isBeta}>
            <span class="beta-badge">Beta</span>
          </Show>
        </span>
        <Tooltip text={volumeTooltip()}>
          <span class="opportunity-card-mid" aria-label="24 hour volume">
            <span class="opportunity-card-mid-label">VOL</span>
            <span class="opportunity-card-mid-value">{volumeText()}</span>
          </span>
        </Tooltip>
        <span class="opportunity-card-profit">
          <span class="opportunity-card-profit-label">Pred. Profit:</span>
          <span class="opportunity-card-profit-value">+{formatGold(opp().expectedProfit)}</span>
        </span>
      </div>

      <div class="opportunity-card-meta">
        <div class="opportunity-card-meta-labels">
          <span>CAPITAL</span>
          <span>TIME</span>
          <span>CONFIDENCE</span>
        </div>
        <div class="opportunity-card-meta-values">
          <span>{formatGold(adjustedCapital())}</span>
          <span>{formatHours(opp().expectedHours)}</span>
          <Tooltip text="Model prediction confidence based on historical accuracy">
            {(() => {
              const conf = opp().confidence ?? 'medium';
              const badge = { high: 'success', medium: 'warning', low: 'danger' }[conf] ?? 'warning';
              return <span class={`confidence-badge badge-${badge}`}>{conf.charAt(0).toUpperCase() + conf.slice(1)}</span>;
            })()}
          </Tooltip>
        </div>
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
          <Show when={!historyAttempted() || historyLoading() || priceHistory()}>
            <div class="opportunity-card-sparkline-wrap">
              <div class="opportunity-card-sparkline" aria-label="Price history">
                <Sparkline
                  highs={priceHistory()?.highs ?? []}
                  lows={priceHistory()?.lows ?? []}
                  loading={!priceHistory() && !historyAttempted()}
                  width={280}
                  height={56}
                />
              </div>
            </div>
          </Show>
          <div class="opportunity-card-quantity" onClick={(e) => e.stopPropagation()}>
            <span class="quantity-label">Quantity</span>
            <div class="quantity-control">
              <button
                class="quantity-btn"
                onClick={() => setQuantitySafe(quantity() - stepSize())}
                aria-label="Decrease quantity"
              >
                −
              </button>
              <input
                class="quantity-input"
                type="number"
                min="1"
                step={stepSize()}
                value={quantity()}
                onInput={(e) => {
                  const value = Number((e.currentTarget as HTMLInputElement).value);
                  if (!Number.isNaN(value)) setQuantitySafe(value);
                }}
              />
              <button
                class="quantity-btn"
                onClick={() => setQuantitySafe(quantity() + stepSize())}
                aria-label="Increase quantity"
              >
                +
              </button>
            </div>
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

          <button
            class="opportunity-card-cta"
            onClick={(e) => {
              e.stopPropagation();
              props.onAddToTrades(quantity());
            }}
            disabled={props.loading}
          >
            {props.loading ? 'Adding...' : 'Add to trades'}
          </button>
        </div>
      </Show>

      <style>{`
        .opportunity-card {
          background: var(--surface-1);
          border: 1px solid var(--border);
          border-radius: var(--radius-xl);
          padding: 1rem;
          cursor: pointer;
          transition: border-color 0.3s ease, box-shadow 0.6s var(--ease-hero), transform 0.6s var(--ease-hero), background 0.3s ease;
          box-shadow: var(--shadow-sm);
          backdrop-filter: blur(16px);
        }

        .opportunity-card:hover {
          border-color: var(--border-light);
          box-shadow: var(--shadow-md);
          transform: translateY(-4px) scale(1.01);
        }

        .opportunity-card-expanded {
          border-color: var(--border-light);
          cursor: pointer;
          background: var(--surface-2);
          box-shadow: none;
          transform: translateY(0) scale(1);
        }

        .opportunity-card-expanded:hover {
          box-shadow: none;
          transform: translateY(0) scale(1);
        }

        .opportunity-card-header {
          display: grid;
          grid-template-columns: 1fr auto 1fr;
          align-items: center;
          gap: 0.75rem;
          margin-top: -0.1rem;
        }

        .opportunity-card-name {
          font-weight: 600;
          color: var(--text-primary);
          justify-self: start;
        }

        .opportunity-card-profit {
          font-weight: 600;
          color: var(--success);
          font-size: var(--font-size-sm);
          white-space: nowrap;
          justify-self: end;
        }

        .opportunity-card-mid {
          position: relative;
          display: inline-flex;
          align-items: baseline;
          justify-self: center;
          gap: 0.4rem;
          padding: 0;
          opacity: 0.9;
        }

        .opportunity-card-mid > * {
          position: relative;
          z-index: 2;
        }

        .opportunity-card-mid-label {
          font-size: 0.55rem;
          letter-spacing: 0.22em;
          font-weight: 700;
          color: color-mix(in srgb, var(--text-muted) 78%, transparent);
          text-transform: uppercase;
          text-shadow:
            0 1px 0 rgba(255, 255, 255, 0.05),
            0 -1px 0 rgba(0, 0, 0, 0.5);
        }

        .opportunity-card-mid-value {
          font-size: 0.7rem;
          letter-spacing: 0.08em;
          font-weight: 700;
          font-family: var(--font-mono);
          color: color-mix(in srgb, var(--text-secondary) 90%, #ffffff);
          text-shadow:
            0 1px 0 rgba(255, 255, 255, 0.06),
            0 -1px 0 rgba(0, 0, 0, 0.55);
        }

        .opportunity-card-profit-value {
          display: inline-block;
          margin-left: 0.25rem;
          background: linear-gradient(120deg, var(--success) 0%, #b7f5be 45%, var(--success) 100%);
          background-size: 200% 100%;
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
          text-shadow: 0 0 10px rgba(126, 231, 135, 0.3);
          animation: profitSheen 4s ease-in-out infinite;
        }

        .opportunity-card-profit-label {
          color: var(--text-muted);
          font-size: var(--font-size-xs);
          font-weight: 600;
          letter-spacing: 0.02em;
          margin-right: 0.25rem;
        }

        @keyframes profitSheen {
          0%, 100% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
        }

        @media (prefers-reduced-motion: reduce) {
          .opportunity-card-profit-value {
            animation: none;
          }
        }

        .opportunity-card-meta {
          display: flex;
          flex-direction: column;
          align-items: stretch;
          gap: 0.2rem;
          margin-top: 0.25rem;
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
        }

        .opportunity-card-meta-labels,
        .opportunity-card-meta-values {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          align-items: center;
          gap: 0.5rem;
        }

        .opportunity-card-meta-labels {
          font-size: 0.62rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: var(--text-muted);
        }

        .opportunity-card-meta-values {
          font-weight: 600;
          color: var(--text-primary);
        }

        .opportunity-card-meta-values > span:first-child {
          background: linear-gradient(135deg, #f6d37a 0%, #f1b34f 35%, #f9e3a2 70%, #d79b3a 100%);
          background-size: 160% 160%;
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
          text-shadow: 0 1px 0 rgba(0, 0, 0, 0.35);
        }

        .opportunity-card-meta-labels > *:nth-child(1),
        .opportunity-card-meta-values > *:nth-child(1) {
          justify-self: start;
        }

        .opportunity-card-meta-labels > *:nth-child(2),
        .opportunity-card-meta-values > *:nth-child(2) {
          justify-self: center;
        }

        .opportunity-card-meta-labels > *:nth-child(3),
        .opportunity-card-meta-values > *:nth-child(3) {
          justify-self: end;
        }

        .opportunity-card-meta-left {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          flex-wrap: wrap;
        }

        .opportunity-card-meta-right {
          white-space: nowrap;
          color: var(--text-muted);
          font-size: var(--font-size-xs);
          letter-spacing: 0.02em;
        }

        .confidence-badge {
          font-size: var(--font-size-xs);
          font-weight: 600;
          padding: 0.2rem 0.55rem;
          border-radius: var(--radius-full);
          cursor: default;
        }

        .beta-badge {
          display: inline-block;
          font-size: 0.7rem;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          padding: 0.25rem 0.6rem;
          border-radius: var(--radius-full);
          background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
          color: var(--btn-text-dark);
          box-shadow: 0 10px 22px rgba(168, 240, 8, 0.45);
          border: 1px solid color-mix(in srgb, var(--accent) 80%, transparent);
          text-shadow: 0 1px 0 rgba(255, 255, 255, 0.5);
          margin-left: 0.4rem;
          vertical-align: middle;
          animation: betaPulse 2.6s ease-in-out infinite;
        }

        @keyframes betaPulse {
          0%, 100% {
            transform: translateY(0);
            box-shadow: 0 10px 22px rgba(168, 240, 8, 0.45);
          }
          50% {
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(168, 240, 8, 0.6);
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .beta-badge {
            animation: none;
          }
        }

        .opportunity-card-sparkline-wrap {
          display: flex;
          justify-content: center;
        }

        .opportunity-card-sparkline {
          margin: 0.7rem 0 0.9rem;
          display: inline-flex;
          justify-content: center;
          position: relative;
          padding: 0.55rem 0.75rem;
          background:
            var(--surface-2),
            repeating-linear-gradient(0deg, rgba(0, 0, 0, 0.35) 0px, rgba(0, 0, 0, 0.35) 1px, transparent 1px, transparent 12px),
            repeating-linear-gradient(90deg, rgba(0, 0, 0, 0.3) 0px, rgba(0, 0, 0, 0.3) 1px, transparent 1px, transparent 18px);
          border-radius: var(--radius-lg);
          border: none;
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
          padding: 0.85rem;
          background: var(--surface-2);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
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

        .opportunity-card-quantity {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.85rem;
          margin-top: 0.25rem;
          padding: 0.85rem;
          background: var(--surface-2);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
        }

        .quantity-label {
          font-size: var(--font-size-xs);
          text-transform: uppercase;
          letter-spacing: 0.12em;
          color: var(--text-muted);
          font-weight: 700;
          line-height: 1;
        }

        .quantity-control {
          display: inline-flex;
          align-items: center;
          gap: 0.35rem;
          padding: 0;
          background: transparent;
          border: none;
          border-radius: 0;
          margin-top: 0;
        }

        .quantity-btn {
          width: 26px;
          height: 26px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          border-radius: var(--radius-full);
          border: 1px solid var(--border);
          background: var(--surface-3);
          color: var(--text-secondary);
          font-size: 1rem;
          cursor: pointer;
          transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease;
        }

        .quantity-btn:hover {
          background: var(--bg-hover);
          color: var(--text-primary);
          border-color: var(--border-light);
        }

        .quantity-input {
          width: 84px;
          padding: 0.3rem 0.4rem;
          border: 1px solid var(--border);
          background: var(--surface-3);
          color: var(--text-primary);
          font-size: var(--font-size-sm);
          text-align: center;
          font-family: var(--font-mono);
          outline: none;
          border-radius: var(--radius-md);
          line-height: 1.15;
        }

        .quantity-input:focus {
          border-color: var(--border-light);
          box-shadow: 0 0 0 3px rgba(168, 240, 8, 0.12);
        }

        .quantity-input::-webkit-outer-spin-button,
        .quantity-input::-webkit-inner-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }

        .quantity-input[type="number"] {
          -moz-appearance: textfield;
        }

        .profit-value {
          color: var(--success);
          font-weight: 600;
        }

        .opportunity-card-chips {
          display: flex;
          flex-wrap: wrap;
          justify-content: center;
          gap: 0.5rem;
          margin: 1rem 0;
        }

        .chip {
          font-size: var(--font-size-xs);
          padding: 0.3rem 0.6rem;
          border-radius: var(--radius-full);
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
          border-radius: var(--radius-full);
          font-weight: 600;
          cursor: pointer;
          margin-top: 1rem;
          box-shadow: 0 16px 30px -22px rgba(168, 240, 8, 0.5);
          transition: transform 0.4s var(--ease-hero), box-shadow 0.4s var(--ease-hero), background 0.3s ease;
        }

        .opportunity-card-cta:hover:not(:disabled) {
          background: var(--action-hover);
          transform: translateY(-2px) scale(1.01);
          box-shadow: 0 20px 36px -24px rgba(168, 240, 8, 0.6);
        }

        .opportunity-card-cta:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}
