// packages/web/src/components/trades/TradeDetail.tsx
import { createSignal } from 'solid-js';
import type { TradeViewModel, Guidance } from '../../lib/trade-types';
import { CheckInBar } from './CheckInBar';
import { GuidancePrompt } from './GuidancePrompt';

interface TradeDetailProps {
  trade: TradeViewModel;
  onCheckIn: (progress: number) => Promise<{ guidance?: Guidance }>;
  onAdvance: () => Promise<void>;
  onCancel: () => void;
  onClose: () => void;
}

export function TradeDetail(props: TradeDetailProps) {
  const [loading, setLoading] = createSignal(false);
  const [guidance, setGuidance] = createSignal<Guidance | undefined>(undefined);
  const [pendingProgress, setPendingProgress] = createSignal<number | undefined>(undefined);

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
  const timeInTradeMinutes = () => Math.round((Date.now() - props.trade.createdAt.getTime()) / (1000 * 60));

  return (
    <div class="trade-detail">
      <button class="trade-detail-close" onClick={() => props.onClose()} aria-label="Close trade detail">×</button>

      <div class="trade-detail-header">
        <h3 class="trade-detail-title">{props.trade.itemName}</h3>
        <span class={`trade-detail-phase phase-${props.trade.phase}`}>
          {props.trade.phase.toUpperCase()}
        </span>
      </div>

      <div class="trade-detail-prices-exact">
        <div class="price-box buy-price">
          <span class="price-label">Buy at</span>
          <span class="price-value">{formatExactGold(actualBuy())}</span>
        </div>
        <div class="price-arrow">→</div>
        <div class="price-box sell-price">
          <span class="price-label">Sell at</span>
          <span class="price-value">{formatExactGold(actualSell())}</span>
        </div>
      </div>

      <div class="trade-detail-profit">
        Target profit: <strong>+{formatGold(props.trade.targetProfit)}</strong>
      </div>

      <hr class="trade-detail-divider" />

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

      <div class="trade-detail-stats">
        <div class="trade-detail-stat">
          <span class="stat-label">Time in trade</span>
          <span class="stat-value">
            {timeInTradeMinutes()} min
          </span>
        </div>
        <div class="trade-detail-stat">
          <span class="stat-label">Quantity</span>
          <span class="stat-value">{props.trade.quantity.toLocaleString()}</span>
        </div>
      </div>

      <button
        class="trade-detail-cancel"
        onClick={() => props.onCancel()}
        disabled={loading()}
      >
        Cancel trade
      </button>

      <style>{`
        .trade-detail {
          background: var(--surface-2, #1a1a2e);
          border: 1px solid var(--accent, #4f46e5);
          border-radius: 8px;
          padding: 1rem;
          position: relative;
        }

        .trade-detail-close {
          position: absolute;
          top: 0.5rem;
          right: 0.5rem;
          background: none;
          border: none;
          color: var(--text-tertiary, #666);
          font-size: 1.5rem;
          cursor: pointer;
          padding: 0.25rem 0.5rem;
          line-height: 1;
        }

        .trade-detail-close:hover {
          color: var(--text-primary, #fff);
        }

        .trade-detail-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .trade-detail-title {
          margin: 0;
          font-size: 1.125rem;
        }

        .trade-detail-phase {
          font-size: 0.75rem;
          font-weight: 700;
          padding: 0.25rem 0.75rem;
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

        .trade-detail-prices-exact {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.5rem;
          margin: 0.75rem 0;
          padding: 0.75rem;
          background: var(--surface-3, #252540);
          border-radius: 6px;
        }

        .price-box {
          display: flex;
          flex-direction: column;
          align-items: center;
          flex: 1;
        }

        .price-label {
          font-size: 0.7rem;
          text-transform: uppercase;
          color: var(--text-tertiary, #666);
          margin-bottom: 0.25rem;
        }

        .price-value {
          font-size: 0.95rem;
          font-weight: 600;
          font-family: monospace;
        }

        .buy-price .price-value {
          color: var(--warning, #f59e0b);
        }

        .sell-price .price-value {
          color: var(--success, #22c55e);
        }

        .price-arrow {
          color: var(--text-tertiary, #666);
          font-size: 1rem;
        }

        .trade-detail-profit {
          margin-top: 0.25rem;
          color: var(--text-secondary, #aaa);
          font-size: 0.875rem;
        }

        .trade-detail-profit strong {
          color: var(--success, #22c55e);
        }

        .trade-detail-divider {
          border: none;
          border-top: 1px solid var(--border, #333);
          margin: 1rem 0;
        }

        .trade-detail-submit {
          width: 100%;
          padding: 0.625rem;
          background: var(--accent, #4f46e5);
          color: white;
          border: none;
          border-radius: 6px;
          font-weight: 600;
          cursor: pointer;
          margin-top: 0.5rem;
        }

        .trade-detail-submit:hover:not(:disabled) {
          background: var(--accent-hover, #4338ca);
        }

        .trade-detail-submit:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .trade-detail-stats {
          display: flex;
          gap: 1.5rem;
        }

        .trade-detail-stat {
          display: flex;
          flex-direction: column;
          gap: 0.125rem;
        }

        .stat-label {
          font-size: 0.75rem;
          color: var(--text-tertiary, #666);
        }

        .stat-value {
          font-weight: 600;
          color: var(--text-primary, #fff);
        }

        .trade-detail-cancel {
          width: 100%;
          padding: 0.5rem;
          background: transparent;
          color: var(--error, #ef4444);
          border: 1px solid var(--error, #ef4444);
          border-radius: 6px;
          cursor: pointer;
          margin-top: 1rem;
        }

        .trade-detail-cancel:hover:not(:disabled) {
          background: var(--error-bg, #3d2020);
        }

        .trade-detail-cancel:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}
