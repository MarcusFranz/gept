// packages/web/src/components/trades/TradeDetail.tsx
import { createSignal, Show } from 'solid-js';
import type { TradeViewModel, Guidance } from '../../lib/trade-types';
import type { UpdateRecommendation } from '../../lib/types';
import { CheckInBar } from './CheckInBar';
import { GuidancePrompt } from './GuidancePrompt';
import { AlertBanner } from './AlertBanner';

interface TradeDetailProps {
  trade: TradeViewModel;
  onCheckIn: (progress: number) => Promise<{ guidance?: Guidance }>;
  onAdvance: () => Promise<void>;
  onCancel: () => void;
  onClose: () => void;
  alert?: UpdateRecommendation;
  onAcceptAlert?: () => void;
  onDismissAlert?: () => void;
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

      <Show when={props.alert}>
        <AlertBanner
          alert={props.alert!}
          onAccept={() => props.onAcceptAlert?.()}
          onDismiss={() => props.onDismissAlert?.()}
          loading={loading()}
        />
      </Show>

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
          background: var(--bg-secondary);
          border: 1px solid var(--accent);
          border-radius: var(--radius-lg);
          padding: 1rem;
          position: relative;
        }

        .trade-detail-close {
          position: absolute;
          top: 0.5rem;
          right: 0.5rem;
          background: none;
          border: none;
          color: var(--text-muted);
          font-size: 1.5rem;
          cursor: pointer;
          padding: 0.25rem 0.5rem;
          line-height: 1;
        }

        .trade-detail-close:hover {
          color: var(--text-primary);
        }

        .trade-detail-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .trade-detail-title {
          margin: 0;
          font-size: var(--font-size-lg);
        }

        .trade-detail-phase {
          font-size: var(--font-size-xs);
          font-weight: 700;
          padding: 0.25rem 0.75rem;
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

        .trade-detail-prices-exact {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 0.5rem;
          margin: 0.75rem 0;
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

        .trade-detail-profit {
          margin-top: 0.25rem;
          color: var(--text-secondary);
          font-size: var(--font-size-sm);
        }

        .trade-detail-profit strong {
          color: var(--success);
        }

        .trade-detail-divider {
          border: none;
          border-top: 1px solid var(--border);
          margin: 1rem 0;
        }

        .trade-detail-submit {
          width: 100%;
          padding: 0.625rem;
          background: var(--accent);
          color: var(--btn-text-dark);
          border: none;
          border-radius: var(--radius-md);
          font-weight: 600;
          cursor: pointer;
          margin-top: 0.5rem;
        }

        .trade-detail-submit:hover:not(:disabled) {
          background: var(--accent-hover);
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
          font-size: var(--font-size-xs);
          color: var(--text-muted);
        }

        .stat-value {
          font-weight: 600;
          color: var(--text-primary);
        }

        .trade-detail-cancel {
          width: 100%;
          padding: 0.5rem;
          background: transparent;
          color: var(--danger);
          border: 1px solid var(--danger);
          border-radius: var(--radius-md);
          cursor: pointer;
          margin-top: 1rem;
        }

        .trade-detail-cancel:hover:not(:disabled) {
          background: var(--danger-light);
        }

        .trade-detail-cancel:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
}
