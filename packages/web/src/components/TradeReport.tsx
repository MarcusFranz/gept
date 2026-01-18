import { createSignal, Show, For } from 'solid-js';
import { formatGold } from '../lib/types';

interface TradeReportProps {
  onClose?: () => void;
  prefilledItem?: {
    itemId: number;
    itemName: string;
    buyPrice?: number;
    sellPrice?: number;
    quantity?: number;
    recId?: string;
    modelId?: string;
  };
}

export default function TradeReport(props: TradeReportProps) {
  const [itemName, setItemName] = createSignal(props.prefilledItem?.itemName || '');
  const [profit, setProfit] = createSignal(0);
  const [notes, setNotes] = createSignal('');
  const [submitting, setSubmitting] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [success, setSuccess] = createSignal(false);

  const submitReport = async () => {
    if (!itemName().trim()) {
      setError('Please enter an item name');
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const response = await fetch('/api/trades/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          itemId: props.prefilledItem?.itemId,
          itemName: itemName(),
          buyPrice: props.prefilledItem?.buyPrice,
          sellPrice: props.prefilledItem?.sellPrice,
          quantity: props.prefilledItem?.quantity,
          profit: profit(),
          notes: notes(),
          recId: props.prefilledItem?.recId,
          modelId: props.prefilledItem?.modelId
        })
      });

      const data = await response.json();
      if (data.success) {
        setSuccess(true);
        // Reset form
        if (!props.prefilledItem) {
          setItemName('');
          setProfit(0);
          setNotes('');
        }
        // Auto close after 2 seconds if in modal
        if (props.onClose) {
          setTimeout(() => props.onClose!(), 2000);
        }
      } else {
        setError(data.error || 'Failed to submit report');
      }
    } catch (err) {
      setError('Failed to submit report');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div class="trade-report">
      <Show when={success()}>
        <div class="report-success">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
            <polyline points="22,4 12,14.01 9,11.01"/>
          </svg>
          <h3>Trade Reported!</h3>
          <p class="text-secondary">Thanks for helping improve GePT recommendations.</p>
        </div>
      </Show>

      <Show when={!success()}>
        <Show when={error()}>
          <div class="alert alert-error mb-4">{error()}</div>
        </Show>

        <div class="form-group">
          <label class="label">Item Name</label>
          <input
            type="text"
            class="input"
            value={itemName()}
            onInput={(e) => setItemName(e.currentTarget.value)}
            placeholder="e.g., Armadyl Godsword"
            disabled={!!props.prefilledItem?.itemName}
          />
        </div>

        <Show when={props.prefilledItem?.buyPrice}>
          <div class="prefilled-info">
            <div class="prefilled-row">
              <span>Buy Price:</span>
              <span class="font-mono">{formatGold(props.prefilledItem!.buyPrice!, 'gp')}</span>
            </div>
            <div class="prefilled-row">
              <span>Sell Price:</span>
              <span class="font-mono">{formatGold(props.prefilledItem!.sellPrice!, 'gp')}</span>
            </div>
            <div class="prefilled-row">
              <span>Quantity:</span>
              <span class="font-mono">{props.prefilledItem!.quantity!.toLocaleString()}</span>
            </div>
          </div>
        </Show>

        <div class="form-group">
          <label class="label">Actual Profit/Loss</label>
          <div class="profit-input-group">
            <button
              class={`profit-toggle ${profit() < 0 ? 'loss' : ''}`}
              onClick={() => setProfit(-Math.abs(profit()))}
              type="button"
            >
              Loss
            </button>
            <input
              type="number"
              class="input profit-input"
              value={Math.abs(profit())}
              onInput={(e) => {
                const val = parseInt(e.currentTarget.value) || 0;
                setProfit(profit() < 0 ? -val : val);
              }}
              placeholder="0"
            />
            <button
              class={`profit-toggle ${profit() >= 0 ? 'profit' : ''}`}
              onClick={() => setProfit(Math.abs(profit()))}
              type="button"
            >
              Profit
            </button>
          </div>
          <p class={`profit-display font-mono text-center mt-2 ${profit() >= 0 ? 'text-success' : 'text-danger'}`}>
            {profit() >= 0 ? '+' : ''}{formatGold(profit())}
          </p>
        </div>

        <div class="form-group">
          <label class="label">Notes (optional)</label>
          <textarea
            class="input"
            rows="3"
            value={notes()}
            onInput={(e) => setNotes(e.currentTarget.value)}
            placeholder="Any additional notes about this trade..."
          />
        </div>

        <div class="report-actions">
          <Show when={props.onClose}>
            <button class="btn btn-secondary" onClick={props.onClose}>
              Cancel
            </button>
          </Show>
          <button
            class="btn btn-primary"
            onClick={submitReport}
            disabled={submitting()}
          >
            {submitting() ? 'Submitting...' : 'Submit Report'}
          </button>
        </div>
      </Show>

      <style>{`
        .trade-report {
          max-width: 480px;
          margin: 0 auto;
        }

        .report-success {
          text-align: center;
          padding: var(--space-6);
        }

        .report-success svg {
          width: 64px;
          height: 64px;
          color: var(--success);
          margin-bottom: var(--space-4);
        }

        .report-success h3 {
          margin-bottom: var(--space-2);
        }

        .alert {
          padding: var(--space-3) var(--space-4);
          border-radius: var(--radius-md);
        }

        .alert-error {
          background-color: var(--danger-light);
          color: var(--danger);
        }

        .prefilled-info {
          padding: var(--space-3);
          background-color: var(--bg-tertiary);
          border-radius: var(--radius-md);
          margin-bottom: var(--space-4);
        }

        .prefilled-row {
          display: flex;
          justify-content: space-between;
          padding: var(--space-1) 0;
        }

        .prefilled-row:not(:last-child) {
          border-bottom: 1px solid var(--border);
        }

        .profit-input-group {
          display: flex;
          gap: var(--space-2);
        }

        .profit-toggle {
          padding: var(--space-2) var(--space-3);
          border: 2px solid var(--border);
          background-color: var(--bg-tertiary);
          border-radius: var(--radius-md);
          cursor: pointer;
          transition: all var(--transition-fast);
          font-weight: 500;
        }

        .profit-toggle:hover {
          border-color: var(--border-light);
        }

        .profit-toggle.profit {
          border-color: var(--success);
          background-color: var(--success-light);
          color: var(--success);
        }

        .profit-toggle.loss {
          border-color: var(--danger);
          background-color: var(--danger-light);
          color: var(--danger);
        }

        .profit-input {
          flex: 1;
          text-align: center;
          font-size: var(--font-size-lg);
        }

        .profit-display {
          font-size: var(--font-size-xl);
          font-weight: 700;
        }

        .report-actions {
          display: flex;
          gap: var(--space-2);
          justify-content: flex-end;
          margin-top: var(--space-4);
        }
      `}</style>
    </div>
  );
}
