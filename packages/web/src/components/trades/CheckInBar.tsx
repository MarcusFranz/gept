// packages/web/src/components/trades/CheckInBar.tsx
import { Show, createSignal } from 'solid-js';

interface CheckInBarProps {
  phase: 'buying' | 'selling';
  quantity: number;
  onMarkFilled: () => void;
  onMarkPartiallyFilled: (filledQuantity: number) => void;
  onMarkSold: () => void;
  disabled?: boolean;
}

export function CheckInBar(props: CheckInBarProps) {
  const [partialMode, setPartialMode] = createSignal(false);
  const [partialQuantityInput, setPartialQuantityInput] = createSignal(String(props.quantity));
  const [error, setError] = createSignal<string | null>(null);

  const submitPartial = () => {
    const qty = Number(partialQuantityInput());
    if (!Number.isFinite(qty)) {
      setError('Enter a valid amount.');
      return;
    }
    const normalized = Math.floor(qty);
    if (normalized <= 0) {
      setError('Amount must be at least 1.');
      return;
    }
    if (normalized > props.quantity) {
      setError(`Amount cannot be more than ${props.quantity.toLocaleString()}.`);
      return;
    }
    setError(null);
    props.onMarkPartiallyFilled(normalized);
  };

  return (
    <div class="check-in-bar-container">
      <p class="check-in-prompt">
        {props.phase === 'buying'
          ? 'Did your buy order fill?'
          : 'Did your sell order complete?'}
      </p>

      <Show
        when={props.phase === 'buying'}
        fallback={
          <button
            class="check-in-action-btn check-in-primary"
            onClick={() => props.onMarkSold()}
            disabled={props.disabled}
          >
            Order sold
          </button>
        }
      >
        <Show
          when={!partialMode()}
          fallback={
            <div class="check-in-partial">
              <label class="check-in-input-label" for="partial-fill-qty">
                Amount bought (max {props.quantity.toLocaleString()})
              </label>
              <input
                id="partial-fill-qty"
                class="check-in-input"
                type="number"
                min="1"
                max={props.quantity}
                inputmode="numeric"
                value={partialQuantityInput()}
                disabled={props.disabled}
                onInput={(e) => {
                  setPartialQuantityInput(e.currentTarget.value);
                  if (error()) setError(null);
                }}
              />
              <Show when={error()}>
                <p class="check-in-error">{error()}</p>
              </Show>
              <div class="check-in-actions-inline">
                <button
                  class="check-in-action-btn check-in-secondary"
                  onClick={() => {
                    setPartialMode(false);
                    setError(null);
                  }}
                  disabled={props.disabled}
                >
                  Back
                </button>
                <button
                  class="check-in-action-btn check-in-primary"
                  onClick={submitPartial}
                  disabled={props.disabled}
                >
                  Save amount & switch to selling
                </button>
              </div>
            </div>
          }
        >
          <div class="check-in-actions">
            <button
              class="check-in-action-btn check-in-primary"
              onClick={() => props.onMarkFilled()}
              disabled={props.disabled}
            >
              Order filled
            </button>
            <button
              class="check-in-action-btn check-in-secondary"
              onClick={() => setPartialMode(true)}
              disabled={props.disabled}
            >
              Order partially filled
            </button>
          </div>
        </Show>
      </Show>

      <style>{`
        .check-in-bar-container {
          padding: 1rem;
          background: var(--surface-2);
          border: 1px solid var(--border);
          border-radius: var(--radius-xl);
          box-shadow: none;
        }

        .check-in-prompt {
          margin: 0 0 0.75rem 0;
          color: var(--text-secondary);
          font-size: var(--font-size-sm);
        }

        .check-in-actions {
          display: grid;
          gap: 0.65rem;
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .check-in-actions-inline {
          display: grid;
          gap: 0.65rem;
          grid-template-columns: minmax(120px, auto) minmax(0, 1fr);
        }

        .check-in-action-btn {
          padding: 0.68rem 1rem;
          border-radius: var(--radius-full);
          font-weight: 600;
          cursor: pointer;
          transition: transform 0.22s ease, background 0.22s ease, border-color 0.22s ease;
        }

        .check-in-action-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
          transform: none;
        }

        .check-in-action-btn:hover:not(:disabled) {
          transform: translateY(-1px);
        }

        .check-in-primary {
          background: var(--action);
          color: var(--btn-text-dark);
          border: 1px solid color-mix(in srgb, var(--action) 70%, #000);
        }

        .check-in-primary:hover:not(:disabled) {
          background: var(--action-hover);
        }

        .check-in-secondary {
          background: var(--surface-2);
          color: var(--text-primary);
          border: 1px solid var(--border);
        }

        .check-in-secondary:hover:not(:disabled) {
          border-color: var(--border-light);
          background: color-mix(in srgb, var(--surface-2) 85%, #fff 15%);
        }

        .check-in-partial {
          display: grid;
          gap: 0.7rem;
        }

        .check-in-input-label {
          font-size: var(--font-size-xs);
          color: var(--text-muted);
        }

        .check-in-input {
          width: 100%;
          border-radius: var(--radius-lg);
          border: 1px solid var(--border);
          background: var(--surface-1);
          color: var(--text-primary);
          padding: 0.6rem 0.7rem;
          font-size: var(--font-size-sm);
        }

        .check-in-input:focus {
          outline: none;
          border-color: var(--border-light);
        }

        .check-in-error {
          margin: 0;
          color: var(--danger);
          font-size: var(--font-size-xs);
        }

        @media (max-width: 640px) {
          .check-in-actions {
            grid-template-columns: 1fr;
          }

          .check-in-actions-inline {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
