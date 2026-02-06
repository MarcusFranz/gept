// packages/web/src/components/trades/CheckInBar.tsx
import { createSignal, createEffect } from 'solid-js';

interface CheckInBarProps {
  progress: number;
  phase: 'buying' | 'selling';
  onProgressChange: (progress: number) => void;
  onDone: () => void;
  disabled?: boolean;
}

export function CheckInBar(props: CheckInBarProps) {
  const [localProgress, setLocalProgress] = createSignal(props.progress);
  let barRef: HTMLInputElement | undefined;

  // Sync with prop changes
  createEffect(() => {
    setLocalProgress(props.progress);
  });

  const handleSliderInput = (e: Event) => {
    const target = e.currentTarget as HTMLInputElement;
    const value = Number(target.value);
    const clamped = Math.max(0, Math.min(100, value));
    setLocalProgress(clamped);
    props.onProgressChange(clamped);
  };

  return (
    <div class="check-in-bar-container">
      <p class="check-in-prompt">
        {props.phase === 'buying'
          ? 'How much of your buy offer has filled?'
          : 'How much of your sell offer has filled?'}
      </p>

      <div class="check-in-bar-wrapper">
        <div class={`check-in-slider ${props.phase === 'buying' ? 'is-buying' : 'is-selling'}`} style={{ '--progress': `${localProgress()}%` }}>
          <input
            ref={barRef}
            class="check-in-range"
            type="range"
            min="0"
            max="100"
            value={localProgress()}
            onInput={handleSliderInput}
            disabled={props.disabled}
            aria-label="Fill indicator"
          />
          <div class="check-in-track" aria-hidden="true">
            <div class="check-in-fill" />
            <div class="check-in-thumb" />
          </div>
          <div class="check-in-labels">
            <span>0%</span>
            <span class="check-in-hint">Drag to set fill</span>
            <span>100%</span>
          </div>
        </div>

        <button
          class="check-in-done-btn"
          onClick={() => props.onDone()}
          disabled={props.disabled}
        >
          Done ✓
        </button>
      </div>

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

        .check-in-bar-wrapper {
          display: flex;
          align-items: flex-start;
          gap: 1rem;
        }

        .check-in-slider {
          flex: 1;
          min-width: 0;
          position: relative;
          padding-top: 0.5rem;
        }

        .check-in-slider.is-buying {
          --fill-color: var(--phase-buy);
          --fill-glow: rgba(134, 216, 162, 0.45);
        }

        .check-in-slider.is-selling {
          --fill-color: var(--phase-sell);
          --fill-glow: rgba(240, 181, 96, 0.45);
        }

        .check-in-range {
          position: absolute;
          inset: 0;
          width: 100%;
          height: 42px;
          opacity: 0;
          cursor: pointer;
          z-index: 3;
        }

        .check-in-track {
          position: relative;
          height: 18px;
          border-radius: var(--radius-full);
          background: var(--surface-3);
          border: 1px solid var(--border);
          overflow: hidden;
          z-index: 1;
          pointer-events: none;
        }

        .check-in-fill {
          position: absolute;
          top: 0;
          left: 0;
          height: 100%;
          width: var(--progress);
          background: linear-gradient(90deg, color-mix(in srgb, var(--fill-color) 90%, #fff) 0%, var(--fill-color) 100%);
          transition: width 0.35s var(--ease-hero);
        }

        .check-in-thumb {
          position: absolute;
          top: 50%;
          left: var(--progress);
          width: 20px;
          height: 20px;
          border-radius: 50%;
          transform: translate(-50%, -50%);
          background: var(--surface-1);
          border: 2px solid var(--fill-color);
          box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
          transition: left 0.35s var(--ease-hero);
        }

        .check-in-done-btn {
          padding: 0.6rem 1.2rem;
          background: var(--action);
          color: var(--btn-text-dark);
          border: 1px solid color-mix(in srgb, var(--action) 70%, #000);
          border-radius: var(--radius-full);
          font-weight: 600;
          cursor: pointer;
          white-space: nowrap;
          transition: transform 0.4s var(--ease-hero), box-shadow 0.4s var(--ease-hero), background 0.3s ease;
        }

        .check-in-done-btn:hover:not(:disabled) {
          transform: translateY(-2px) scale(1.01);
          box-shadow: 0 18px 30px -20px rgba(168, 240, 8, 0.6);
        }

        .check-in-done-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .check-in-labels {
          display: flex;
          justify-content: space-between;
          margin-top: 0.55rem;
          font-size: var(--font-size-xs);
          color: var(--text-muted);
          pointer-events: none;
        }

        .check-in-hint {
          color: var(--text-secondary);
        }

        @media (max-width: 640px) {
          .check-in-bar-wrapper {
            flex-direction: column;
            align-items: stretch;
          }

          .check-in-done-btn {
            width: 100%;
          }
        }
      `}</style>
    </div>
  );
}
