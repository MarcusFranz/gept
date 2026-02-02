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
  let barRef: HTMLDivElement | undefined;

  // Sync with prop changes
  createEffect(() => {
    setLocalProgress(props.progress);
  });

  const handleBarClick = (e: MouseEvent) => {
    if (props.disabled || !barRef) return;

    const rect = barRef.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.round((x / rect.width) * 100);
    const clampedProgress = Math.max(0, Math.min(100, percentage));

    setLocalProgress(clampedProgress);
    props.onProgressChange(clampedProgress);
  };

  return (
    <div class="check-in-bar-container">
      <p class="check-in-prompt">
        {props.phase === 'buying'
          ? 'How much of your buy offer has filled?'
          : 'How much of your sell offer has filled?'}
      </p>

      <div class="check-in-bar-wrapper">
        <div
          ref={barRef}
          class="check-in-bar"
          onClick={handleBarClick}
          role="slider"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={localProgress()}
          tabIndex={0}
        >
          <div
            class={`check-in-bar-fill check-in-fill-${props.phase}`}
            style={{ width: `${localProgress()}%` }}
          />
          <div
            class="check-in-bar-handle"
            style={{ left: `${localProgress()}%` }}
          />
        </div>

        <button
          class="check-in-done-btn"
          onClick={() => props.onDone()}
          disabled={props.disabled}
        >
          Done ✓
        </button>
      </div>

      <div class="check-in-labels">
        <span>0%</span>
        <span class="check-in-hint">Tap the bar to set fill %</span>
        <span>100%</span>
      </div>

      <style>{`
        .check-in-bar-container {
          padding: 1rem;
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-lg);
          backdrop-filter: blur(10px);
        }

        .check-in-prompt {
          margin: 0 0 0.75rem 0;
          color: var(--text-secondary);
          font-size: var(--font-size-sm);
        }

        .check-in-bar-wrapper {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .check-in-bar {
          flex: 1;
          height: 24px;
          background: rgba(255, 255, 255, 0.08);
          border-radius: 12px;
          cursor: pointer;
          position: relative;
          overflow: visible;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .check-in-bar-fill {
          height: 100%;
          border-radius: 12px;
          transition: width 0.15s ease-out;
        }

        .check-in-fill-buying {
          background: var(--phase-buy);
        }

        .check-in-fill-selling {
          background: var(--phase-sell);
        }

        .check-in-bar-handle {
          position: absolute;
          top: 50%;
          transform: translate(-50%, -50%);
          width: 16px;
          height: 16px;
          background: #ffffff;
          border-radius: 50%;
          box-shadow: 0 6px 14px rgba(0, 0, 0, 0.35);
          transition: left 0.15s ease-out;
        }

        .check-in-done-btn {
          padding: 0.5rem 1rem;
          background: linear-gradient(135deg, var(--accent) 0%, var(--action) 100%);
          color: #0b0d12;
          border: none;
          border-radius: var(--radius-md);
          font-weight: 600;
          cursor: pointer;
          white-space: nowrap;
          box-shadow: 0 10px 22px rgba(126, 231, 135, 0.2);
        }

        .check-in-done-btn:hover:not(:disabled) {
          filter: brightness(1.1);
        }

        .check-in-done-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .check-in-labels {
          display: flex;
          justify-content: space-between;
          margin-top: 0.5rem;
          font-size: var(--font-size-xs);
          color: var(--text-muted);
        }

        .check-in-hint {
          color: var(--text-secondary);
        }
      `}</style>
    </div>
  );
}
