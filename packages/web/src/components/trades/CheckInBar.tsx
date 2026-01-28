// packages/web/src/components/trades/CheckInBar.tsx
import { createSignal, createEffect } from 'solid-js';

interface CheckInBarProps {
  progress: number;
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
      <p class="check-in-prompt">How's your order?</p>

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
            class="check-in-bar-fill"
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
        <span class="check-in-hint">Tap the bar where you are</span>
        <span>100%</span>
      </div>

      <style>{`
        .check-in-bar-container {
          padding: 1rem;
          background: var(--bg-secondary);
          border-radius: var(--radius-lg);
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
          background: var(--bg-tertiary);
          border-radius: 12px;
          cursor: pointer;
          position: relative;
          overflow: visible;
        }

        .check-in-bar-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--accent), var(--accent-hover));
          border-radius: 12px;
          transition: width 0.15s ease-out;
        }

        .check-in-bar-handle {
          position: absolute;
          top: 50%;
          transform: translate(-50%, -50%);
          width: 16px;
          height: 16px;
          background: white;
          border-radius: 50%;
          box-shadow: 0 2px 4px rgba(0,0,0,0.3);
          transition: left 0.15s ease-out;
        }

        .check-in-done-btn {
          padding: 0.5rem 1rem;
          background: var(--success);
          color: var(--btn-text-dark);
          border: none;
          border-radius: var(--radius-md);
          font-weight: 600;
          cursor: pointer;
          white-space: nowrap;
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
