import { createSignal, createEffect, Show, For } from 'solid-js';
import { capitalPresets } from '../lib/types';

interface OnboardingProps {
  onComplete: (capital: number) => void;
  onSkip: () => void;
}

type Step = 'welcome' | 'capital' | 'recommendation' | 'slots' | 'ready';

const STEPS: Step[] = ['welcome', 'capital', 'recommendation', 'slots', 'ready'];

export default function Onboarding(props: OnboardingProps) {
  const [currentStep, setCurrentStep] = createSignal<Step>('welcome');
  const [selectedCapital, setSelectedCapital] = createSignal(20_000_000);
  const [customCapital, setCustomCapital] = createSignal('');

  const stepIndex = () => STEPS.indexOf(currentStep());

  const next = () => {
    const idx = stepIndex();
    if (idx < STEPS.length - 1) {
      setCurrentStep(STEPS[idx + 1]);
    }
  };

  const prev = () => {
    const idx = stepIndex();
    if (idx > 0) {
      setCurrentStep(STEPS[idx - 1]);
    }
  };

  const complete = () => {
    const capital = customCapital()
      ? parseInt(customCapital().replace(/[^0-9]/g, ''), 10) * 1_000_000
      : selectedCapital();
    props.onComplete(capital);
  };

  const formatCapital = (value: number) => {
    if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(1)}B`;
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(0)}M`;
    return value.toLocaleString();
  };

  return (
    <div class="onboarding-overlay">
      <div class="onboarding-modal">
        <button class="onboarding-skip" onClick={props.onSkip}>
          Skip tutorial
        </button>

        <div class="onboarding-progress">
          <For each={STEPS}>
            {(step, i) => (
              <div
                class={`progress-dot ${i() <= stepIndex() ? 'active' : ''} ${i() === stepIndex() ? 'current' : ''}`}
              />
            )}
          </For>
        </div>

        {/* Welcome Step */}
        <Show when={currentStep() === 'welcome'}>
          <div class="onboarding-step">
            <div class="step-icon">
              <img src="/images/logo-icon.png" alt="GePT" />
            </div>
            <h2>Welcome to GePT!</h2>
            <p>
              Let's set up your AI-powered Grand Exchange flipping assistant.
              This will only take a minute.
            </p>
            <div class="step-actions">
              <button class="btn btn-primary btn-lg" onClick={next}>
                Get Started
              </button>
            </div>
          </div>
        </Show>

        {/* Capital Step */}
        <Show when={currentStep() === 'capital'}>
          <div class="onboarding-step">
            <div class="step-icon capital-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"/>
                <path d="M12 6v12M6 12h12"/>
              </svg>
            </div>
            <h2>How much GP do you have?</h2>
            <p>
              Select your starting capital. This helps us find flips that fit your budget.
            </p>

            <div class="capital-presets">
              <For each={capitalPresets}>
                {(preset) => (
                  <button
                    class={`capital-preset ${selectedCapital() === preset.value ? 'active' : ''}`}
                    onClick={() => {
                      setSelectedCapital(preset.value);
                      setCustomCapital('');
                    }}
                  >
                    {preset.label}
                  </button>
                )}
              </For>
            </div>

            <div class="capital-custom">
              <label>Or enter a custom amount (in millions):</label>
              <input
                type="text"
                placeholder="e.g., 75"
                value={customCapital()}
                onInput={(e) => {
                  setCustomCapital(e.currentTarget.value);
                  setSelectedCapital(0);
                }}
              />
            </div>

            <p class="step-note">
              Selected: <strong>{formatCapital(customCapital() ? parseInt(customCapital().replace(/[^0-9]/g, ''), 10) * 1_000_000 || 0 : selectedCapital())} GP</strong>
            </p>

            <div class="step-actions">
              <button class="btn btn-ghost" onClick={prev}>Back</button>
              <button class="btn btn-primary" onClick={next}>Continue</button>
            </div>
          </div>
        </Show>

        {/* Recommendation Step */}
        <Show when={currentStep() === 'recommendation'}>
          <div class="onboarding-step">
            <div class="step-icon rec-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M9 12l2 2 4-4"/>
                <rect x="3" y="3" width="18" height="18" rx="2"/>
              </svg>
            </div>
            <h2>Understanding Recommendations</h2>
            <p>
              Each recommendation card shows you a potential flip:
            </p>

            <div class="rec-explainer">
              <div class="rec-item">
                <span class="rec-label">Buy / Sell Price</span>
                <span class="rec-desc">The recommended prices to offer</span>
              </div>
              <div class="rec-item">
                <span class="rec-label">Profit</span>
                <span class="rec-desc">Expected profit if the flip completes</span>
              </div>
              <div class="rec-item">
                <span class="rec-label">Confidence</span>
                <span class="rec-desc">How confident the AI is in this recommendation</span>
              </div>
              <div class="rec-item">
                <span class="rec-label">Fill Probability</span>
                <span class="rec-desc">Likelihood your order will fill at these prices</span>
              </div>
            </div>

            <div class="step-actions">
              <button class="btn btn-ghost" onClick={prev}>Back</button>
              <button class="btn btn-primary" onClick={next}>Continue</button>
            </div>
          </div>
        </Show>

        {/* Slots Step */}
        <Show when={currentStep() === 'slots'}>
          <div class="onboarding-step">
            <div class="step-icon slots-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="7" height="7"/>
                <rect x="14" y="3" width="7" height="7"/>
                <rect x="14" y="14" width="7" height="7"/>
                <rect x="3" y="14" width="7" height="7"/>
              </svg>
            </div>
            <h2>8 GE Slots</h2>
            <p>
              Just like the Grand Exchange, you have 8 slots for active trades.
            </p>

            <div class="slots-info">
              <div class="slots-visual">
                <For each={[1, 2, 3, 4, 5, 6, 7, 8]}>
                  {(slot) => (
                    <div class={`slot-box ${slot <= 3 ? 'active' : ''}`}>
                      {slot}
                    </div>
                  )}
                </For>
              </div>
              <p class="slots-desc">
                Track your active trades and see profit/loss for each slot.
                GePT will recommend items to fill your empty slots.
              </p>
            </div>

            <div class="step-actions">
              <button class="btn btn-ghost" onClick={prev}>Back</button>
              <button class="btn btn-primary" onClick={next}>Continue</button>
            </div>
          </div>
        </Show>

        {/* Ready Step */}
        <Show when={currentStep() === 'ready'}>
          <div class="onboarding-step">
            <div class="step-icon ready-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                <polyline points="22 4 12 14.01 9 11.01"/>
              </svg>
            </div>
            <h2>You're all set!</h2>
            <p>
              Start flipping smarter with AI-powered recommendations.
              Good luck on your flipping journey!
            </p>

            <div class="ready-tips">
              <div class="tip">
                <strong>Tip:</strong> Use the thumbs up/down buttons to give feedback on recommendations.
              </div>
            </div>

            <div class="step-actions single">
              <button class="btn btn-primary btn-lg" onClick={complete}>
                Start Flipping
              </button>
            </div>
          </div>
        </Show>
      </div>

      <style>{`
        .onboarding-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.85);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 2000;
          padding: var(--space-4);
        }

        .onboarding-modal {
          background: var(--bg-primary);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          max-width: 480px;
          width: 100%;
          padding: var(--space-6);
          position: relative;
        }

        .onboarding-skip {
          position: absolute;
          top: var(--space-4);
          right: var(--space-4);
          background: none;
          border: none;
          color: var(--text-muted);
          font-size: var(--font-size-sm);
          cursor: pointer;
        }

        .onboarding-skip:hover {
          color: var(--text-primary);
        }

        .onboarding-progress {
          display: flex;
          justify-content: center;
          gap: var(--space-2);
          margin-bottom: var(--space-5);
        }

        .progress-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--border);
          transition: all var(--transition-fast);
        }

        .progress-dot.active {
          background: var(--accent);
        }

        .progress-dot.current {
          width: 24px;
          border-radius: 4px;
        }

        .onboarding-step {
          text-align: center;
        }

        .step-icon {
          width: 64px;
          height: 64px;
          margin: 0 auto var(--space-4);
        }

        .step-icon img {
          width: 100%;
          height: 100%;
        }

        .step-icon svg {
          width: 100%;
          height: 100%;
          color: var(--accent);
        }

        .onboarding-step h2 {
          font-family: 'Rajdhani', sans-serif;
          font-size: var(--font-size-2xl);
          font-weight: 700;
          margin-bottom: var(--space-3);
          color: var(--text-primary);
        }

        .onboarding-step > p {
          color: var(--text-secondary);
          line-height: 1.6;
          margin-bottom: var(--space-5);
        }

        .step-actions {
          display: flex;
          gap: var(--space-3);
          justify-content: center;
          margin-top: var(--space-5);
        }

        .step-actions.single {
          margin-top: var(--space-6);
        }

        .btn {
          padding: var(--space-2) var(--space-4);
          border-radius: var(--radius-md);
          font-weight: 600;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .btn-lg {
          padding: var(--space-3) var(--space-5);
        }

        .btn-primary {
          background: var(--accent);
          color: var(--bg-primary);
          border: none;
        }

        .btn-primary:hover {
          background: var(--accent-hover);
        }

        .btn-ghost {
          background: transparent;
          border: 1px solid var(--border);
          color: var(--text-secondary);
        }

        .btn-ghost:hover {
          background: var(--bg-hover);
          color: var(--text-primary);
        }

        /* Capital Step */
        .capital-presets {
          display: flex;
          flex-wrap: wrap;
          gap: var(--space-2);
          justify-content: center;
          margin-bottom: var(--space-4);
        }

        .capital-preset {
          padding: var(--space-2) var(--space-4);
          background: var(--bg-secondary);
          border: 2px solid var(--border);
          border-radius: var(--radius-md);
          color: var(--text-primary);
          font-weight: 600;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .capital-preset:hover {
          border-color: var(--accent);
        }

        .capital-preset.active {
          border-color: var(--accent);
          background: var(--accent-light);
          color: var(--accent);
        }

        .capital-custom {
          margin-bottom: var(--space-3);
        }

        .capital-custom label {
          display: block;
          font-size: var(--font-size-sm);
          color: var(--text-muted);
          margin-bottom: var(--space-2);
        }

        .capital-custom input {
          width: 120px;
          padding: var(--space-2) var(--space-3);
          border: 1px solid var(--border);
          border-radius: var(--radius-md);
          background: var(--bg-secondary);
          color: var(--text-primary);
          text-align: center;
          font-size: var(--font-size-base);
        }

        .capital-custom input:focus {
          outline: none;
          border-color: var(--accent);
        }

        .step-note {
          font-size: var(--font-size-sm);
          color: var(--text-muted);
        }

        .step-note strong {
          color: var(--gold);
        }

        /* Recommendation Step */
        .rec-explainer {
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius-md);
          padding: var(--space-4);
          text-align: left;
        }

        .rec-item {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: var(--space-3);
          padding: var(--space-2) 0;
          border-bottom: 1px solid var(--border);
        }

        .rec-item:last-child {
          border-bottom: none;
        }

        .rec-label {
          font-weight: 600;
          color: var(--text-primary);
          font-size: var(--font-size-sm);
          flex-shrink: 0;
        }

        .rec-desc {
          color: var(--text-muted);
          font-size: var(--font-size-sm);
          text-align: right;
        }

        /* Slots Step */
        .slots-info {
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius-md);
          padding: var(--space-4);
        }

        .slots-visual {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: var(--space-2);
          margin-bottom: var(--space-3);
        }

        .slot-box {
          aspect-ratio: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--bg-tertiary);
          border: 1px solid var(--border);
          border-radius: var(--radius-sm);
          font-weight: 600;
          color: var(--text-muted);
        }

        .slot-box.active {
          background: var(--accent-light);
          border-color: var(--accent);
          color: var(--accent);
        }

        .slots-desc {
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
          margin: 0;
        }

        /* Ready Step */
        .ready-tips {
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius-md);
          padding: var(--space-4);
          text-align: left;
        }

        .tip {
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
        }

        .tip strong {
          color: var(--accent);
        }

        @media (max-width: 480px) {
          .onboarding-modal {
            padding: var(--space-4);
          }

          .rec-item {
            flex-direction: column;
            gap: var(--space-1);
          }
        }
      `}</style>
    </div>
  );
}
