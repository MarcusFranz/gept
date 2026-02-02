import { createSignal, Show } from 'solid-js';
import type { FeedbackType } from '../lib/types';

export default function FeedbackWidget() {
  const [isOpen, setIsOpen] = createSignal(false);
  const [type, setType] = createSignal<Exclude<FeedbackType, 'recommendation'>>('general');
  const [message, setMessage] = createSignal('');
  const [email, setEmail] = createSignal('');
  const [isSubmitting, setIsSubmitting] = createSignal(false);
  const [submitted, setSubmitted] = createSignal(false);
  const [error, setError] = createSignal('');

  const typeLabels: Record<Exclude<FeedbackType, 'recommendation'>, string> = {
    bug: 'Bug Report',
    feature: 'Feature Request',
    general: 'General Feedback'
  };

  const handleSubmit = async (e: Event) => {
    e.preventDefault();
    setError('');

    if (!message().trim()) {
      setError('Please enter your feedback');
      return;
    }

    setIsSubmitting(true);

    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: type(),
          message: message().trim(),
          email: email().trim() || undefined
        })
      });

      const data = await response.json();

      if (data.success) {
        setSubmitted(true);
        setTimeout(() => {
          setIsOpen(false);
          setSubmitted(false);
          setMessage('');
          setEmail('');
          setType('general');
        }, 2000);
      } else {
        setError(data.error || 'Failed to submit feedback');
      }
    } catch {
      setError('Failed to submit feedback. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    setIsOpen(false);
    setError('');
  };

  return (
    <>
      <button
        class="feedback-trigger"
        onClick={() => setIsOpen(true)}
        title="Send Feedback"
        aria-label="Send Feedback"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        </svg>
        <span>Feedback</span>
      </button>

      <Show when={isOpen()}>
        <div class="feedback-overlay" onClick={handleClose}>
          <div class="feedback-panel" onClick={(e) => e.stopPropagation()}>
            <Show when={!submitted()} fallback={
              <div class="feedback-success">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                  <polyline points="22 4 12 14.01 9 11.01" />
                </svg>
                <h3>Thank you!</h3>
                <p>Your feedback helps us improve GePT.</p>
              </div>
            }>
              <div class="feedback-header">
                <h3>Send Feedback</h3>
                <button class="feedback-close" onClick={handleClose}>×</button>
              </div>

              <form class="feedback-form" onSubmit={handleSubmit}>
                <div class="feedback-types">
                  {(['bug', 'feature', 'general'] as const).map((t) => (
                    <button
                      type="button"
                      class={`feedback-type-btn ${type() === t ? 'active' : ''}`}
                      onClick={() => setType(t)}
                    >
                      {t === 'bug' && '🐛'}
                      {t === 'feature' && '💡'}
                      {t === 'general' && '💬'}
                      <span>{typeLabels[t]}</span>
                    </button>
                  ))}
                </div>

                <div class="feedback-field">
                  <label for="feedback-message">Your feedback</label>
                  <textarea
                    id="feedback-message"
                    placeholder={
                      type() === 'bug'
                        ? "Describe the bug you encountered..."
                        : type() === 'feature'
                        ? "Describe the feature you'd like to see..."
                        : "Share your thoughts with us..."
                    }
                    value={message()}
                    onInput={(e) => setMessage(e.currentTarget.value)}
                    rows={4}
                    required
                  />
                </div>

                <div class="feedback-field">
                  <label for="feedback-email">Email (optional)</label>
                  <input
                    type="email"
                    id="feedback-email"
                    placeholder="For follow-up questions"
                    value={email()}
                    onInput={(e) => setEmail(e.currentTarget.value)}
                  />
                </div>

                <Show when={error()}>
                  <p class="feedback-error">{error()}</p>
                </Show>

                <button
                  type="submit"
                  class="btn btn-primary feedback-submit"
                  disabled={isSubmitting()}
                >
                  {isSubmitting() ? 'Sending...' : 'Send Feedback'}
                </button>
              </form>
            </Show>
          </div>
        </div>
      </Show>

      <style>{`
        .feedback-trigger {
          position: fixed;
          bottom: var(--space-4);
          right: var(--space-4);
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-3) var(--space-4);
          background: linear-gradient(135deg, var(--accent) 0%, var(--action) 100%);
          color: #0b0d12;
          border: none;
          border-radius: var(--radius-full);
          font-weight: 600;
          font-size: var(--font-size-sm);
          cursor: pointer;
          box-shadow: var(--shadow-lg);
          transition: all var(--transition-fast);
          z-index: 100;
        }

        .feedback-trigger:hover {
          transform: translateY(-2px);
          box-shadow: 0 12px 28px rgba(126, 231, 135, 0.3);
        }

        .feedback-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(8, 10, 15, 0.7);
          display: flex;
          align-items: flex-end;
          justify-content: flex-end;
          padding: var(--space-4);
          z-index: 1000;
        }

        .feedback-panel {
          background: linear-gradient(145deg, var(--surface-2), var(--surface-1));
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-lg);
          width: 360px;
          max-width: 100%;
          max-height: calc(100vh - 2rem);
          overflow-y: auto;
          box-shadow: var(--shadow-lg);
          animation: slideUp 0.2s ease-out;
          backdrop-filter: blur(16px);
        }

        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .feedback-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: var(--space-4);
          border-bottom: 1px solid var(--border);
        }

        .feedback-header h3 {
          font-size: var(--font-size-lg);
          margin: 0;
        }

        .feedback-close {
          background: none;
          border: none;
          font-size: 24px;
          color: var(--text-muted);
          cursor: pointer;
          line-height: 1;
          padding: var(--space-1);
        }

        .feedback-form {
          padding: var(--space-4);
        }

        .feedback-types {
          display: flex;
          gap: var(--space-2);
          margin-bottom: var(--space-4);
        }

        .feedback-type-btn {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: var(--space-1);
          padding: var(--space-3);
          background: var(--glass-bg);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-md);
          color: var(--text-secondary);
          font-size: var(--font-size-xs);
          cursor: pointer;
          transition: all var(--transition-fast);
          backdrop-filter: blur(10px);
        }

        .feedback-type-btn:hover {
          border-color: var(--accent);
          color: var(--text-primary);
        }

        .feedback-type-btn.active {
          border-color: var(--accent);
          background: rgba(126, 231, 135, 0.12);
          color: var(--text-primary);
        }

        .feedback-field {
          margin-bottom: var(--space-4);
        }

        .feedback-field label {
          display: block;
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
          margin-bottom: var(--space-2);
        }

        .feedback-field textarea,
        .feedback-field input {
          width: 100%;
          padding: var(--space-3);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-md);
          background: var(--glass-bg);
          color: var(--text-primary);
          font-family: inherit;
          font-size: var(--font-size-sm);
        }

        .feedback-field textarea {
          resize: vertical;
          min-height: 100px;
        }

        .feedback-field textarea:focus,
        .feedback-field input:focus {
          outline: none;
          border-color: var(--accent);
        }

        .feedback-error {
          color: var(--danger);
          font-size: var(--font-size-sm);
          margin-bottom: var(--space-3);
        }

        .feedback-submit {
          width: 100%;
        }

        .feedback-success {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          padding: var(--space-8);
        }

        .feedback-success h3 {
          margin: var(--space-4) 0 var(--space-2);
          color: var(--success);
        }

        .feedback-success p {
          color: var(--text-secondary);
          margin: 0;
        }

        @media (max-width: 480px) {
          .feedback-trigger span {
            display: none;
          }

          .feedback-trigger {
            padding: var(--space-3);
            border-radius: 50%;
          }

          .feedback-overlay {
            align-items: flex-end;
            justify-content: center;
            padding: 0;
          }

          .feedback-panel {
            width: 100%;
            border-radius: var(--radius-lg) var(--radius-lg) 0 0;
            max-height: 85vh;
          }
        }
      `}</style>
    </>
  );
}
