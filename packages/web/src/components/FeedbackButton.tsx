import { createSignal, Show } from 'solid-js';
import type { Recommendation } from '../lib/types';

interface FeedbackButtonProps {
  recommendation: Recommendation;
  compact?: boolean;
}

export default function FeedbackButton(props: FeedbackButtonProps) {
  const [submitted, setSubmitted] = createSignal<'positive' | 'negative' | null>(null);
  const [showModal, setShowModal] = createSignal(false);
  const [pendingRating, setPendingRating] = createSignal<'positive' | 'negative' | null>(null);
  const [message, setMessage] = createSignal('');
  const [isSubmitting, setIsSubmitting] = createSignal(false);

  const handleRating = async (rating: 'positive' | 'negative') => {
    if (submitted()) return;

    // For negative feedback, show modal to collect more info
    if (rating === 'negative') {
      setPendingRating(rating);
      setShowModal(true);
      return;
    }

    // For positive, submit immediately
    await submitFeedback(rating, '');
  };

  const submitFeedback = async (rating: 'positive' | 'negative', feedbackMessage: string) => {
    setIsSubmitting(true);

    try {
      const response = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'recommendation',
          rating,
          message: feedbackMessage || undefined,
          recommendation: {
            id: props.recommendation.id,
            itemId: props.recommendation.itemId,
            item: props.recommendation.item,
            buyPrice: props.recommendation.buyPrice,
            sellPrice: props.recommendation.sellPrice,
            quantity: props.recommendation.quantity,
            expectedProfit: props.recommendation.expectedProfit,
            confidence: props.recommendation.confidence,
            modelId: props.recommendation.modelId
          }
        })
      });

      if (response.ok) {
        setSubmitted(rating);
        setShowModal(false);
      }
    } catch {
      // Silently fail - feedback is non-critical
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleModalSubmit = async () => {
    const rating = pendingRating();
    if (!rating) return;
    await submitFeedback(rating, message());
  };

  return (
    <>
      <div class={`feedback-buttons ${props.compact ? 'compact' : ''}`}>
        <Show when={!submitted()}>
          <button
            class={`feedback-btn feedback-up`}
            onClick={() => handleRating('positive')}
            title="Good recommendation"
            aria-label="Good recommendation"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
            </svg>
          </button>
          <button
            class={`feedback-btn feedback-down`}
            onClick={() => handleRating('negative')}
            title="Bad recommendation"
            aria-label="Bad recommendation"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
            </svg>
          </button>
        </Show>
        <Show when={submitted()}>
          <span class={`feedback-submitted ${submitted()}`}>
            {submitted() === 'positive' ? '👍' : '👎'} Thanks!
          </span>
        </Show>
      </div>

      <Show when={showModal()}>
        <div class="modal-overlay" onClick={() => setShowModal(false)}>
          <div class="modal feedback-modal" onClick={(e) => e.stopPropagation()}>
            <div class="modal-header">
              <h3>What was wrong with this recommendation?</h3>
              <button class="modal-close" onClick={() => setShowModal(false)}>×</button>
            </div>
            <div class="modal-body">
              <p class="feedback-item-name">{props.recommendation.item}</p>
              <textarea
                class="feedback-textarea"
                placeholder="Price was off, didn't fill, bad timing, etc. (optional)"
                value={message()}
                onInput={(e) => setMessage(e.currentTarget.value)}
                rows={3}
              />
            </div>
            <div class="modal-footer">
              <button class="btn btn-ghost" onClick={() => setShowModal(false)}>
                Cancel
              </button>
              <button
                class="btn btn-primary"
                onClick={handleModalSubmit}
                disabled={isSubmitting()}
              >
                {isSubmitting() ? 'Submitting...' : 'Submit Feedback'}
              </button>
            </div>
          </div>
        </div>
      </Show>

      <style>{`
        .feedback-buttons {
          display: flex;
          gap: var(--space-1);
          align-items: center;
        }

        .feedback-buttons.compact {
          gap: 2px;
        }

        .feedback-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 28px;
          height: 28px;
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-sm);
          background: var(--glass-bg);
          color: var(--text-muted);
          cursor: pointer;
          transition: all var(--transition-fast);
          backdrop-filter: blur(8px);
        }

        .feedback-btn:hover {
          border-color: var(--accent);
        }

        .feedback-up:hover {
          color: var(--success);
          border-color: var(--success);
          background: rgba(var(--success-rgb, 139, 195, 74), 0.1);
        }

        .feedback-down:hover {
          color: var(--danger);
          border-color: var(--danger);
          background: rgba(var(--danger-rgb, 239, 83, 80), 0.1);
        }

        .feedback-submitted {
          font-size: var(--font-size-sm);
          color: var(--text-muted);
          padding: var(--space-1) var(--space-2);
        }

        .feedback-submitted.positive {
          color: var(--success);
        }

        .feedback-submitted.negative {
          color: var(--danger);
        }

        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(8, 10, 15, 0.7);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: var(--space-4);
        }

        .feedback-modal {
          background: linear-gradient(145deg, var(--surface-2), var(--surface-1));
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-lg);
          max-width: 400px;
          width: 100%;
          box-shadow: var(--shadow-lg);
          backdrop-filter: blur(16px);
        }

        .modal-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: var(--space-4);
          border-bottom: 1px solid var(--border);
        }

        .modal-header h3 {
          font-size: var(--font-size-base);
          margin: 0;
        }

        .modal-close {
          background: none;
          border: none;
          font-size: 24px;
          color: var(--text-muted);
          cursor: pointer;
          line-height: 1;
        }

        .modal-body {
          padding: var(--space-4);
        }

        .feedback-item-name {
          font-weight: 600;
          margin-bottom: var(--space-3);
          color: var(--gold);
        }

        .feedback-textarea {
          width: 100%;
          padding: var(--space-3);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-md);
          background: var(--glass-bg);
          color: var(--text-primary);
          font-family: inherit;
          font-size: var(--font-size-sm);
          resize: vertical;
        }

        .feedback-textarea:focus {
          outline: none;
          border-color: var(--accent);
        }

        .modal-footer {
          display: flex;
          justify-content: flex-end;
          gap: var(--space-2);
          padding: var(--space-4);
          border-top: 1px solid var(--border);
        }
      `}</style>
    </>
  );
}
