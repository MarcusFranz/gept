import { createSignal, createEffect, Show, For } from 'solid-js';
import type { Recommendation } from '../lib/types';
import FlipCard from './FlipCard';

interface FlipCarouselProps {
  userId: string;
}

export default function FlipCarousel(props: FlipCarouselProps) {
  const [recommendations, setRecommendations] = createSignal<Recommendation[]>([]);
  const [currentIndex, setCurrentIndex] = createSignal(0);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [exhausted, setExhausted] = createSignal(false);
  const [tracking, setTracking] = createSignal(false);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/recommendations');
      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to fetch recommendations');
      }

      setRecommendations(data.data || []);
      setExhausted(data.exhausted || false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  createEffect(() => {
    fetchRecommendations();
  });

  const currentRec = () => recommendations()[currentIndex()];
  const hasNext = () => currentIndex() < recommendations().length - 1;
  const hasPrev = () => currentIndex() > 0;

  const next = () => {
    if (hasNext()) {
      setCurrentIndex(i => i + 1);
    }
  };

  const prev = () => {
    if (hasPrev()) {
      setCurrentIndex(i => i - 1);
    }
  };

  const trackTrade = async () => {
    const rec = currentRec();
    if (!rec) return;

    setTracking(true);
    try {
      const response = await fetch('/api/trades/active', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          itemId: rec.itemId,
          itemName: rec.item,
          buyPrice: rec.buyPrice,
          sellPrice: rec.sellPrice,
          quantity: rec.quantity,
          recId: rec.id,
          modelId: rec.modelId,
          expectedHours: rec.expectedHours,
          confidence: rec.confidence,
          fillProbability: rec.fillProbability,
          expectedProfit: rec.expectedProfit
        })
      });

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error);
      }

      // Move to next recommendation after tracking
      if (hasNext()) {
        next();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to track trade');
    } finally {
      setTracking(false);
    }
  };

  return (
    <div class="carousel">
      <Show when={loading()}>
        <div class="carousel-loading">
          <div class="spinner"></div>
          <p class="text-secondary mt-3">Loading recommendations...</p>
        </div>
      </Show>

      <Show when={error()}>
        <div class="carousel-error card">
          <div class="text-center">
            <p class="text-danger mb-3">{error()}</p>
            <button class="btn btn-primary" onClick={fetchRecommendations}>
              Try Again
            </button>
          </div>
        </div>
      </Show>

      <Show when={!loading() && !error() && recommendations().length === 0}>
        <div class="empty-state card">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <path d="M12 6v6l4 2"/>
          </svg>
          <p>No recommendations available right now.</p>
          <button class="btn btn-primary mt-3" onClick={fetchRecommendations}>
            Refresh
          </button>
        </div>
      </Show>

      <Show when={!loading() && !error() && currentRec()}>
        <FlipCard
          recommendation={currentRec()!}
          onTrack={trackTrade}
          isTracking={tracking()}
        />

        <div class="carousel-nav">
          <button
            class="btn btn-secondary btn-icon"
            onClick={prev}
            disabled={!hasPrev()}
            aria-label="Previous"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
              <path d="M15 18l-6-6 6-6"/>
            </svg>
          </button>

          <div class="carousel-info">
            <span class="text-secondary">
              {currentIndex() + 1} of {recommendations().length}
              <Show when={exhausted()}>
                <span class="text-muted ml-2">(no more available)</span>
              </Show>
            </span>
          </div>

          <button
            class="btn btn-secondary btn-icon"
            onClick={next}
            disabled={!hasNext()}
            aria-label="Next"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
              <path d="M9 18l6-6-6-6"/>
            </svg>
          </button>
        </div>

        <div class="carousel-dots">
          <For each={recommendations()}>
            {(_, i) => (
              <button
                class={`carousel-dot ${i() === currentIndex() ? 'active' : ''}`}
                onClick={() => setCurrentIndex(i())}
                aria-label={`Go to recommendation ${i() + 1}`}
              />
            )}
          </For>
        </div>
      </Show>

      <style>{`
        .carousel {
          max-width: 480px;
          margin: 0 auto;
        }

        .carousel-loading {
          text-align: center;
          padding: var(--space-6);
        }

        .carousel-loading .spinner {
          margin: 0 auto;
        }

        .carousel-error {
          padding: var(--space-6);
        }

        .carousel-nav {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-top: var(--space-4);
        }

        .carousel-info {
          font-size: var(--font-size-sm);
        }

        .carousel-dots {
          display: flex;
          justify-content: center;
          gap: var(--space-2);
          margin-top: var(--space-3);
        }

        .carousel-dot {
          width: 8px;
          height: 8px;
          padding: 0;
          border: none;
          border-radius: 50%;
          background-color: rgba(255, 255, 255, 0.12);
          cursor: pointer;
          transition: background-color var(--transition-fast), transform var(--transition-fast);
        }

        .carousel-dot:hover {
          background-color: var(--border-light);
        }

        .carousel-dot.active {
          background-color: var(--accent);
          transform: scale(1.25);
        }
      `}</style>
    </div>
  );
}
