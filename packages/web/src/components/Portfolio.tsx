import { createSignal, createEffect, Show } from 'solid-js';
import type { PortfolioStats } from '../lib/types';
import { formatGold, formatPercent } from '../lib/types';

type Timeframe = 'today' | 'week' | 'month' | 'all';

export default function Portfolio() {
  const [stats, setStats] = createSignal<PortfolioStats | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [error, setError] = createSignal<string | null>(null);
  const [timeframe, setTimeframe] = createSignal<Timeframe>('all');

  const fetchStats = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`/api/portfolio?timeframe=${timeframe()}`);
      const data = await response.json();

      if (data.success) {
        setStats(data.data);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('Failed to load portfolio stats');
    } finally {
      setLoading(false);
    }
  };

  createEffect(() => {
    const _ = timeframe();
    fetchStats();
  });

  const timeframeLabel = () => {
    switch (timeframe()) {
      case 'today': return 'Today';
      case 'week': return 'This Week';
      case 'month': return 'This Month';
      default: return 'All Time';
    }
  };

  return (
    <div class="portfolio">
      <div class="portfolio-header">
        <h2>Portfolio</h2>
        <div class="timeframe-selector">
          <button
            class={`btn ${timeframe() === 'today' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
            onClick={() => setTimeframe('today')}
          >
            Today
          </button>
          <button
            class={`btn ${timeframe() === 'week' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
            onClick={() => setTimeframe('week')}
          >
            Week
          </button>
          <button
            class={`btn ${timeframe() === 'month' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
            onClick={() => setTimeframe('month')}
          >
            Month
          </button>
          <button
            class={`btn ${timeframe() === 'all' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
            onClick={() => setTimeframe('all')}
          >
            All
          </button>
        </div>
      </div>

      <Show when={loading()}>
        <div class="portfolio-loading">
          <div class="spinner"></div>
        </div>
      </Show>

      <Show when={error()}>
        <div class="portfolio-error card">
          <p class="text-danger">{error()}</p>
          <button class="btn btn-secondary btn-sm mt-2" onClick={fetchStats}>
            Try Again
          </button>
        </div>
      </Show>

      <Show when={!loading() && !error() && stats()}>
        <div class="stats-overview">
          <div class="stat-card primary">
            <div class="stat-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
              </svg>
            </div>
            <div class="stat-content">
              <span class={`stat-value font-mono ${stats()!.totalProfit >= 0 ? 'text-success' : 'text-danger'}`}>
                {stats()!.totalProfit >= 0 ? '+' : ''}{formatGold(stats()!.totalProfit)}
              </span>
              <span class="stat-label">Total Profit ({timeframeLabel()})</span>
            </div>
          </div>

          <div class="stat-card">
            <div class="stat-content">
              <span class="stat-value font-mono">{stats()!.totalTrades}</span>
              <span class="stat-label">Total Trades</span>
            </div>
          </div>

          <div class="stat-card">
            <div class="stat-content">
              <span class="stat-value font-mono text-success">{stats()!.winCount}</span>
              <span class="stat-label">Wins</span>
            </div>
          </div>

          <div class="stat-card">
            <div class="stat-content">
              <span class="stat-value font-mono text-danger">{stats()!.lossCount}</span>
              <span class="stat-label">Losses</span>
            </div>
          </div>
        </div>

        <div class="stats-grid grid-2">
          <div class="card">
            <h3 class="card-title">Win Rate</h3>
            <div class="win-rate">
              <div class="win-rate-circle">
                <svg viewBox="0 0 36 36">
                  <path
                    class="win-rate-bg"
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke-width="3"
                  />
                  <path
                    class="win-rate-fill"
                    d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none"
                    stroke-width="3"
                    stroke-dasharray={`${stats()!.winRate * 100}, 100`}
                  />
                </svg>
                <span class="win-rate-value font-mono">{formatPercent(stats()!.winRate)}</span>
              </div>
            </div>
          </div>

          <div class="card">
            <h3 class="card-title">Average Profit</h3>
            <div class="avg-profit">
              <span class={`avg-profit-value font-mono ${stats()!.averageProfit >= 0 ? 'text-success' : 'text-danger'}`}>
                {stats()!.averageProfit >= 0 ? '+' : ''}{formatGold(stats()!.averageProfit)}
              </span>
              <span class="avg-profit-label">per trade</span>
            </div>
          </div>

          <div class="card">
            <h3 class="card-title">Current Streak</h3>
            <div class="streak">
              <span class={`streak-value font-mono ${stats()!.currentStreak.type === 'win' ? 'text-success' : 'text-danger'}`}>
                {stats()!.currentStreak.count}
              </span>
              <span class="streak-label">
                {stats()!.currentStreak.type === 'win' ? 'Wins' : 'Losses'} in a row
              </span>
            </div>
          </div>

          <div class="card">
            <h3 class="card-title">Most Traded</h3>
            <Show when={stats()!.mostTradedItem}>
              <div class="most-traded">
                <span class="most-traded-item">{stats()!.mostTradedItem!.item}</span>
                <span class="most-traded-count">{stats()!.mostTradedItem!.count} trades</span>
              </div>
            </Show>
            <Show when={!stats()!.mostTradedItem}>
              <p class="text-muted">No trades yet</p>
            </Show>
          </div>
        </div>

        <div class="best-worst grid-2">
          <div class="card best">
            <h3 class="card-title">Best Trade</h3>
            <Show when={stats()!.bestTrade}>
              <div class="trade-highlight">
                <span class="trade-highlight-item">{stats()!.bestTrade!.item}</span>
                <span class="trade-highlight-profit text-success font-mono">
                  +{formatGold(stats()!.bestTrade!.profit)}
                </span>
              </div>
            </Show>
            <Show when={!stats()!.bestTrade}>
              <p class="text-muted">No trades yet</p>
            </Show>
          </div>

          <div class="card worst">
            <h3 class="card-title">Worst Trade</h3>
            <Show when={stats()!.worstTrade}>
              <div class="trade-highlight">
                <span class="trade-highlight-item">{stats()!.worstTrade!.item}</span>
                <span class="trade-highlight-profit text-danger font-mono">
                  {formatGold(stats()!.worstTrade!.profit)}
                </span>
              </div>
            </Show>
            <Show when={!stats()!.worstTrade}>
              <p class="text-muted">No trades yet</p>
            </Show>
          </div>
        </div>
      </Show>

      <style>{`
        .portfolio {
          max-width: 900px;
          margin: 0 auto;
        }

        .portfolio-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: var(--space-4);
          flex-wrap: wrap;
          gap: var(--space-3);
        }

        .timeframe-selector {
          display: flex;
          gap: var(--space-2);
        }

        .portfolio-loading, .portfolio-error {
          text-align: center;
          padding: var(--space-6);
        }

        .portfolio-loading .spinner {
          margin: 0 auto;
        }

        .stats-overview {
          display: grid;
          grid-template-columns: 2fr 1fr 1fr 1fr;
          gap: var(--space-4);
          margin-bottom: var(--space-4);
        }

        .stat-card {
          background-color: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          padding: var(--space-4);
          display: flex;
          align-items: center;
          gap: var(--space-3);
        }

        .stat-card.primary {
          background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
        }

        .stat-icon {
          width: 48px;
          height: 48px;
          display: flex;
          align-items: center;
          justify-content: center;
          background-color: var(--accent-light);
          border-radius: var(--radius-md);
          color: var(--accent);
        }

        .stat-icon svg {
          width: 24px;
          height: 24px;
        }

        .stat-content {
          display: flex;
          flex-direction: column;
        }

        .stat-value {
          font-size: var(--font-size-xl);
          font-weight: 700;
        }

        .stat-label {
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
        }

        .stats-grid {
          margin-bottom: var(--space-4);
        }

        .card-title {
          font-size: var(--font-size-base);
          font-weight: 600;
          margin-bottom: var(--space-3);
          color: var(--text-secondary);
        }

        .win-rate {
          display: flex;
          justify-content: center;
        }

        .win-rate-circle {
          position: relative;
          width: 120px;
          height: 120px;
        }

        .win-rate-circle svg {
          transform: rotate(-90deg);
        }

        .win-rate-bg {
          stroke: var(--bg-tertiary);
        }

        .win-rate-fill {
          stroke: var(--success);
          stroke-linecap: round;
          transition: stroke-dasharray var(--transition-normal);
        }

        .win-rate-value {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          font-size: var(--font-size-xl);
          font-weight: 700;
        }

        .avg-profit {
          text-align: center;
          padding: var(--space-4);
        }

        .avg-profit-value {
          display: block;
          font-size: var(--font-size-2xl);
          font-weight: 700;
        }

        .avg-profit-label {
          font-size: var(--font-size-sm);
          color: var(--text-muted);
        }

        .streak {
          text-align: center;
          padding: var(--space-4);
        }

        .streak-value {
          display: block;
          font-size: var(--font-size-3xl);
          font-weight: 700;
        }

        .streak-label {
          font-size: var(--font-size-sm);
          color: var(--text-muted);
        }

        .most-traded {
          display: flex;
          flex-direction: column;
          text-align: center;
          padding: var(--space-3);
        }

        .most-traded-item {
          font-size: var(--font-size-lg);
          font-weight: 600;
        }

        .most-traded-count {
          font-size: var(--font-size-sm);
          color: var(--text-muted);
        }

        .best-worst {
          margin-bottom: var(--space-4);
        }

        .trade-highlight {
          display: flex;
          flex-direction: column;
          text-align: center;
          padding: var(--space-3);
        }

        .trade-highlight-item {
          font-size: var(--font-size-lg);
          font-weight: 600;
          margin-bottom: var(--space-1);
        }

        .trade-highlight-profit {
          font-size: var(--font-size-xl);
          font-weight: 700;
        }

        @media (max-width: 768px) {
          .stats-overview {
            grid-template-columns: 1fr 1fr;
          }

          .stat-card.primary {
            grid-column: span 2;
          }

          .timeframe-selector {
            width: 100%;
            justify-content: center;
          }
        }

        @media (max-width: 480px) {
          .stats-overview {
            grid-template-columns: 1fr;
          }

          .stat-card.primary {
            grid-column: span 1;
          }
        }
      `}</style>
    </div>
  );
}
