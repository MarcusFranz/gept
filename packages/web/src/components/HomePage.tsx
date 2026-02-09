// packages/web/src/components/HomePage.tsx
import { createSignal, createMemo, Show } from 'solid-js';
import { TradeList } from './trades';
import { OpportunityBrowser } from './opportunities';
import type { ActiveTrade } from '../lib/db';
import type { UpdateRecommendation } from '../lib/types';
import { useAlerts } from '../lib/useAlerts';
import { useTradeUpdates } from '../lib/useTradeUpdates';
import { addToast } from './ToastContainer';

interface HomePageProps {
  user: { id: string; email: string };
  initialTrades: ActiveTrade[];
}

type Tab = 'trades' | 'opportunities';

export function HomePage(props: HomePageProps) {
  const [activeTab, setActiveTab] = createSignal<Tab>('trades');
  const [trades, setTrades] = createSignal(props.initialTrades);

  const { alerts: sseAlerts, dismissAlert: dismissSseAlert } = useAlerts();
  const { updates: polledAlerts, dismissUpdate: dismissPolledAlert } = useTradeUpdates(trades);

  // Merge SSE (real-time) and polled alerts — SSE takes priority
  const alerts = createMemo(() => {
    const merged = new Map<string, UpdateRecommendation>(polledAlerts());
    // SSE overwrites polled for same tradeId
    for (const [id, alert] of sseAlerts()) {
      merged.set(id, alert);
    }
    return merged;
  });

  const dismissAlert = (tradeId: string) => {
    dismissSseAlert(tradeId);
    dismissPolledAlert(tradeId);
  };

  // Active trade item IDs (for filtering opportunities)
  const activeTradeItemIds = () => trades().map(t => t.item_id);

  // Refresh trades and capital after adding a trade
  const handleTradeAdded = async () => {
    try {
      const res = await fetch('/api/trades/active');
      if (!res.ok) throw new Error('Failed to fetch trades');
      const data = await res.json();
      if (data.success) {
        setTrades(data.data);
      }
    } catch (err) {
      console.error('Failed to refresh trades:', err);
    }
  };

  const handleAcceptAlert = async (tradeId: string, newSellPrice: number) => {
    try {
      const res = await fetch(`/api/trades/active/${tradeId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sellPrice: newSellPrice })
      });
      if (!res.ok) throw new Error('Failed to update sell price');
      dismissAlert(tradeId);
      await handleTradeAdded();
      addToast({ type: 'success', title: 'Sell price updated', message: `Revised to ${newSellPrice.toLocaleString()} gp` });
    } catch (err) {
      console.error('Failed to accept alert:', err);
      addToast({ type: 'error', title: 'Update failed', message: 'Could not update sell price' });
    }
  };

  const handleDismissPriceAlert = async (tradeId: string) => {
    try {
      // Clear the persisted suggestion so the inline UI (and badge) disappears.
      // Also ensures polling doesn't immediately recreate the same suggestion.
      await fetch(`/api/trades/active/${tradeId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clearSuggestedSellPrice: true })
      });
    } catch (err) {
      console.error('Failed to dismiss price alert:', err);
    } finally {
      dismissAlert(tradeId);
      await handleTradeAdded();
    }
  };

  return (
    <div class="home-page">
      <nav class="home-nav">
        <div class="nav-switch" role="tablist" aria-label="Trade views">
          <span class={`nav-switch-blob ${activeTab() === 'opportunities' ? 'is-right' : ''}`} aria-hidden="true" />
          <button
            class={`nav-switch-tab ${activeTab() === 'trades' ? 'is-active' : ''}`}
            onClick={() => setActiveTab('trades')}
            role="tab"
            aria-selected={activeTab() === 'trades'}
          >
            My Trades
            <Show when={trades().length > 0}>
              <span class="nav-switch-badge">{trades().length}</span>
            </Show>
          </button>
          <button
            class={`nav-switch-tab ${activeTab() === 'opportunities' ? 'is-active' : ''}`}
            onClick={() => setActiveTab('opportunities')}
            role="tab"
            aria-selected={activeTab() === 'opportunities'}
          >
            Opportunities
          </button>
        </div>
      </nav>

      <main class="home-content">
        <Show when={activeTab() === 'trades'}>
          <TradeList
            initialTrades={trades()}
            onNavigateToOpportunities={() => setActiveTab('opportunities')}
            alerts={alerts()}
            onAcceptAlert={handleAcceptAlert}
            onDismissAlert={(tradeId) => handleDismissPriceAlert(tradeId)}
          />
        </Show>

        <Show when={activeTab() === 'opportunities'}>
          <OpportunityBrowser
            activeTradeItemIds={activeTradeItemIds()}
            onTradeAdded={handleTradeAdded}
            onNavigateToTrades={() => setActiveTab('trades')}
          />
        </Show>
      </main>

      <style>{`
        .home-page {
          min-height: 100vh;
          background: transparent;
        }

        .home-nav {
          --nav-scrim-h: 18px;
          display: flex;
          justify-content: center;
          padding: 1rem;
          position: sticky;
          top: 64px;
          z-index: 10;
          isolation: isolate;
          /* Match the content area's background so this doesn't read as a separate "title bar". */
          background-color: var(--bg-primary);
          background-image: var(--page-gradient);
          background-attachment: fixed;
          background-repeat: no-repeat;
        }

        /* Bottom scrim that fades cards as they scroll under the sticky tabs. */
        .home-nav::after {
          content: '';
          position: absolute;
          bottom: calc(-1 * var(--nav-scrim-h));
          left: 0;
          right: 0;
          height: var(--nav-scrim-h);
          /* Paint the actual page background, then mask it into a fade. */
          background-color: var(--bg-primary);
          background-image: var(--page-gradient);
          background-attachment: fixed;
          background-repeat: no-repeat;
          -webkit-mask-image: linear-gradient(
            to bottom,
            rgba(0, 0, 0, 1) 0%,
            rgba(0, 0, 0, 0.82) 30%,
            rgba(0, 0, 0, 0.38) 68%,
            rgba(0, 0, 0, 0) 100%
          );
          mask-image: linear-gradient(
            to bottom,
            rgba(0, 0, 0, 1) 0%,
            rgba(0, 0, 0, 0.82) 30%,
            rgba(0, 0, 0, 0.38) 68%,
            rgba(0, 0, 0, 0) 100%
          );
          pointer-events: none;
          z-index: 0;
        }

        .nav-switch {
          --switch-pad: 0px;
          --switch-gap: 6px;
          --switch-border: 1px;
          position: relative;
          z-index: 1;
          display: inline-flex;
          align-items: center;
          gap: var(--switch-gap);
          padding: var(--switch-pad);
          border-radius: 999px;
          background: color-mix(in srgb, var(--accent) 60%, #0b0c0f);
          border: 1px solid color-mix(in srgb, var(--accent) 55%, #0b0c0f);
          overflow: hidden;
          box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.35);
        }

        .nav-switch-blob {
          position: absolute;
          inset: calc(var(--switch-pad) + var(--switch-border));
          width: calc((100% - (2 * var(--switch-pad)) - (2 * var(--switch-border)) - var(--switch-gap)) / 2);
          border-radius: calc(999px - (var(--switch-pad) + var(--switch-border)) + 2px);
          background: var(--surface-2);
          border: 1px solid color-mix(in srgb, #000 40%, transparent);
          box-shadow: 0 1px 1px rgba(255, 255, 255, 0.06), inset 0 0 0 1px rgba(255, 255, 255, 0.03);
          transition: transform 0.32s var(--ease-hero), width 0.32s var(--ease-hero);
          z-index: 0;
        }

        .nav-switch-blob.is-right {
          transform: translateX(calc(100% + var(--switch-gap)));
        }

        .nav-switch-tab {
          position: relative;
          z-index: 1;
          padding: 0.55rem 1.3rem;
          background: transparent;
          color: var(--text-primary);
          border: none;
          border-radius: 999px;
          font-weight: 600;
          cursor: pointer;
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          transition: color 0.2s ease, transform 0.2s ease;
        }

        .nav-switch-tab:hover {
          transform: translateY(-1px);
        }

        .nav-switch-tab.is-active {
          color: var(--text-primary);
        }

        .nav-switch-badge {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          background: rgba(0, 0, 0, 0.18);
          color: var(--text-primary);
          padding: 0 0.42rem;
          height: 18px;
          min-width: 24px;
          border-radius: var(--radius-full);
          font-size: var(--font-size-xs);
        }

        .nav-switch-tab.is-active .nav-switch-badge {
          background: color-mix(in srgb, var(--accent) 80%, #0b0c0f);
          color: #0b0c0f;
          transform: none;
          border: none;
          box-shadow: none;
        }

        .home-content {
          padding-bottom: 2rem;
        }
      `}</style>
    </div>
  );
}
