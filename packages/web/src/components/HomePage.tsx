// packages/web/src/components/HomePage.tsx
import { createSignal, Show } from 'solid-js';
import { TradeList } from './trades';
import { OpportunityBrowser } from './opportunities';
import type { ActiveTrade } from '../lib/db';
import { useAlerts } from '../lib/useAlerts';
import { addToast } from './ToastContainer';

interface HomePageProps {
  user: { id: string; email: string };
  initialTrades: ActiveTrade[];
}

type Tab = 'trades' | 'opportunities';

export function HomePage(props: HomePageProps) {
  const [activeTab, setActiveTab] = createSignal<Tab>('trades');
  const [trades, setTrades] = createSignal(props.initialTrades);

  const { alerts, dismissAlert } = useAlerts();

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

  return (
    <div class="home-page">
      <nav class="home-nav">
        <button
          class={`nav-tab ${activeTab() === 'trades' ? 'nav-tab-active' : ''}`}
          onClick={() => setActiveTab('trades')}
        >
          My Trades
          <Show when={trades().length > 0}>
            <span class="nav-tab-badge">{trades().length}</span>
          </Show>
        </button>
        <button
          class={`nav-tab ${activeTab() === 'opportunities' ? 'nav-tab-active' : ''}`}
          onClick={() => setActiveTab('opportunities')}
        >
          Opportunities
        </button>
      </nav>

      <main class="home-content">
        <Show when={activeTab() === 'trades'}>
          <TradeList
            initialTrades={trades()}
            onNavigateToOpportunities={() => setActiveTab('opportunities')}
            alerts={alerts()}
            onAcceptAlert={handleAcceptAlert}
            onDismissAlert={(tradeId) => dismissAlert(tradeId)}
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
          display: flex;
          justify-content: center;
          gap: 0.5rem;
          padding: 1rem;
          background: var(--nav-bg);
          border-bottom: 1px solid var(--border);
          position: sticky;
          top: 56px;
          z-index: 10;
          backdrop-filter: blur(14px);
          box-shadow: var(--shadow-sm);
        }

        .nav-tab {
          padding: 0.625rem 1.25rem;
          background: var(--glass-bg);
          color: var(--text-secondary);
          border: 1px solid var(--glass-border);
          border-radius: var(--radius-full);
          font-weight: 600;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          transition: background var(--transition-fast), color var(--transition-fast), border-color var(--transition-fast);
        }

        .nav-tab:hover:not(.nav-tab-active) {
          color: var(--text-primary);
          background: var(--hover-bg);
          border-color: var(--border-light);
        }

        .nav-tab-active,
        .nav-tab-active:hover {
          background: linear-gradient(135deg, var(--accent) 0%, var(--action) 100%);
          color: var(--btn-text-dark);
          border-color: transparent;
        }

        .nav-tab-badge {
          background: rgba(11, 13, 18, 0.25);
          padding: 0.125rem 0.5rem;
          border-radius: var(--radius-full);
          font-size: var(--font-size-xs);
        }

        .home-content {
          padding-bottom: 2rem;
        }
      `}</style>
    </div>
  );
}
