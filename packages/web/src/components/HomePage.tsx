// packages/web/src/components/HomePage.tsx
import { createSignal, Show } from 'solid-js';
import { TradeList } from './trades';
import { OpportunityBrowser } from './opportunities';
import type { ActiveTrade } from '../lib/db';

interface HomePageProps {
  user: { id: string; email: string };
  initialTrades: ActiveTrade[];
  userCapital: number;
  tiedCapital: number;
}

type Tab = 'trades' | 'opportunities';

export function HomePage(props: HomePageProps) {
  const [activeTab, setActiveTab] = createSignal<Tab>('trades');
  const [trades, setTrades] = createSignal(props.initialTrades);
  const [tiedCapital, setTiedCapital] = createSignal(props.tiedCapital);

  const availableCapital = () => Math.max(0, props.userCapital - tiedCapital());

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
        // Recalculate tied capital
        const tied = data.data.reduce(
          (sum: number, t: ActiveTrade) => sum + t.buy_price * t.quantity,
          0
        );
        setTiedCapital(tied);
      }
    } catch (err) {
      console.error('Failed to refresh trades:', err);
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
            availableCapital={availableCapital()}
            totalCapital={props.userCapital}
            onNavigateToOpportunities={() => setActiveTab('opportunities')}
          />
        </Show>

        <Show when={activeTab() === 'opportunities'}>
          <OpportunityBrowser
            availableCapital={availableCapital()}
            totalCapital={props.userCapital}
            activeTradeItemIds={activeTradeItemIds()}
            onTradeAdded={handleTradeAdded}
            onNavigateToTrades={() => setActiveTab('trades')}
          />
        </Show>
      </main>

      <style>{`
        .home-page {
          min-height: 100vh;
          background: var(--bg-primary);
        }

        .home-nav {
          display: flex;
          justify-content: center;
          gap: 0.5rem;
          padding: 1rem;
          background: var(--bg-secondary);
          border-bottom: 1px solid var(--border);
          position: sticky;
          top: 56px;
          z-index: 10;
        }

        .nav-tab {
          padding: 0.625rem 1.25rem;
          background: transparent;
          color: var(--text-secondary);
          border: none;
          border-radius: var(--radius-md);
          font-weight: 600;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          transition: background var(--transition-fast), color var(--transition-fast);
        }

        .nav-tab:hover:not(.nav-tab-active) {
          color: var(--text-primary);
          background: var(--bg-hover);
        }

        .nav-tab-active,
        .nav-tab-active:hover {
          background: var(--accent);
          color: var(--btn-text-dark);
        }

        .nav-tab-badge {
          background: rgba(0,0,0,0.2);
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
