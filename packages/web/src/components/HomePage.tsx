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

    // Switch to trades tab
    setActiveTab('trades');
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
            onTradeAdded={handleTradeAdded}
          />
        </Show>
      </main>

      <style>{`
        .home-page {
          min-height: 100vh;
          background: var(--surface-1, #0f0f1a);
        }

        .home-nav {
          display: flex;
          justify-content: center;
          gap: 0.5rem;
          padding: 1rem;
          background: var(--surface-2, #1a1a2e);
          border-bottom: 1px solid var(--border, #333);
          position: sticky;
          top: 0;
          z-index: 10;
        }

        .nav-tab {
          padding: 0.625rem 1.25rem;
          background: transparent;
          color: var(--text-secondary, #aaa);
          border: none;
          border-radius: 6px;
          font-weight: 600;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          transition: background 0.15s, color 0.15s;
        }

        .nav-tab:hover {
          color: var(--text-primary, #fff);
        }

        .nav-tab-active {
          background: var(--accent, #4f46e5);
          color: white;
        }

        .nav-tab-badge {
          background: rgba(255,255,255,0.2);
          padding: 0.125rem 0.5rem;
          border-radius: 10px;
          font-size: 0.75rem;
        }

        .home-content {
          padding-bottom: 2rem;
        }
      `}</style>
    </div>
  );
}
