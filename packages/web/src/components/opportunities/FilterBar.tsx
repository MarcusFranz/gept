// packages/web/src/components/opportunities/FilterBar.tsx
import type { OpportunityFilters } from '../../lib/trade-types';
import { FILTER_STORAGE_KEY } from '../../lib/trade-types';

interface FilterBarProps {
  filters: OpportunityFilters;
  onChange: (filters: OpportunityFilters) => void;
  availableCapital: number;
}

export function FilterBar(props: FilterBarProps) {
  const updateFilter = <K extends keyof OpportunityFilters>(
    key: K,
    value: OpportunityFilters[K]
  ) => {
    const newFilters = { ...props.filters, [key]: value };
    props.onChange(newFilters);

    // Persist to localStorage
    try {
      localStorage.setItem(FILTER_STORAGE_KEY, JSON.stringify(newFilters));
    } catch {}
  };

  const formatGold = (amount: number) => {
    if (amount >= 1_000_000) {
      return (amount / 1_000_000).toFixed(1) + 'M';
    } else if (amount >= 1_000) {
      return Math.round(amount / 1_000) + 'K';
    }
    return amount.toLocaleString();
  };

  return (
    <div class="filter-bar">
      <div class="filter-bar-content">
        <label class="filter-label">Max Capital</label>
        <input
          type="number"
          class="filter-input"
          placeholder={`e.g. ${formatGold(props.availableCapital)}`}
          value={props.filters.capitalMax || ''}
          onInput={(e) => updateFilter('capitalMax', e.currentTarget.value ? Number(e.currentTarget.value) : undefined)}
        />
        {props.filters.capitalMax && (
          <button
            class="filter-clear"
            onClick={() => {
              updateFilter('capitalMax', undefined);
            }}
          >
            ×
          </button>
        )}
      </div>

      <style>{`
        .filter-bar {
          margin-bottom: 1rem;
        }

        .filter-bar-content {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .filter-label {
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
          white-space: nowrap;
        }

        .filter-input {
          flex: 1;
          max-width: 200px;
          padding: 0.5rem 0.75rem;
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius-md);
          color: var(--text-primary);
          font-size: var(--font-size-sm);
        }

        .filter-input:focus {
          outline: none;
          border-color: var(--accent);
        }

        .filter-input::placeholder {
          color: var(--text-muted);
        }

        .filter-clear {
          background: none;
          border: none;
          color: var(--text-muted);
          cursor: pointer;
          font-size: 1.25rem;
          padding: 0.25rem;
          line-height: 1;
        }

        .filter-clear:hover {
          color: var(--text-primary);
        }
      `}</style>
    </div>
  );
}
