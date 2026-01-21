// packages/web/src/components/opportunities/FilterBar.tsx
import { createSignal } from 'solid-js';
import type { OpportunityFilters } from '../../lib/trade-types';
import { FILTER_STORAGE_KEY } from '../../lib/trade-types';

interface FilterBarProps {
  filters: OpportunityFilters;
  onChange: (filters: OpportunityFilters) => void;
  availableCapital: number;
}

export function FilterBar(props: FilterBarProps) {
  const [expanded, setExpanded] = createSignal(false);

  const timeOptions = [
    { label: 'Any', value: undefined },
    { label: '< 2 hrs', value: 2 },
    { label: '< 4 hrs', value: 4 },
    { label: '< 8 hrs', value: 8 },
  ];

  const confidenceOptions = [
    { label: 'All', value: 'all' as const },
    { label: 'High', value: 'high' as const },
    { label: 'Medium+', value: 'medium' as const },
  ];

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

  const activeFilterCount = () => {
    let count = 0;
    if (props.filters.profitMin) count++;
    if (props.filters.profitMax) count++;
    if (props.filters.timeMax) count++;
    if (props.filters.confidence && props.filters.confidence !== 'all') count++;
    if (props.filters.capitalMax) count++;
    if (props.filters.category) count++;
    return count;
  };

  return (
    <div class="filter-bar">
      <div class="filter-bar-header">
        <button
          class="filter-toggle"
          onClick={() => setExpanded(!expanded())}
        >
          Filters {activeFilterCount() > 0 && `(${activeFilterCount()})`}
          <span class="filter-toggle-icon">{expanded() ? '▲' : '▼'}</span>
        </button>

        {activeFilterCount() > 0 && (
          <button
            class="filter-clear"
            onClick={() => props.onChange({})}
          >
            Clear all
          </button>
        )}
      </div>

      {expanded() && (
        <div class="filter-bar-content">
          <div class="filter-group">
            <label>Min Profit</label>
            <input
              type="number"
              placeholder="e.g. 50000"
              value={props.filters.profitMin || ''}
              onInput={(e) => updateFilter('profitMin', e.currentTarget.value ? Number(e.currentTarget.value) : undefined)}
            />
          </div>

          <div class="filter-group">
            <label>Max Time</label>
            <select
              value={props.filters.timeMax || ''}
              onChange={(e) => updateFilter('timeMax', e.currentTarget.value ? Number(e.currentTarget.value) : undefined)}
            >
              {timeOptions.map(opt => (
                <option value={opt.value || ''}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div class="filter-group">
            <label>Confidence</label>
            <select
              value={props.filters.confidence || 'all'}
              onChange={(e) => updateFilter('confidence', e.currentTarget.value as OpportunityFilters['confidence'])}
            >
              {confidenceOptions.map(opt => (
                <option value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>

          <div class="filter-group">
            <label>
              <input
                type="checkbox"
                checked={props.filters.capitalMax === props.availableCapital}
                onChange={(e) => updateFilter('capitalMax', e.currentTarget.checked ? props.availableCapital : undefined)}
              />
              Only show what I can afford
            </label>
          </div>
        </div>
      )}

      <style>{`
        .filter-bar {
          background: var(--surface-2, #1a1a2e);
          border: 1px solid var(--border, #333);
          border-radius: 8px;
          margin-bottom: 1rem;
        }

        .filter-bar-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 0.75rem 1rem;
        }

        .filter-toggle {
          background: none;
          border: none;
          color: var(--text-primary, #fff);
          font-weight: 600;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .filter-toggle-icon {
          font-size: 0.75rem;
          color: var(--text-tertiary, #666);
        }

        .filter-clear {
          background: none;
          border: none;
          color: var(--accent, #4f46e5);
          cursor: pointer;
          font-size: 0.875rem;
        }

        .filter-bar-content {
          padding: 0 1rem 1rem;
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 1rem;
        }

        .filter-group {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .filter-group label {
          font-size: 0.75rem;
          color: var(--text-secondary, #aaa);
        }

        .filter-group input[type="number"],
        .filter-group select {
          padding: 0.5rem;
          background: var(--surface-3, #252540);
          border: 1px solid var(--border, #444);
          border-radius: 4px;
          color: var(--text-primary, #fff);
        }

        .filter-group input[type="checkbox"] {
          margin-right: 0.5rem;
        }
      `}</style>
    </div>
  );
}
