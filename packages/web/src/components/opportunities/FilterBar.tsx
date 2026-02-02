// packages/web/src/components/opportunities/FilterBar.tsx
import type { OpportunityFilters } from '../../lib/trade-types';
import { FILTER_STORAGE_KEY } from '../../lib/trade-types';

interface FilterBarProps {
  filters: OpportunityFilters;
  onChange: (filters: OpportunityFilters) => void;
}

const timeOptions = [
  { label: 'Any time', value: undefined },
  { label: '< 1 hr', value: 1 },
  { label: '< 2 hrs', value: 2 },
  { label: '< 4 hrs', value: 4 },
  { label: '< 8 hrs', value: 8 },
  { label: '< 12 hrs', value: 12 },
  { label: '< 24 hrs', value: 24 },
];

const confidenceOptions = [
  { label: 'Any confidence', value: undefined },
  { label: 'Medium+', value: 'medium' },
  { label: 'High only', value: 'high' },
];

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
    } catch (err) { console.warn('[FilterBar] Failed to persist preferences:', err); }
  };

  return (
    <div class="filter-bar">
      <div class="filter-bar-content">
        <div class="filter-group">
          <label class="filter-label">Max Capital</label>
          <input
            type="number"
            class="filter-input"
            placeholder="Filter by GP"
            value={props.filters.capitalMax || ''}
            onInput={(e) => updateFilter('capitalMax', e.currentTarget.value ? Number(e.currentTarget.value) : undefined)}
          />
        </div>

        <div class="filter-group">
          <label class="filter-label">Max Time</label>
          <select
            class="filter-select"
            value={props.filters.timeMax || ''}
            onChange={(e) => updateFilter('timeMax', e.currentTarget.value ? Number(e.currentTarget.value) : undefined)}
          >
            {timeOptions.map(opt => (
              <option value={opt.value || ''}>{opt.label}</option>
            ))}
          </select>
        </div>

        <div class="filter-group">
          <label class="filter-label">Confidence</label>
          <select
            class="filter-select"
            value={props.filters.confidence || ''}
            onChange={(e) => updateFilter('confidence', e.currentTarget.value || undefined)}
          >
            {confidenceOptions.map(opt => (
              <option value={opt.value || ''}>{opt.label}</option>
            ))}
          </select>
        </div>

        {(props.filters.capitalMax || props.filters.timeMax || props.filters.confidence) && (
          <button
            class="filter-clear-all"
            onClick={() => {
              props.onChange({});
              try {
                localStorage.removeItem(FILTER_STORAGE_KEY);
              } catch (err) { console.warn('[FilterBar] Failed to clear preferences:', err); }
            }}
          >
            Clear
          </button>
        )}
      </div>

      <style>{`
        .filter-bar {
          margin-bottom: 1rem;
        }

        .filter-bar-content {
          display: flex;
          align-items: flex-end;
          gap: 1rem;
          flex-wrap: wrap;
        }

        .filter-group {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
        }

        .filter-label {
          font-size: var(--font-size-xs);
          color: var(--text-secondary);
        }

        .filter-input,
        .filter-select {
          padding: 0.5rem 0.75rem;
          background: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius-md);
          color: var(--text-primary);
          font-size: var(--font-size-sm);
          min-width: 140px;
        }

        .filter-input:focus,
        .filter-select:focus {
          outline: none;
          border-color: var(--accent);
        }

        .filter-input::placeholder {
          color: var(--text-muted);
        }

        .filter-clear-all {
          background: none;
          border: none;
          color: var(--accent);
          cursor: pointer;
          font-size: var(--font-size-sm);
          padding: 0.5rem 0;
        }

        .filter-clear-all:hover {
          text-decoration: underline;
        }
      `}</style>
    </div>
  );
}
