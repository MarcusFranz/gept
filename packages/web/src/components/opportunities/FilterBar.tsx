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
            placeholder="Max GP"
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
            class={`filter-select filter-select-confidence ${
              props.filters.confidence === 'high'
                ? 'is-high'
                : props.filters.confidence === 'medium'
                  ? 'is-medium'
                  : props.filters.confidence === 'low'
                    ? 'is-low'
                    : 'is-any'
            }`}
            value={props.filters.confidence === 'low' ? '' : (props.filters.confidence || '')}
            onChange={(e) => updateFilter('confidence', (e.currentTarget.value || undefined) as OpportunityFilters['confidence'])}
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
          margin-bottom: 0.65rem;
        }

        .filter-bar-content {
          display: flex;
          align-items: center;
          gap: 0.65rem;
          flex-wrap: nowrap;
          background: color-mix(in srgb, var(--surface-1) 90%, #000 10%);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          padding: 0.65rem;
          box-shadow: none;
          transition: border-color 0.18s ease, background 0.18s ease;
        }

        .filter-bar-content:hover {
          border-color: var(--border-light);
        }

        .filter-group {
          display: flex;
          flex-direction: column;
          gap: 0.25rem;
          flex: 1 1 0;
          min-width: 0;
        }

        .filter-label {
          font-size: 0.6rem;
          letter-spacing: 0.18em;
          text-transform: uppercase;
          color: var(--text-muted);
          font-weight: 700;
          white-space: nowrap;
        }

        .filter-input,
        .filter-select {
          width: 100%;
          min-width: 0;
          padding: 0.45rem 0.6rem;
          background: var(--surface-2);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          color: var(--text-primary);
          font-size: var(--font-size-sm);
          height: 36px;
        }

        .filter-input::-webkit-outer-spin-button,
        .filter-input::-webkit-inner-spin-button {
          -webkit-appearance: none;
          margin: 0;
        }

        .filter-input[type="number"] {
          -moz-appearance: textfield;
          appearance: textfield;
        }

        .filter-select {
          appearance: none;
          padding-right: 1.8rem;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          background-image:
            linear-gradient(45deg, transparent 50%, rgba(244, 246, 242, 0.75) 50%),
            linear-gradient(135deg, rgba(244, 246, 242, 0.75) 50%, transparent 50%),
            linear-gradient(to right, transparent, transparent);
          background-position:
            calc(100% - 0.95rem) 55%,
            calc(100% - 0.75rem) 55%,
            0 0;
          background-size:
            6px 6px,
            6px 6px,
            100% 100%;
          background-repeat: no-repeat;
        }

        .filter-select-confidence {
          --conf-accent: color-mix(in srgb, var(--text-muted) 55%, transparent);
          padding-left: 1.35rem;
          background-image:
            radial-gradient(circle at 0.7rem 50%, var(--conf-accent) 0 4px, transparent 5px),
            linear-gradient(45deg, transparent 50%, rgba(244, 246, 242, 0.75) 50%),
            linear-gradient(135deg, rgba(244, 246, 242, 0.75) 50%, transparent 50%),
            linear-gradient(to right, transparent, transparent);
          background-position:
            0 0,
            calc(100% - 0.95rem) 55%,
            calc(100% - 0.75rem) 55%,
            0 0;
          background-size:
            100% 100%,
            6px 6px,
            6px 6px,
            100% 100%;
          background-repeat: no-repeat;
        }

        .filter-select-confidence.is-high {
          --conf-accent: var(--success);
          border-color: color-mix(in srgb, var(--success) 45%, var(--border));
        }

        .filter-select-confidence.is-medium {
          --conf-accent: var(--warning);
          border-color: color-mix(in srgb, var(--warning) 45%, var(--border));
        }

        .filter-select-confidence.is-low {
          --conf-accent: var(--danger);
          border-color: color-mix(in srgb, var(--danger) 45%, var(--border));
        }

        .filter-input:focus,
        .filter-select:focus {
          outline: none;
          border-color: var(--border-light);
          box-shadow: 0 0 0 3px rgba(168, 240, 8, 0.12);
        }

        .filter-input::placeholder {
          color: var(--text-muted);
        }

        .filter-clear-all {
          flex: 0 0 auto;
          background: var(--surface-2);
          border: 1px solid var(--border);
          color: var(--text-secondary);
          cursor: pointer;
          font-size: var(--font-size-sm);
          padding: 0 0.7rem;
          height: 36px;
          border-radius: var(--radius-lg);
          white-space: nowrap;
        }

        .filter-clear-all:hover {
          background: var(--surface-3);
          border-color: var(--border-light);
        }

        @media (max-width: 420px) {
          .filter-bar-content {
            gap: 0.5rem;
            padding: 0.55rem;
          }

          .filter-label {
            font-size: 0.58rem;
            letter-spacing: 0.16em;
          }

          .filter-input,
          .filter-select,
          .filter-clear-all {
            height: 34px;
          }
        }

        @media (max-width: 360px) {
          .filter-group {
            gap: 0;
          }

          .filter-label {
            display: none;
          }
        }
      `}</style>
    </div>
  );
}
