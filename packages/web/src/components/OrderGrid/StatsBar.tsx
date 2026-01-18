import { Show, For, type JSX } from 'solid-js';
import type { SlotSettings, OrderState } from './types';
import { formatRsNumber } from './utils';
import { capitalPresets } from '../../lib/types';

interface TradeStats {
  totalInvested: number;
  potentialProfit: number;
  activeCount: number;
  buyingCount: number;
  sellingCount: number;
}

interface StatsBarProps {
  userCapital: number;
  globalDefaults: SlotSettings;
  pendingCapital: number | null;
  pendingSettings: Partial<SlotSettings> | null;
  tradeStats: TradeStats;
  showStatsPanel: boolean;
  showHistoryPanel: boolean;
  capitalInput: string;
  hasPendingChanges: boolean;
  onCapitalInputChange: (value: string) => void;
  onCapitalUpdate: (value: number) => void;
  onSettingsUpdate: (key: keyof SlotSettings, value: string) => void;
  onToggleStatsPanel: () => void;
  onToggleHistoryPanel: () => void;
  onConfirmChanges: () => void;
  onDiscardChanges: () => void;
}

export default function StatsBar(props: StatsBarProps): JSX.Element {
  const effectiveCapital = () => props.pendingCapital ?? props.userCapital;
  const effectiveSettings = () => ({
    ...props.globalDefaults,
    ...props.pendingSettings,
  });

  return (
    <div class="controls-row">
      {/* Left: Capital */}
      <div class="control-group capital-control">
        <button
          class={`dropdown-trigger ${props.showStatsPanel ? 'active' : ''}`}
          onClick={props.onToggleStatsPanel}
        >
          <span class="dropdown-label">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <path d="M12 6v12M8 12h8"/>
            </svg>
            <span class="capital-display">{formatRsNumber(effectiveCapital())}</span>
            <Show when={props.hasPendingChanges}>
              <span class="pending-indicator">*</span>
            </Show>
          </span>
          <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </button>

        <Show when={props.showStatsPanel}>
          <div class="dropdown-panel capital-panel">
            <div class="panel-section">
              <label class="panel-label">Trading Capital</label>
              <div class="capital-presets">
                <For each={capitalPresets}>
                  {(preset) => (
                    <button
                      class={`preset-btn ${effectiveCapital() === preset.value ? 'active' : ''}`}
                      onClick={() => props.onCapitalUpdate(preset.value)}
                    >
                      {preset.label}
                    </button>
                  )}
                </For>
              </div>
              <div class="capital-custom">
                <input
                  type="text"
                  placeholder="Custom (e.g. 75m)"
                  value={props.capitalInput}
                  onInput={(e) => props.onCapitalInputChange(e.currentTarget.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      const value = parseInt(props.capitalInput.replace(/[^0-9]/g, ''), 10);
                      if (value > 0) {
                        const multiplier = props.capitalInput.toLowerCase().includes('m') ? 1_000_000 :
                                          props.capitalInput.toLowerCase().includes('k') ? 1_000 : 1;
                        props.onCapitalUpdate(value * multiplier);
                      }
                    }
                  }}
                />
              </div>
            </div>

            <div class="panel-section">
              <label class="panel-label">Default Flip Settings</label>

              <div class="setting-row">
                <span class="setting-name">Style</span>
                <div class="setting-options">
                  <For each={['passive', 'hybrid', 'active'] as const}>
                    {(opt) => (
                      <button
                        class={`setting-opt ${effectiveSettings().style === opt ? 'active' : ''}`}
                        onClick={() => props.onSettingsUpdate('style', opt)}
                      >
                        {opt}
                      </button>
                    )}
                  </For>
                </div>
              </div>

              <div class="setting-row">
                <span class="setting-name">Risk</span>
                <div class="setting-options">
                  <For each={['low', 'medium', 'high'] as const}>
                    {(opt) => (
                      <button
                        class={`setting-opt ${effectiveSettings().risk === opt ? 'active' : ''}`}
                        onClick={() => props.onSettingsUpdate('risk', opt)}
                      >
                        {opt}
                      </button>
                    )}
                  </For>
                </div>
              </div>

              <div class="setting-row">
                <span class="setting-name">Margin</span>
                <div class="setting-options">
                  <For each={['conservative', 'moderate', 'aggressive'] as const}>
                    {(opt) => (
                      <button
                        class={`setting-opt ${effectiveSettings().margin === opt ? 'active' : ''}`}
                        onClick={() => props.onSettingsUpdate('margin', opt)}
                      >
                        {opt.slice(0, 4)}
                      </button>
                    )}
                  </For>
                </div>
              </div>
            </div>

            <Show when={props.hasPendingChanges}>
              <div class="panel-actions">
                <button class="btn btn-ghost btn-sm" onClick={props.onDiscardChanges}>
                  Cancel
                </button>
                <button class="btn btn-primary btn-sm" onClick={props.onConfirmChanges}>
                  Apply Changes
                </button>
              </div>
            </Show>
          </div>
        </Show>
      </div>

      {/* Center: Stats */}
      <div class="quick-stats">
        <div class="quick-stat">
          <span class="quick-stat-value">{props.tradeStats.activeCount}</span>
          <span class="quick-stat-label">Active</span>
        </div>
        <div class="quick-stat profit">
          <span class="quick-stat-value">+{formatRsNumber(props.tradeStats.potentialProfit)}</span>
          <span class="quick-stat-label">Potential</span>
        </div>
      </div>

      {/* Right: History */}
      <button
        class={`control-btn ${props.showHistoryPanel ? 'active' : ''}`}
        onClick={props.onToggleHistoryPanel}
        title="Trade History"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <polyline points="12 6 12 12 16 14"/>
        </svg>
      </button>

      <style>{`
        .controls-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: var(--space-3);
          margin-bottom: var(--space-4);
          padding: var(--space-3);
          background-color: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
        }

        .control-group {
          position: relative;
        }

        .dropdown-trigger {
          display: flex;
          align-items: center;
          gap: var(--space-2);
          padding: var(--space-2) var(--space-3);
          background-color: var(--bg-tertiary);
          border: 1px solid var(--border);
          border-radius: var(--radius-md);
          color: var(--text-primary);
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .dropdown-trigger:hover,
        .dropdown-trigger.active {
          border-color: var(--accent);
          background-color: var(--bg-hover);
        }

        .dropdown-label {
          display: flex;
          align-items: center;
          gap: var(--space-2);
        }

        .dropdown-label svg {
          width: 16px;
          height: 16px;
          color: var(--gold);
        }

        .capital-display {
          font-family: 'Barlow Condensed', sans-serif;
          font-weight: 600;
          font-size: 16px;
          color: var(--gold);
        }

        .pending-indicator {
          color: var(--accent);
          font-weight: bold;
        }

        .chevron {
          width: 14px;
          height: 14px;
          color: var(--text-muted);
          transition: transform var(--transition-fast);
        }

        .dropdown-trigger.active .chevron {
          transform: rotate(180deg);
        }

        .dropdown-panel {
          position: absolute;
          top: calc(100% + 8px);
          left: 0;
          min-width: 280px;
          background-color: var(--bg-secondary);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          z-index: 100;
          padding: var(--space-4);
        }

        .panel-section {
          margin-bottom: var(--space-4);
        }

        .panel-section:last-child {
          margin-bottom: 0;
        }

        .panel-label {
          display: block;
          font-size: var(--font-size-xs);
          font-weight: 600;
          text-transform: uppercase;
          color: var(--text-muted);
          margin-bottom: var(--space-2);
        }

        .capital-presets {
          display: flex;
          flex-wrap: wrap;
          gap: var(--space-1);
          margin-bottom: var(--space-2);
        }

        .preset-btn {
          padding: var(--space-1) var(--space-2);
          font-size: var(--font-size-xs);
          background-color: var(--bg-tertiary);
          border: 1px solid var(--border);
          border-radius: var(--radius-sm);
          color: var(--text-secondary);
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .preset-btn:hover {
          border-color: var(--accent);
          color: var(--text-primary);
        }

        .preset-btn.active {
          background-color: var(--accent-light);
          border-color: var(--accent);
          color: var(--accent);
        }

        .capital-custom input {
          width: 100%;
          padding: var(--space-2);
          font-size: var(--font-size-sm);
          background-color: var(--bg-tertiary);
          border: 1px solid var(--border);
          border-radius: var(--radius-md);
          color: var(--text-primary);
        }

        .capital-custom input:focus {
          outline: none;
          border-color: var(--accent);
        }

        .setting-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: var(--space-2);
        }

        .setting-name {
          font-size: var(--font-size-sm);
          color: var(--text-secondary);
        }

        .setting-options {
          display: flex;
          gap: var(--space-1);
        }

        .setting-opt {
          padding: var(--space-1) var(--space-2);
          font-size: 10px;
          text-transform: capitalize;
          background-color: var(--bg-tertiary);
          border: 1px solid var(--border);
          border-radius: var(--radius-sm);
          color: var(--text-secondary);
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .setting-opt:hover {
          border-color: var(--accent);
        }

        .setting-opt.active {
          background-color: var(--accent-light);
          border-color: var(--accent);
          color: var(--accent);
        }

        .panel-actions {
          display: flex;
          gap: var(--space-2);
          justify-content: flex-end;
          padding-top: var(--space-3);
          border-top: 1px solid var(--border);
        }

        .quick-stats {
          display: flex;
          align-items: center;
          gap: var(--space-4);
        }

        .quick-stat {
          display: flex;
          flex-direction: column;
          align-items: center;
        }

        .quick-stat-value {
          font-family: 'Barlow Condensed', sans-serif;
          font-size: 18px;
          font-weight: 600;
          color: var(--text-primary);
        }

        .quick-stat.profit .quick-stat-value {
          color: var(--success);
        }

        .quick-stat-label {
          font-size: 10px;
          text-transform: uppercase;
          color: var(--text-muted);
        }

        .control-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 36px;
          height: 36px;
          background-color: var(--bg-tertiary);
          border: 1px solid var(--border);
          border-radius: var(--radius-md);
          color: var(--text-secondary);
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .control-btn:hover,
        .control-btn.active {
          border-color: var(--accent);
          color: var(--text-primary);
        }

        .control-btn svg {
          width: 18px;
          height: 18px;
        }

        @media (max-width: 600px) {
          .quick-stats {
            display: none;
          }
        }
      `}</style>
    </div>
  );
}
