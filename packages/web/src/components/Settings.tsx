import { createSignal, onMount, Show, For } from 'solid-js';
import { type AppSettings, defaultAppSettings } from '../lib/types';

interface UserSettings {
  tier: 'free' | 'premium';
  useBetaModel: boolean;
}

interface RateLimitInfo {
  daily: {
    used: number;
    limit: number;
    remaining: number;
  };
  tier: string;
}

export default function Settings() {
  const [settings, setSettings] = createSignal<UserSettings | null>(null);
  const [appSettings, setAppSettings] = createSignal<AppSettings>(defaultAppSettings);
  const [rateLimit, setRateLimit] = createSignal<RateLimitInfo | null>(null);
  const [loading, setLoading] = createSignal(true);
  const [saving, setSaving] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [success, setSuccess] = createSignal<string | null>(null);
  const [activeTab, setActiveTab] = createSignal<'account' | 'app'>('account');

  const fetchSettings = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/settings');
      if (!response.ok) {
        setError('Failed to load settings');
        return;
      }
      const data = await response.json();
      if (data.success) {
        setSettings(data.data.user);
        setRateLimit(data.data.rateLimit);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('Failed to load settings');
    } finally {
      setLoading(false);
    }
  };

  const loadAppSettings = () => {
    const saved = localStorage.getItem('appSettings');
    if (saved) {
      try {
        setAppSettings({ ...defaultAppSettings, ...JSON.parse(saved) });
      } catch {
        setAppSettings(defaultAppSettings);
      }
    }
  };

  onMount(() => {
    fetchSettings();
    loadAppSettings();
  });

  const saveAccountSettings = async () => {
    if (!settings()) return;
    setSaving(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings())
      });
      if (!response.ok) {
        setError('Failed to save settings');
        return;
      }
      const data = await response.json();
      if (data.success) {
        setSuccess('Settings saved successfully!');
        setTimeout(() => setSuccess(null), 3000);
      } else {
        setError(data.error);
      }
    } catch (err) {
      setError('Failed to save settings');
    } finally {
      setSaving(false);
    }
  };

  const saveAppSettings = () => {
    localStorage.setItem('appSettings', JSON.stringify(appSettings()));
    // Apply theme
    document.documentElement.dataset.theme = appSettings().theme === 'system'
      ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
      : appSettings().theme;
    setSuccess('App settings saved!');
    setTimeout(() => setSuccess(null), 3000);
  };

  const updateSetting = <K extends keyof UserSettings>(key: K, value: UserSettings[K]) => {
    setSettings(s => s ? { ...s, [key]: value } : null);
  };

  const updateAppSetting = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setAppSettings(s => ({ ...s, [key]: value }));
  };

  const rateLimitPercent = () => {
    if (!rateLimit()) return 0;
    return (rateLimit()!.daily.used / rateLimit()!.daily.limit) * 100;
  };

  const rateLimitClass = () => {
    const pct = rateLimitPercent();
    if (pct >= 90) return 'danger';
    if (pct >= 70) return 'warning';
    return '';
  };

  return (
    <div class="settings">
      <h2>Settings</h2>

      <div class="tabs" role="tablist">
        <button
          class={`tab ${activeTab() === 'account' ? 'active' : ''}`}
          onClick={() => setActiveTab('account')}
          role="tab"
          aria-selected={activeTab() === 'account'}
        >
          Account Settings
        </button>
        <button
          class={`tab ${activeTab() === 'app' ? 'active' : ''}`}
          onClick={() => setActiveTab('app')}
          role="tab"
          aria-selected={activeTab() === 'app'}
        >
          App Settings
        </button>
      </div>

      <Show when={success()}>
        <div class="alert alert-success">{success()}</div>
      </Show>

      <Show when={error()}>
        <div class="alert alert-error">{error()}</div>
      </Show>

      <Show when={loading()}>
        <div class="settings-loading">
          <div class="spinner"></div>
        </div>
      </Show>

      {/* Account Settings Tab */}
      <Show when={!loading() && activeTab() === 'account' && settings()}>
        <div class="settings-section">
          {/* Rate Limit */}
          <div class="card rate-limit-card">
            <div class="rate-limit-header">
              <span class="rate-limit-title">Daily Queries</span>
              <span class="rate-limit-count font-mono">
                {rateLimit()?.daily.used} / {rateLimit()?.daily.limit}
              </span>
            </div>
            <div class="progress">
              <div
                class={`progress-bar ${rateLimitClass()}`}
                style={{ width: `${rateLimitPercent()}%` }}
              />
            </div>
            <p class="rate-limit-tier text-sm text-muted mt-2">
              Tier: <span class="font-medium">{settings()!.tier}</span>
            </p>
          </div>

          {/* Beta Model Toggle */}
          <div class="form-group" style={{ "margin-bottom": "var(--space-4)" }}>
            <label class="toggle-label">
              <input
                type="checkbox"
                checked={settings()!.useBetaModel}
                onChange={(e) => updateSetting('useBetaModel', e.currentTarget.checked)}
              />
              <span>Use beta model if available</span>
            </label>
            <p class="text-sm text-muted" style={{ "margin-top": "var(--space-1)", "margin-left": "calc(18px + var(--space-3))" }}>
              Try our latest prediction model. Beta models may be less stable but can offer improved accuracy.
            </p>
          </div>

          <button
            class="btn btn-primary w-full"
            onClick={saveAccountSettings}
            disabled={saving()}
          >
            {saving() ? 'Saving...' : 'Save Account Settings'}
          </button>
        </div>
      </Show>

      {/* App Settings Tab */}
      <Show when={!loading() && activeTab() === 'app'}>
        <div class="settings-section">
          {/* Theme */}
          <div class="form-group">
            <label class="label">Theme</label>
            <div class="option-group horizontal">
              <For each={['light', 'dark', 'system'] as const}>
                {(theme) => (
                  <button
                    class={`option-btn small ${appSettings().theme === theme ? 'active' : ''}`}
                    onClick={() => updateAppSetting('theme', theme)}
                  >
                    {theme.charAt(0).toUpperCase() + theme.slice(1)}
                  </button>
                )}
              </For>
            </div>
          </div>

          {/* Currency Format */}
          <div class="form-group">
            <label class="label">Currency Format</label>
            <div class="option-group horizontal">
              <For each={['gp', 'k', 'm'] as const}>
                {(currency) => (
                  <button
                    class={`option-btn small ${appSettings().currency === currency ? 'active' : ''}`}
                    onClick={() => updateAppSetting('currency', currency)}
                  >
                    {currency === 'gp' && '1,234,567 gp'}
                    {currency === 'k' && '1,234.5k'}
                    {currency === 'm' && '1.23M'}
                  </button>
                )}
              </For>
            </div>
          </div>

          {/* Auto Refresh */}
          <div class="form-group">
            <label class="toggle-label">
              <input
                type="checkbox"
                checked={appSettings().autoRefresh}
                onChange={(e) => updateAppSetting('autoRefresh', e.currentTarget.checked)}
              />
              <span>Auto-refresh data</span>
            </label>
          </div>

          {/* Compact View */}
          <div class="form-group">
            <label class="toggle-label">
              <input
                type="checkbox"
                checked={appSettings().compactView}
                onChange={(e) => updateAppSetting('compactView', e.currentTarget.checked)}
              />
              <span>Compact view</span>
            </label>
          </div>

          {/* Show Profit Percent */}
          <div class="form-group">
            <label class="toggle-label">
              <input
                type="checkbox"
                checked={appSettings().showProfitPercent}
                onChange={(e) => updateAppSetting('showProfitPercent', e.currentTarget.checked)}
              />
              <span>Show profit as percentage</span>
            </label>
          </div>

          <button
            class="btn btn-primary w-full"
            onClick={saveAppSettings}
          >
            Save App Settings
          </button>
        </div>
      </Show>

      <style>{`
        .settings {
          max-width: 600px;
          margin: 0 auto;
        }

        .settings-loading {
          display: flex;
          justify-content: center;
          padding: var(--space-6);
        }

        .settings-section {
          margin-top: var(--space-4);
        }

        .alert {
          padding: var(--space-3) var(--space-4);
          border-radius: var(--radius-md);
          margin-bottom: var(--space-4);
        }

        .alert-success {
          background-color: var(--success-light);
          color: var(--success);
        }

        .alert-error {
          background-color: var(--danger-light);
          color: var(--danger);
        }

        .rate-limit-card {
          margin-bottom: var(--space-4);
        }

        .rate-limit-header {
          display: flex;
          justify-content: space-between;
          margin-bottom: var(--space-2);
        }

        .rate-limit-title {
          font-weight: 500;
        }

        .option-group {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: var(--space-2);
        }

        .option-group.horizontal {
          grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        }

        .option-btn {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          padding: var(--space-3);
          background-color: var(--bg-tertiary);
          border: 2px solid var(--border);
          border-radius: var(--radius-md);
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .option-btn:hover {
          border-color: var(--border-light);
        }

        .option-btn.active {
          border-color: var(--accent);
          background-color: var(--accent-light);
        }

        .option-btn.small {
          padding: var(--space-2);
        }

        .option-title {
          font-weight: 600;
          margin-bottom: var(--space-1);
        }

        .option-desc {
          font-size: var(--font-size-xs);
          color: var(--text-muted);
        }

        .toggle-label {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          cursor: pointer;
        }

        .toggle-label input {
          width: 18px;
          height: 18px;
          cursor: pointer;
        }

        @media (max-width: 480px) {
          .option-group {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
}
