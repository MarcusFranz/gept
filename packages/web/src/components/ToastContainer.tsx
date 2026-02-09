import { createSignal, For, Show, onCleanup } from 'solid-js';
import type { Toast } from '../lib/types';
import { generateId } from '../lib/types';
import './ToastContainer.css';

// Module-level signals for toast state (shared across components)
const [toasts, setToasts] = createSignal<Toast[]>([]);
const toastTimeouts = new Map<string, ReturnType<typeof setTimeout>>();

const DEFAULT_DURATION_MS: Record<Toast['type'], number | undefined> = {
  success: 3200,
  info: 4200,
  warning: 6500,
  error: undefined, // keep errors sticky unless caller opts in
};

const MAX_TOASTS = 4;

// Export functions for adding/removing toasts
export function addToast(toast: Omit<Toast, 'id' | 'createdAt'>): string {
  const duration = toast.duration ?? DEFAULT_DURATION_MS[toast.type];

  let id = generateId();

  setToasts(prev => {
    // Replace existing toast with same key rather than stacking.
    if (toast.key) {
      const existingIdx = prev.findIndex(t => t.key === toast.key);
      if (existingIdx !== -1) {
        const existing = prev[existingIdx];
        id = existing.id;
        const next = [...prev];
        next[existingIdx] = {
          ...existing,
          ...toast,
          id: existing.id,
          createdAt: Date.now(),
        };
        return next;
      }
    }

    const nextToast: Toast = {
      ...toast,
      id,
      createdAt: Date.now(),
    };

    const next = [...prev, nextToast];

    // Cap stack size to avoid "toast tower" during rapid actions.
    if (next.length > MAX_TOASTS) {
      const overflow = next.length - MAX_TOASTS;
      const toDrop = next.slice(0, overflow);
      // Clear any timers for dropped toasts.
      for (const t of toDrop) {
        const timer = toastTimeouts.get(t.id);
        if (timer) clearTimeout(timer);
        toastTimeouts.delete(t.id);
      }
      return next.slice(overflow);
    }

    return next;
  });

  // (Re)schedule auto-dismiss.
  const existingTimer = toastTimeouts.get(id);
  if (existingTimer) {
    clearTimeout(existingTimer);
    toastTimeouts.delete(id);
  }

  if (typeof duration === 'number' && duration > 0) {
    toastTimeouts.set(
      id,
      setTimeout(() => removeToast(id), duration)
    );
  }

  return id;
}

export function removeToast(id: string) {
  const timer = toastTimeouts.get(id);
  if (timer) clearTimeout(timer);
  toastTimeouts.delete(id);
  setToasts(prev => prev.filter(t => t.id !== id));
}

export function clearAllToasts() {
  for (const timer of toastTimeouts.values()) clearTimeout(timer);
  toastTimeouts.clear();
  setToasts([]);
}

// Toast Container component
export default function ToastContainer() {
  onCleanup(() => {
    // If this component ever unmounts, ensure we don't leak timers.
    for (const timer of toastTimeouts.values()) clearTimeout(timer);
    toastTimeouts.clear();
  });

  return (
    <div class="toast-container">
      <For each={toasts()}>
        {(toast) => (
          <div class={`toast toast-${toast.type}`}>
            <div class="toast-icon">
              <Show when={toast.type === 'warning'}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
                  <line x1="12" y1="9" x2="12" y2="13"/>
                  <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
              </Show>
              <Show when={toast.type === 'success'}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/>
                  <polyline points="22 4 12 14.01 9 11.01"/>
                </svg>
              </Show>
              <Show when={toast.type === 'error'}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="15" y1="9" x2="9" y2="15"/>
                  <line x1="9" y1="9" x2="15" y2="15"/>
                </svg>
              </Show>
              <Show when={toast.type === 'info'}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="16" x2="12" y2="12"/>
                  <line x1="12" y1="8" x2="12.01" y2="8"/>
                </svg>
              </Show>
            </div>
            <div class="toast-content">
              <div class="toast-title">{toast.title}</div>
              <div class="toast-message">{toast.message}</div>
            </div>
            <Show when={toast.action}>
              <button class="toast-action" onClick={toast.action!.onClick}>
                {toast.action!.label}
              </button>
            </Show>
            <button class="toast-dismiss" aria-label="Dismiss notification" onClick={() => removeToast(toast.id)}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </div>
        )}
      </For>
    </div>
  );
}
