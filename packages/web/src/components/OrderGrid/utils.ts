import type { OrderState, SlotSettings } from './types';
import {
  MAX_SLOTS,
  STORAGE_KEY,
  USER_CAPITAL_KEY,
  DEFAULT_SETTINGS_KEY,
  PLAN_KEY,
  defaultSlotSettings,
} from './types';

// Format number in RuneScape style with k/m/b suffixes
export const formatRsNumber = (value: number, sigFigCount: number = 5): string => {
  const toSigFigs = (n: number, digits: number): number => {
    if (n === 0) return 0;
    const magnitude = Math.floor(Math.log10(Math.abs(n))) + 1;
    const scale = Math.pow(10, digits - magnitude);
    return Math.round(n * scale) / scale;
  };

  const rounded = toSigFigs(value, sigFigCount);

  if (rounded >= 1_000_000_000) {
    return (rounded / 1_000_000_000).toLocaleString(undefined, { maximumFractionDigits: 3 }) + 'b';
  } else if (rounded >= 1_000_000) {
    return (rounded / 1_000_000).toLocaleString(undefined, { maximumFractionDigits: 3 }) + 'm';
  } else if (rounded >= 1_000) {
    return (rounded / 1_000).toLocaleString(undefined, { maximumFractionDigits: 3 }) + 'k';
  }
  return rounded.toLocaleString();
};

// Format price with 5 sig figs
export const formatRsPrice = (price: number): string => formatRsNumber(price, 5);

export function getPlanTier(): 'free' | 'paid' {
  // Beta: everyone gets full access
  return 'paid';
}

// Load user capital from localStorage
export function loadUserCapital(): number {
  if (typeof window === 'undefined') return 50_000_000;
  const saved = localStorage.getItem(USER_CAPITAL_KEY);
  return saved ? parseInt(saved, 10) : 50_000_000;
}

// Save user capital to localStorage
export function saveUserCapital(capital: number): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(USER_CAPITAL_KEY, capital.toString());
}

// Load default flip settings from localStorage
export function loadDefaultSettings(): SlotSettings {
  if (typeof window === 'undefined') return defaultSlotSettings;
  const saved = localStorage.getItem(DEFAULT_SETTINGS_KEY);
  if (saved) {
    try {
      return { ...defaultSlotSettings, ...JSON.parse(saved) };
    } catch {
      return defaultSlotSettings;
    }
  }
  return defaultSlotSettings;
}

// Save default settings to localStorage
export function saveDefaultSettings(settings: SlotSettings): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(DEFAULT_SETTINGS_KEY, JSON.stringify(settings));
}

// Create empty slots array
export function createEmptySlots(): OrderState[] {
  // Use user's saved default settings, not hardcoded defaults
  const savedDefaults = loadDefaultSettings();
  return Array(MAX_SLOTS).fill(null).map(() => ({
    isOrdered: false,
    phase: 'buying' as const,
    quantity: 0,
    filled: 0,
    sold: 0,
    recommendation: null,
    settings: { ...savedDefaults },
    originalSettings: null,
    showSettings: false,
    pendingUpdate: null,
    acceptedUpdate: null,
    lastCheckedAt: null,
    isCustomItem: false,
    originalRecommendation: null
  }));
}

// Load slots from localStorage
export function loadSavedSlots(): OrderState[] {
  if (typeof window === 'undefined') return createEmptySlots();

  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return createEmptySlots();

    const parsed = JSON.parse(saved);
    if (!Array.isArray(parsed)) return createEmptySlots();

    // Use user's saved default settings for slots without settings
    const savedDefaults = loadDefaultSettings();

    return Array(MAX_SLOTS).fill(null).map((_, i) => {
      const slot = parsed[i] as Partial<OrderState> | undefined;
      const isOrdered = slot?.isOrdered ?? false;
      const recommendation = isOrdered ? (slot?.recommendation ?? null) : null;
      return {
        isOrdered: isOrdered && recommendation !== null,
        phase: slot?.phase ?? 'buying',
        quantity: slot?.quantity ?? 0,
        filled: slot?.filled ?? 0,
        sold: slot?.sold ?? 0,
        recommendation,
        settings: slot?.settings ?? { ...savedDefaults },
        originalSettings: null,
        showSettings: false,
        pendingUpdate: null,
        acceptedUpdate: null,
        lastCheckedAt: null,
        isCustomItem: isOrdered ? (slot?.isCustomItem ?? false) : false,
        originalRecommendation: isOrdered ? (slot?.originalRecommendation ?? null) : null
      };
    });
  } catch {
    return createEmptySlots();
  }
}

// Save slots to localStorage
export function saveSlots(slots: OrderState[]): void {
  if (typeof window === 'undefined') return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(slots));
}

// Item image URL helper
export function itemImageUrl(itemId: number): string {
  return `https://chisel.weirdgloop.org/static/img/osrs-sprite/${itemId}.png`;
}

// Calculate fill percentage
export function fillPercent(slot: OrderState): number {
  if (slot.quantity === 0) return 0;
  return (slot.filled / slot.quantity) * 100;
}

// Calculate sell percentage
export function sellPercent(slot: OrderState): number {
  if (slot.sold === 0) return 0;
  return (slot.sold / slot.filled) * 100;
}

// Check if slot settings have changed from original
export function slotSettingsChanged(slot: OrderState): boolean {
  if (!slot.originalSettings) return false;
  return (
    slot.settings.style !== slot.originalSettings.style ||
    slot.settings.risk !== slot.originalSettings.risk ||
    slot.settings.margin !== slot.originalSettings.margin
  );
}
