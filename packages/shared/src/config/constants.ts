/**
 * Shared constants across GePT packages
 */

// Cache TTLs (seconds)
export const CACHE_TTL = {
  ITEM_PRICES: 120,         // 2 minutes
  ITEM_SEARCH: 300,         // 5 minutes
  ITEM_METADATA: 3600,      // 1 hour
  RECOMMENDATIONS: 30,      // 30 seconds
  USER_SETTINGS: 300,       // 5 minutes
  PREDICTIONS: 300,         // 5 minutes (matches inference cycle)
} as const;

// Rate limits by tier
export const RATE_LIMITS = {
  beta: { daily: 500, weekly: 2000 },
  free: { daily: 50, weekly: 200 },
  premium: { daily: 500, weekly: 2000 },
} as const;

// Capital presets for UI
export const CAPITAL_PRESETS = [
  { label: '5M', value: 5_000_000 },
  { label: '20M', value: 20_000_000 },
  { label: '50M', value: 50_000_000 },
  { label: '100M', value: 100_000_000 },
  { label: '250M', value: 250_000_000 },
] as const;

// GE Tax configuration
export const TAX = {
  RATE: 0.02,              // 2% tax
  THRESHOLD: 100,          // Items under 100gp exempt
  MAX: 5_000_000,          // Max tax 5M per trade
} as const;

// Prediction thresholds
export const PREDICTION_THRESHOLDS = {
  MIN_EV: 0.005,           // Minimum expected value
  MIN_FILL_PROB: 0.03,     // Minimum fill probability
  MAX_FILL_PROB: 0.30,     // Maximum fill probability
  DATA_STALE_SECONDS: 600, // 10 minutes
} as const;

// Time horizons by trading style (hours)
export const HOUR_RANGES = {
  active: { min: 1, max: 4 },
  hybrid: { min: 2, max: 12 },
  passive: { min: 8, max: 48 },
} as const;

// EV thresholds by risk level
export const EV_THRESHOLDS = {
  low: 0.008,
  medium: 0.005,
  high: 0.003,
} as const;

// Fill probability minimums by risk level
export const FILL_PROB_MINIMUMS = {
  low: 0.08,
  medium: 0.05,
  high: 0.03,
} as const;
