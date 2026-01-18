import type { Recommendation, UpdateRecommendation } from '../../lib/types';

export interface SlotSettings {
  style: 'passive' | 'hybrid' | 'active';
  risk: 'low' | 'medium' | 'high';
  margin: 'conservative' | 'moderate' | 'aggressive';
}

export interface OrderState {
  isOrdered: boolean;
  phase: 'buying' | 'selling';
  quantity: number;
  filled: number;
  sold: number;
  recommendation: Recommendation | null;
  settings: SlotSettings;
  originalSettings: SlotSettings | null;
  showSettings: boolean;
  pendingUpdate: UpdateRecommendation | null;
  acceptedUpdate: UpdateRecommendation | null;
  lastCheckedAt: string | null;
  isCustomItem: boolean;
  originalRecommendation: Recommendation | null;
}

export const MAX_SLOTS = 8;

export const TIER_LIMITS = {
  free: 3,
  paid: 8
} as const;

export const PLAN_KEY = 'gept-plan';
export const STORAGE_KEY = 'gept-active-trades';
export const USER_CAPITAL_KEY = 'gept-user-capital';
export const DEFAULT_SETTINGS_KEY = 'gept-default-settings';

export const defaultSlotSettings: SlotSettings = {
  style: 'hybrid',
  risk: 'medium',
  margin: 'moderate'
};

export type Tier = 'free' | 'paid';
