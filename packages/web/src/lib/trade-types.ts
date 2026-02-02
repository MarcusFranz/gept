// packages/web/src/lib/trade-types.ts
// Trade-centric UI types - replaces slot-based OrderState

import type { ActiveTrade } from './db';
import type { Recommendation } from './types';

/** Trade status for UI display */
export type TradeStatus =
  | 'on_track'      // No action needed
  | 'check_in'      // Periodic check-in due
  | 'needs_attention' // Something's off (behind schedule, price moved)
  | 'ready';        // Phase complete, tap to advance

/** Computed trade state for UI */
export interface TradeViewModel {
  id: string;
  itemId: number;
  itemName: string;
  phase: 'buying' | 'selling';
  progress: number; // 0-100
  status: TradeStatus;

  // Prices
  buyPrice: number;
  sellPrice: number;
  actualBuyPrice: number | null;
  actualSellPrice: number | null;
  suggestedSellPrice: number | null;

  // Profit
  targetProfit: number;
  quantity: number;

  // Timing
  createdAt: Date;
  expectedHours: number;
  lastCheckIn: Date | null;
  nextCheckIn: Date | null;

  // References
  recId: string | null;
  modelId: string | null;
}

/** Guidance action from system */
export type GuidanceAction = 'hold' | 'relist' | 'exit' | 'sell_now';

export interface Guidance {
  action: GuidanceAction;
  reason: string;
  params?: {
    newPrice?: number;
    priceDelta?: number;
    expectedSpeedup?: string;
  };
}

/** Check-in request payload */
export interface CheckInRequest {
  tradeId: string;
  progress: number; // 0-100
}

/** Check-in response */
export interface CheckInResponse {
  success: boolean;
  trade: ActiveTrade;
  guidance?: Guidance;
  nextCheckIn: string;
}

/** Filter state for opportunity browser */
export interface OpportunityFilters {
  profitMin?: number;
  profitMax?: number;
  timeMax?: number; // hours
  confidence?: 'low' | 'medium' | 'high';
  capitalMax?: number;
  category?: string;
}

/** Why chips - feature explanations for recommendations */
export interface WhyChip {
  icon: string;
  label: string;
  type: 'positive' | 'neutral' | 'negative';
}

/** Opportunity from the opportunities browser endpoint */
export interface Opportunity {
  id: string;
  itemId: number;
  item: string;
  iconUrl?: string;
  buyPrice: number;
  sellPrice: number;
  quantity: number;
  capitalRequired: number;
  expectedProfit: number;
  expectedHours: number;
  confidence: string;
  fillProbability: number;
  volume24h?: number;
  trend?: string;
  whyChips: WhyChip[];
  category?: string;
  modelId?: string;
}

/** Response from the opportunities endpoint */
export interface OpportunitiesResponse {
  items: Opportunity[];
  total: number;
  hasMore: boolean;
}

/** Extended recommendation with why chips */
export interface OpportunityViewModel extends Recommendation {
  whyChips: WhyChip[];
}

// Helper: Convert ActiveTrade to TradeViewModel
export function toTradeViewModel(trade: ActiveTrade): TradeViewModel {
  const now = new Date();
  const createdAt = new Date(trade.created_at);
  const nextCheckIn = trade.next_check_in ? new Date(trade.next_check_in) : null;

  // Compute status
  let status: TradeStatus = 'on_track';
  if (trade.progress >= 100) {
    status = 'ready';
  } else if (nextCheckIn && nextCheckIn <= now) {
    status = 'check_in';
  }
  // TODO: Add 'needs_attention' based on guidance endpoint

  return {
    id: trade.id,
    itemId: trade.item_id,
    itemName: trade.item_name,
    phase: trade.phase,
    progress: trade.progress,
    status,
    buyPrice: trade.buy_price,
    sellPrice: trade.sell_price,
    actualBuyPrice: trade.actual_buy_price,
    actualSellPrice: trade.actual_sell_price,
    suggestedSellPrice: trade.suggested_sell_price ?? null,
    targetProfit: (trade.sell_price - trade.buy_price) * trade.quantity,
    quantity: trade.quantity,
    createdAt,
    expectedHours: trade.expected_hours || 4, // Default 4 hours
    lastCheckIn: trade.last_check_in ? new Date(trade.last_check_in) : null,
    nextCheckIn,
    recId: trade.rec_id,
    modelId: trade.model_id,
  };
}

// Helper: Generate why chips from recommendation
export function generateWhyChips(rec: Recommendation): WhyChip[] {
  const chips: WhyChip[] = [];

  // Confidence chip
  if (rec.confidence >= 0.8) {
    chips.push({ icon: '🎯', label: 'High confidence', type: 'positive' });
  } else if (rec.confidence >= 0.6) {
    chips.push({ icon: '🎯', label: 'Med confidence', type: 'neutral' });
  }

  // Volume chip
  if (rec.volume24h > 100000) {
    chips.push({ icon: '🔥', label: 'High volume', type: 'positive' });
  } else if (rec.volume24h > 10000) {
    chips.push({ icon: '📊', label: 'Good volume', type: 'neutral' });
  }

  // Fill probability
  if (rec.fillProbability >= 0.30) {
    chips.push({ icon: '⚡', label: 'Fast fill', type: 'positive' });
  }

  // Trend chip
  if (rec.trend === 'up') {
    chips.push({ icon: '📈', label: 'Trending up', type: 'positive' });
  } else if (rec.trend === 'down') {
    chips.push({ icon: '📉', label: 'Trending down', type: 'negative' });
  }

  // Time chip
  if (rec.expectedHours <= 2) {
    chips.push({ icon: '⏱', label: 'Quick flip', type: 'positive' });
  } else if (rec.expectedHours >= 6) {
    chips.push({ icon: '🕐', label: 'Longer hold', type: 'neutral' });
  }

  return chips.slice(0, 4); // Max 4 chips
}

// LocalStorage keys for filter persistence
export const FILTER_STORAGE_KEY = 'gept-opportunity-filters';
export const CAPITAL_STORAGE_KEY = 'gept-user-capital';
