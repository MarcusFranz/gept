/**
 * Recommendation types - Contract between engine and web
 *
 * These types define the API contract. Changes here require
 * coordination between engine and web agents.
 */

export interface Recommendation {
  /** Stable ID: rec_{itemId}_{YYYYMMDDHH} */
  id: string;

  /** OSRS item ID */
  itemId: number;

  /** Item name */
  item: string;

  /** Recommended buy price in GP */
  buyPrice: number;

  /** Recommended sell price in GP */
  sellPrice: number;

  /** Recommended quantity */
  quantity: number;

  /** Total capital required (buyPrice * quantity) */
  capitalRequired: number;

  /** Expected profit after tax */
  expectedProfit: number;

  /** Confidence level based on model AUC */
  confidence: ConfidenceLevel;

  /** Raw fill probability from model [0, 1] */
  fillProbability: number;

  /** Human-readable fill confidence */
  fillConfidence: FillConfidence;

  /** Price trend direction */
  trend: TrendDirection;

  /** Expected hours to complete trade */
  expectedHours: number;

  /** 24-hour trading volume */
  volume24h: number;

  /** Optional price history for sparkline */
  priceHistory?: number[];

  /** Model version that generated this prediction */
  modelId: string;

  /** Model lifecycle status */
  modelStatus?: ModelStatus;

  /** Multi-limit strategy indicator */
  isMultiLimitStrategy?: boolean;

  /** Staged buy orders for multi-limit */
  stagedBuys?: StagedBuy[];

  /** Base buy limit for the item */
  baseBuyLimit?: number;
}

export interface StagedBuy {
  price: number;
  quantity: number;
}

export type ConfidenceLevel = 'high' | 'medium' | 'low';
export type FillConfidence = 'Strong' | 'Good' | 'Fair';
export type TrendDirection = 'up' | 'down' | 'stable';
export type ModelStatus = 'ACTIVE' | 'DEPRECATED' | 'SUNSET' | 'ARCHIVED';

export interface UserSettings {
  /** Total capital in GP */
  capital: number;

  /** Trading style affects time horizons */
  style: TradingStyle;

  /** Risk tolerance affects thresholds */
  risk: RiskLevel;

  /** Margin preference */
  margin: MarginPreference;

  /** Item IDs to exclude from recommendations */
  excludedItems?: number[];
}

export type TradingStyle = 'passive' | 'hybrid' | 'active';
export type RiskLevel = 'low' | 'medium' | 'high';
export type MarginPreference = 'conservative' | 'moderate' | 'aggressive';

export interface RecommendationRequest {
  userId: string;
  capital: number;
  style: TradingStyle;
  risk: RiskLevel;
  slots: number;
  excludeIds?: string[];
  excludeItemIds?: number[];
}

export interface ItemSearchResult {
  itemId: number;
  name: string;
  icon?: string;
}
