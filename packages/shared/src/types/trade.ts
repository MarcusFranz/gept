/**
 * Trade types - Contract between engine and web
 */

export interface TradeCompletion {
  tradeId: string;
  sellPrice: number;
  quantity: number;
  profit: number;
  notes?: string;
}

export interface TradeOutcomeRequest {
  userId: string;
  itemId: number;
  itemName: string;
  buyPrice: number;
  sellPrice: number;
  quantity: number;
  actualProfit: number;
  recId?: string;
  modelId?: string;
}

export interface ActiveTrade {
  id: string;
  userId: string;
  itemId: number;
  itemName: string;
  buyPrice: number;
  quantity: number;
  recId?: string;
  modelId?: string;
  createdAt: Date;
}

export interface TradeHistory {
  id: string;
  userId: string;
  itemId: number;
  itemName: string;
  buyPrice: number;
  sellPrice: number;
  quantity: number;
  profit: number;
  recId?: string;
  modelId?: string;
  completedAt: Date;
}

export type UpdateRecommendationType =
  | 'SWITCH_ITEM'
  | 'SELL_NOW'
  | 'ADJUST_PRICE'
  | 'HOLD';

export interface UpdateRecommendation {
  id: string;
  tradeId: string;
  type: UpdateRecommendationType;
  reason: string;
  confidence: number;
  urgency: 'low' | 'medium' | 'high';
  timestamp: string;
  newItem?: {
    itemId: number;
    name: string;
    buyPrice: number;
    sellPrice: number;
  };
  adjustedSellPrice?: number;
  newSellPrice?: number;
  originalSellPrice?: number;
  profitDelta?: number;
}
