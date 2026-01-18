/**
 * Webhook types - Contract for trade event notifications
 */

export type WebhookEventType =
  | 'TRADE_CREATED'
  | 'TRADE_COMPLETED'
  | 'TRADE_CANCELLED'
  | 'TRADE_UPDATED';

export interface TradeEventPayload {
  itemId: number;
  itemName: string;
  buyPrice: number;
  sellPrice: number | null;
  quantity: number;
  profit?: number;
  recId?: string | null;
  modelId?: string | null;
}

export interface WebhookEvent {
  eventType: WebhookEventType;
  timestamp: string;
  userId: string;
  tradeId: string;
  payload: TradeEventPayload;
}

export interface WebhookResponse {
  success: boolean;
  message?: string;
}
