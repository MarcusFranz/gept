// Recommendation from the GePT API
export interface Recommendation {
  id: string;
  item: string;
  itemId: number;
  buyPrice: number;
  sellPrice: number;
  offsetPct?: number;
  quantity: number;
  capitalRequired: number;
  expectedProfit: number;
  confidence: number;
  fillProbability: number;
  fillConfidence: number;
  expectedHours: number;
  trend: 'up' | 'down' | 'stable';
  reason: string;
  volume24h: number;
  priceHistory?: number[];
  modelId: string;
  modelStatus?: string;
  isMultiLimitStrategy?: boolean;
  stagedBuys?: StagedBuy[];
  baseBuyLimit?: number;
}

export interface StagedBuy {
  price: number;
  quantity: number;
}

// User settings for the API call
export interface UserSettings {
  capital: number;
  style: 'passive' | 'hybrid' | 'active';
  risk: 'low' | 'medium' | 'high';
  margin: 'conservative' | 'moderate' | 'aggressive';
  excludedItems?: number[];
  useBetaModel?: boolean;
}

// Item search result
export interface ItemSearchResult {
  id: number;
  name: string;
  acronym?: string;
  icon?: string;
}

// Portfolio statistics
export interface PortfolioStats {
  totalTrades: number;
  totalProfit: number;
  winCount: number;
  lossCount: number;
  averageProfit: number;
  bestTrade: { item: string; profit: number } | null;
  worstTrade: { item: string; profit: number } | null;
  mostTradedItem: { item: string; count: number } | null;
  currentStreak: { type: 'win' | 'loss'; count: number };
  winRate: number;
}

// Rate limit info
export interface RateLimitInfo {
  daily: {
    used: number;
    limit: number;
    remaining: number;
    resetAt: string;
  };
  weekly: {
    used: number;
    limit: number;
    remaining: number;
    resetAt: string;
  };
}

// Session cache entry
export interface SessionCache {
  recommendations: Recommendation[];
  style: string;
  capital: number;
  risk: string;
  exhausted: boolean;
  seenIds: Set<string>;
  expiresAt: number;
}

// API Response types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Trade completion data
export interface TradeCompletion {
  tradeId: string;
  sellPrice: number;
  quantity: number;
  profit: number;
  notes?: string;
}

// Update recommendation types for dynamic trade reevaluation
export type UpdateRecommendationType = 'SWITCH_ITEM' | 'SELL_NOW' | 'ADJUST_PRICE' | 'HOLD';

export interface UpdateRecommendation {
  id: string;
  tradeId: string;
  type: UpdateRecommendationType;
  reason: string;
  confidence: number;
  urgency: 'low' | 'medium' | 'high';
  timestamp: string;

  // For SWITCH_ITEM - new item to switch to
  newItem?: {
    itemId: number;
    item: string;
    buyPrice: number;
    sellPrice: number;
    quantity: number;
    expectedProfit: number;
    confidence: number;
  };

  // For SELL_NOW - adjusted sell price (optional)
  adjustedSellPrice?: number;

  // For ADJUST_PRICE - price reduction details
  newSellPrice?: number;
  originalSellPrice?: number;

  // Computed impact
  profitDelta?: number;
}

export interface UpdateCheckResponse {
  updates: UpdateRecommendation[];
  nextCheckIn: number; // Suggested polling interval in seconds
}

// Toast notification types
export interface Toast {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message: string;
  duration?: number; // Auto-dismiss in ms, undefined = manual dismiss
  action?: {
    label: string;
    onClick: () => void;
  };
  createdAt: number;
}

// App settings (global)
export interface AppSettings {
  theme: 'light' | 'dark' | 'system';
  compactView: boolean;
  autoRefresh: boolean;
  refreshInterval: number; // in seconds
  showProfitPercent: boolean;
  currency: 'gp' | 'k' | 'm';
}

// Default app settings
export const defaultAppSettings: AppSettings = {
  theme: 'dark',
  compactView: false,
  autoRefresh: true,
  refreshInterval: 30,
  showProfitPercent: true,
  currency: 'm'
};

// Helper functions
export function formatGold(amount: number, format: 'gp' | 'k' | 'm' = 'm'): string {
  if (format === 'gp') return amount.toLocaleString() + ' gp';
  if (format === 'k') return (amount / 1000).toFixed(1) + 'k';
  return (amount / 1_000_000).toFixed(2) + 'M';
}

export function formatPercent(value: number): string {
  return (value * 100).toFixed(1) + '%';
}

export function generateId(): string {
  return crypto.randomUUID();
}

// Capital presets
export const capitalPresets = [
  { label: '5M', value: 5_000_000 },
  { label: '20M', value: 20_000_000 },
  { label: '50M', value: 50_000_000 },
  { label: '100M', value: 100_000_000 },
  { label: '250M', value: 250_000_000 }
];

// Rate limits by tier
export const rateLimits: Record<string, { daily: number; weekly: number }> = {
  beta: { daily: 500, weekly: 2000 },    // Beta users get premium limits
  free: { daily: 50, weekly: 200 },
  premium: { daily: 500, weekly: 2000 }
};

// Webhook event types for trade lifecycle
export type WebhookEventType = 'TRADE_CREATED' | 'TRADE_COMPLETED' | 'TRADE_CANCELLED' | 'TRADE_UPDATED';

export interface TradeEventPayload {
  itemId: number;
  itemName: string;
  buyPrice: number;
  sellPrice: number | null;
  offsetPct?: number | null;
  quantity: number;
  profit?: number;
  recId?: string | null;
  modelId?: string | null;
  expectedHours?: number | null;
  createdAt?: string | null;
}

export interface WebhookEvent {
  eventType: WebhookEventType;
  timestamp: string;
  userId: string;
  tradeId: string;
  payload: TradeEventPayload;
}

// SSE message types for real-time alerts
export type SSEMessageType = 'ALERT' | 'CONNECTED' | 'HEARTBEAT';

export interface SSEMessage {
  type: SSEMessageType;
  data?: UpdateRecommendation;
  timestamp?: string;
}

// User feedback types
export type FeedbackType = 'bug' | 'feature' | 'general' | 'recommendation';
export type FeedbackRating = 'positive' | 'negative';

export interface GeneralFeedback {
  id: string;
  userId: string;
  type: Exclude<FeedbackType, 'recommendation'>;
  message: string;
  email?: string;
  createdAt: string;
}

export interface RecommendationFeedback {
  id: string;
  userId: string;
  type: 'recommendation';
  rating: FeedbackRating;
  message?: string;
  // Snapshot of the recommendation at the time of feedback
  recommendation: {
    id: string;
    itemId: number;
    item: string;
    buyPrice: number;
    sellPrice: number;
    quantity: number;
    expectedProfit: number;
    confidence: number;
    modelId: string;
  };
  createdAt: string;
}
