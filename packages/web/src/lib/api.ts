import type { Recommendation, UserSettings, ItemSearchResult } from './types';
import { cache, cacheKey, bucketCapital, TTL, KEY } from './cache';

function getApiBase(): string {
  // Server-side: use process.env directly (Vercel injects at runtime)
  const url = process.env.PREDICTION_API;
  if (url) return url;

  return 'http://localhost:8000';
}

function getApiKey(): string {
  return process.env.PREDICTION_API_KEY || '';
}
const TIMEOUT_MS = 5000;
const MAX_RETRIES = 3;

// Circuit breaker state
let circuitOpen = false;
let circuitOpenTime = 0;
const CIRCUIT_RESET_MS = 30000;

// API response types (what the server returns - already camelCase)
interface ApiRecommendation {
  id: string;
  item: string;
  itemId: number;
  buyPrice: number;
  sellPrice: number;
  quantity: number;
  capitalRequired: number;
  expectedProfit: number;
  confidence: 'high' | 'medium' | 'low';
  fillProbability: number;
  fillConfidence: string; // "Strong", "Fair", etc.
  expectedHours: number;
  trend: 'Rising' | 'Stable' | 'Falling';
  reason: string;
  volume24h: number;
  priceHistory?: number[] | null;
  modelId?: number | null;
  isMultiLimitStrategy?: boolean;
}

interface ApiSearchResult {
  id: number;
  name: string;
  acronym?: string;
  icon?: string;
}

// Transform API response to our internal format
function transformRecommendation(api: ApiRecommendation): Recommendation {
  const confidenceMap: Record<string, number> = {
    high: 0.85,
    medium: 0.70,
    low: 0.55
  };

  const trendMap: Record<string, 'up' | 'down' | 'stable'> = {
    Rising: 'up',
    Falling: 'down',
    Stable: 'stable'
  };

  const fillConfidenceMap: Record<string, number> = {
    Strong: 0.90,
    Fair: 0.70,
    Weak: 0.50
  };

  return {
    id: api.id,
    item: api.item,
    itemId: api.itemId,
    buyPrice: api.buyPrice,
    sellPrice: api.sellPrice,
    quantity: api.quantity,
    capitalRequired: api.capitalRequired,
    expectedProfit: api.expectedProfit,
    confidence: confidenceMap[api.confidence] || 0.70,
    fillProbability: api.fillProbability,
    fillConfidence: fillConfidenceMap[api.fillConfidence] || 0.70,
    expectedHours: api.expectedHours,
    trend: trendMap[api.trend] || 'stable',
    reason: api.reason,
    volume24h: api.volume24h,
    priceHistory: api.priceHistory || undefined,
    modelId: api.modelId?.toString() || 'unknown',
    isMultiLimitStrategy: api.isMultiLimitStrategy
  };
}

function getAuthHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };
  const apiKey = getApiKey();
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  return headers;
}

async function fetchWithRetry<T>(
  url: string,
  options: RequestInit = {},
  retries = MAX_RETRIES
): Promise<T> {
  // Check circuit breaker
  if (circuitOpen) {
    if (Date.now() - circuitOpenTime > CIRCUIT_RESET_MS) {
      circuitOpen = false;
    } else {
      throw new Error('Service temporarily unavailable');
    }
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        ...getAuthHeaders(),
        ...options.headers
      }
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);

    if (retries > 0 && !(error instanceof Error && error.name === 'AbortError')) {
      // Exponential backoff: 100ms, 200ms, 400ms
      await new Promise(resolve => setTimeout(resolve, 100 * Math.pow(2, MAX_RETRIES - retries)));
      return fetchWithRetry<T>(url, options, retries - 1);
    }

    // Open circuit on failure
    circuitOpen = true;
    circuitOpenTime = Date.now();
    throw error;
  }
}

// Hash user ID for privacy
function hashUserId(userId: string): string {
  // Simple hash for client-side - server should use proper crypto
  let hash = 0;
  for (let i = 0; i < userId.length; i++) {
    const char = userId.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
}

// Get margin offset based on preference
function getMarginParams(margin: 'conservative' | 'moderate' | 'aggressive'): Record<string, number> {
  switch (margin) {
    case 'conservative':
      return { max_offset_pct: 0.015 };
    case 'aggressive':
      return { min_offset_pct: 0.0225 };
    default:
      return {};
  }
}

export async function getRecommendations(
  userId: string,
  settings: UserSettings,
  count = 8
): Promise<Recommendation[]> {
  // Bypass cache temporarily to debug
  const params = new URLSearchParams({
    user_id: hashUserId(userId),
    capital: settings.capital.toString(),
    style: settings.style,
    risk: settings.risk,
    slots: count.toString(),
    ...getMarginParams(settings.margin)
  });

  if (settings.excludedItems?.length) {
    params.append('exclude_item_ids', settings.excludedItems.join(','));
  }

  const url = `${getApiBase()}/api/v1/recommendations?${params}`;
  const apiResponse = await fetchWithRetry<ApiRecommendation[]>(url);
  return apiResponse.map(transformRecommendation);
}

export async function getRecommendationByItemId(
  itemId: number,
  userId: string,
  settings: UserSettings
): Promise<Recommendation | null> {
  // Cache by item ID + settings bucket (not user-specific)
  const capitalBucket = bucketCapital(settings.capital);
  const key = cacheKey(
    KEY.ITEM_PRICE,
    itemId.toString(),
    settings.style,
    settings.risk,
    capitalBucket
  );

  try {
    return await cache.getOrFetch(
      key,
      async () => {
        const params = new URLSearchParams({
          user_id: hashUserId(userId),
          capital: settings.capital.toString(),
          style: settings.style,
          risk: settings.risk
        });
        const url = `${getApiBase()}/api/v1/recommendations/item/${itemId}?${params}`;
        const apiResponse = await fetchWithRetry<ApiRecommendation>(url);
        return transformRecommendation(apiResponse);
      },
      TTL.ITEM_PRICES
    );
  } catch {
    return null;
  }
}

export async function searchItems(query: string, limit = 15): Promise<ItemSearchResult[]> {
  // Cache search results - same query = same results for all users
  const key = cacheKey(KEY.ITEM_SEARCH, query.toLowerCase(), limit.toString());

  try {
    return await cache.getOrFetch(
      key,
      async () => {
        const params = new URLSearchParams({
          q: query,
          limit: limit.toString()
        });
        const url = `${getApiBase()}/api/v1/items/search?${params}`;
        return fetchWithRetry<ItemSearchResult[]>(url);
      },
      TTL.ITEM_SEARCH
    );
  } catch {
    return [];
  }
}

export async function reportTradeOutcome(data: {
  userId: string;
  itemId: number;
  itemName: string;
  buyPrice: number;
  sellPrice: number;
  quantity: number;
  actualProfit: number;
  recId?: string;
  modelId?: string;
}): Promise<boolean> {
  // Trade outcome is reported to the recommendation endpoint
  if (!data.recId) {
    // Can't report without a recommendation ID
    return false;
  }

  const url = `${getApiBase()}/api/v1/recommendations/${data.recId}/outcome`;

  try {
    await fetchWithRetry(url, {
      method: 'POST',
      body: JSON.stringify({
        user_id: hashUserId(data.userId),
        item_id: data.itemId,
        item_name: data.itemName,
        buy_price: data.buyPrice,
        sell_price: data.sellPrice,
        quantity: data.quantity,
        actual_profit: data.actualProfit,
        model_id: data.modelId ? parseInt(data.modelId) : undefined,
        reported_at: new Date().toISOString()
      })
    });
    return true;
  } catch {
    return false;
  }
}

// Check if API is available
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${getApiBase()}/api/v1/health`, {
      method: 'GET',
      headers: getAuthHeaders(),
      signal: AbortSignal.timeout(2000)
    });
    return response.ok;
  } catch {
    return false;
  }
}
