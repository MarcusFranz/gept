import type { APIRoute } from 'astro';
import type { UpdateCheckResponse, UpdateRecommendation } from '../../../lib/types';
import { activeTradesRepo } from '../../../lib/repositories';
import { cache, cacheKey, TTL, KEY } from '../../../lib/cache';

const PREDICTION_API = process.env.PREDICTION_API || 'http://localhost:8000';
const API_KEY = process.env.PREDICTION_API_KEY || '';

type EngineOrderUpdateResponse = {
  action: 'adjust_price' | 'wait' | 'abort_retry' | 'liquidate';
  confidence: number;
  reasoning: string;
  recommendations?: {
    adjust_price?: {
      suggested_price: number;
      new_fill_probability: number;
      cost_difference: number;
    };
    abort_retry?: {
      alternative_items: Array<{
        item_id: number;
        item_name: string;
        expected_profit: number;
        fill_probability: number;
        expected_hours: number;
      }>;
    };
    liquidate?: {
      instant_price: number;
      loss_amount: number;
    };
  };
};

const getUrgency = (confidence: number): UpdateRecommendation['urgency'] => {
  if (confidence >= 0.85) return 'high';
  if (confidence >= 0.65) return 'medium';
  return 'low';
};

const getTimeElapsedMinutes = (createdAt: unknown): number => {
  const ts = new Date(createdAt as any).getTime();
  if (!Number.isFinite(ts)) return 0;
  return Math.max(0, Math.round((Date.now() - ts) / (1000 * 60)));
};

export const GET: APIRoute = async ({ locals }) => {
  if (!locals.user) {
    return new Response(JSON.stringify({
      success: false,
      error: 'Unauthorized'
    }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  const userId = locals.user.id;

  try {
    // Price alerts currently only support revising SELL offers. (PATCH /trades/active/:id
    // updates sellPrice, not buyPrice.) So we only evaluate trades in the selling phase.
    const trades = await activeTradesRepo.findByUserId(userId);
    const sellTrades = trades.filter(t => t.phase === 'selling');

    if (sellTrades.length === 0) {
      const response: UpdateCheckResponse = { updates: [], nextCheckIn: 60 };
      return new Response(JSON.stringify({ success: true, data: response }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const redisKey = cacheKey(KEY.SSE, 'updates', userId);

    // Try cache first (30s TTL)
    const cached = await cache.get<UpdateCheckResponse>(redisKey);
    if (cached) {
      return new Response(JSON.stringify({ success: true, data: cached }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (API_KEY) headers['X-API-Key'] = API_KEY;

    // Evaluate each sell trade using the engine's order-evaluation endpoint, passing
    // the user's actual sell offer price and time-in-trade. This avoids relying on
    // rec_id (which may be stale/rehydrated) and fixes long-running trades never
    // producing updates.
    const evaluated = await Promise.all(sellTrades.map(async (trade) => {
      try {
        const timeElapsedMinutes = getTimeElapsedMinutes(trade.created_at);
        const engineRes = await fetch(
          `${PREDICTION_API}/api/v1/recommendations/update`,
          {
            method: 'POST',
            headers,
            body: JSON.stringify({
              item_id: trade.item_id,
              order_type: 'sell',
              user_price: trade.sell_price,
              quantity: trade.quantity,
              time_elapsed_minutes: timeElapsedMinutes,
            }),
            signal: AbortSignal.timeout(5000)
          }
        );

        if (!engineRes.ok) {
          console.error(`[TradeUpdates] Engine returned ${engineRes.status}`);
          return null;
        }

        const data = await engineRes.json() as EngineOrderUpdateResponse;
        const confidence = Number(data.confidence) || 0;
        const urgency = getUrgency(confidence);

        if (data.action === 'wait') return null;

        const base: Omit<UpdateRecommendation, 'type'> = {
          id: `update_${trade.id}_${Date.now()}`,
          tradeId: trade.id,
          reason: data.reasoning || 'Trade update available',
          confidence,
          urgency,
          timestamp: new Date().toISOString(),
        };

        if (data.action === 'adjust_price' && data.recommendations?.adjust_price?.suggested_price) {
          const newSellPrice = Math.round(data.recommendations.adjust_price.suggested_price);
          const profitDelta = (newSellPrice - trade.sell_price) * trade.quantity;
          return {
            ...base,
            type: 'ADJUST_PRICE' as const,
            newSellPrice,
            originalSellPrice: trade.sell_price,
            profitDelta,
          };
        }

        if (data.action === 'liquidate' && data.recommendations?.liquidate?.instant_price) {
          const adjustedSellPrice = Math.round(data.recommendations.liquidate.instant_price);
          return {
            ...base,
            type: 'SELL_NOW' as const,
            adjustedSellPrice,
          };
        }

        if (data.action === 'abort_retry') {
          return { ...base, type: 'SWITCH_ITEM' as const };
        }

        return null;
      } catch (err) {
        console.error('[TradeUpdates] Trade eval error:', (err as Error)?.message);
        return null;
      }
    }));

    const updates = evaluated.filter((u): u is UpdateRecommendation => Boolean(u));

    // Calculate next check-in interval based on update urgency
    const response: UpdateCheckResponse = {
      updates,
      nextCheckIn: updates.some(u => u.urgency === 'high') ? 15 : updates.length > 0 ? 30 : 60,
    };

    // Cache briefly
    cache.set(redisKey, response, TTL.RECOMMENDATIONS).catch(() => {});

    return new Response(JSON.stringify({ success: true, data: response }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('[TradeUpdates] Error:', error);
    const response: UpdateCheckResponse = { updates: [], nextCheckIn: 30 };
    return new Response(JSON.stringify({ success: true, data: response }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
