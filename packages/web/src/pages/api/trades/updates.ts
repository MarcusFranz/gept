import type { APIRoute } from 'astro';
import type { UpdateCheckResponse, UpdateRecommendation } from '../../../lib/types';
import { activeTradesRepo } from '../../../lib/repositories';
import { cache, cacheKey, TTL, KEY } from '../../../lib/cache';

const PREDICTION_API = process.env.PREDICTION_API || 'http://localhost:8000';
const API_KEY = process.env.PREDICTION_API_KEY || '';

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
    // Get active trades that have engine rec_ids
    const trades = await activeTradesRepo.findByUserId(userId);
    const tradesWithRecId = trades.filter(t => t.rec_id);

    if (tradesWithRecId.length === 0) {
      const response: UpdateCheckResponse = { updates: [], nextCheckIn: 60 };
      return new Response(JSON.stringify({ success: true, data: response }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const recIds = tradesWithRecId.map(t => t.rec_id!);
    const redisKey = cacheKey(KEY.SSE, 'updates', userId);

    // Try cache first (30s TTL)
    const cached = await cache.get<UpdateCheckResponse>(redisKey);
    if (cached) {
      return new Response(JSON.stringify({ success: true, data: cached }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Call engine batch endpoint
    const url = `${PREDICTION_API}/api/v1/trades/updates?tradeIds=${recIds.join(',')}`;
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (API_KEY) headers['X-API-Key'] = API_KEY;

    const engineRes = await fetch(url, {
      headers,
      signal: AbortSignal.timeout(5000)
    });

    if (!engineRes.ok) {
      console.error(`[TradeUpdates] Engine returned ${engineRes.status}`);
      const response: UpdateCheckResponse = { updates: [], nextCheckIn: 30 };
      return new Response(JSON.stringify({ success: true, data: response }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const engineData = await engineRes.json() as { updates: UpdateRecommendation[]; nextCheckIn?: number };

    // Map engine rec_ids back to our active trade IDs so the client can match alerts to trades
    const recIdToTradeId = new Map(tradesWithRecId.map(t => [t.rec_id!, t.id]));
    const mappedUpdates: UpdateRecommendation[] = engineData.updates.map(u => ({
      ...u,
      tradeId: recIdToTradeId.get(u.tradeId) || u.tradeId
    }));

    const response: UpdateCheckResponse = {
      updates: mappedUpdates,
      nextCheckIn: engineData.nextCheckIn ?? 30
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
