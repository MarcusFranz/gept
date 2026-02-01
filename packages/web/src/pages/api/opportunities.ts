// packages/web/src/pages/api/opportunities.ts
import type { APIRoute } from 'astro';
import { cache, cacheKey, TTL, KEY } from '../../lib/cache';

const PREDICTION_API = import.meta.env.PREDICTION_API;
const API_KEY = import.meta.env.PREDICTION_API_KEY;

export const POST: APIRoute = async ({ request, locals }) => {
  // Require authentication
  if (!locals.user) {
    return new Response(JSON.stringify({
      success: false,
      error: 'Unauthorized'
    }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  if (!PREDICTION_API) {
    console.error('[Opportunities] PREDICTION_API env var is not configured');
    return new Response(JSON.stringify({
      success: false,
      error: 'Service misconfigured'
    }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  try {
    // Parse filter parameters from body
    const filters = await request.json();

    // Build the engine request payload
    const enginePayload = {
      min_profit: filters.profitMin,
      max_profit: filters.profitMax,
      min_hours: filters.timeMin,
      max_hours: filters.timeMax,
      confidence: filters.confidence, // array of levels
      max_capital: filters.capitalMax,
      categories: filters.categories,
      limit: filters.limit || 50,
      offset: filters.offset || 0
    };

    // Build cache key from all filter parameters that affect the response
    const redisCacheKey = cacheKey(
      KEY.OPPS,
      enginePayload.min_profit ?? '',
      enginePayload.max_profit ?? '',
      enginePayload.min_hours ?? '',
      enginePayload.max_hours ?? '',
      (enginePayload.confidence ?? []).join(','),
      enginePayload.max_capital ?? '',
      (enginePayload.categories ?? []).join(','),
      enginePayload.limit,
      enginePayload.offset
    );

    // Try cache first
    let responseData: any = null;
    try {
      responseData = await cache.get(redisCacheKey);
    } catch {
      // Continue without cache on Redis errors
    }

    if (!responseData) {
      // Call engine opportunities endpoint
      const engineRes = await fetch(`${PREDICTION_API}/api/v1/opportunities`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(API_KEY ? { 'X-API-Key': API_KEY } : {})
        },
        body: JSON.stringify(enginePayload)
      });

      if (!engineRes.ok) {
        throw new Error(`Engine returned ${engineRes.status}`);
      }

      const data = await engineRes.json();

      // Transform to frontend format.
      // Defence in depth: coerce nullable fields so the UI never receives
      // unexpected types (e.g. NaN from pandas serialization edge cases).
      responseData = {
        items: data.items.map((item: any) => ({
          id: item.id,
          itemId: item.item_id,
          item: item.item_name ?? `Item ${item.item_id}`,
          iconUrl: item.icon_url,
          buyPrice: item.buy_price,
          sellPrice: item.sell_price,
          quantity: item.quantity,
          capitalRequired: item.capital_required,
          expectedProfit: item.expected_profit,
          expectedHours: item.expected_hours,
          confidence: item.confidence ?? 'medium',
          fillProbability: item.fill_probability,
          volume24h: Number.isFinite(item.volume_24h) ? item.volume_24h : null,
          trend: item.trend ?? 'Stable',
          whyChips: item.why_chips ?? [],
          category: item.category,
          modelId: item.model_id
        })),
        total: data.total,
        hasMore: data.has_more
      };

      // Cache the transformed response (fire and forget)
      cache.set(redisCacheKey, responseData, TTL.OPPORTUNITIES).catch(() => {});
    }

    return new Response(JSON.stringify({
      success: true,
      data: responseData
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('[Opportunities] API Error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to fetch opportunities'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
