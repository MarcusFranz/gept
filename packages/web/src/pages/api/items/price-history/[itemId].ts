import type { APIRoute } from 'astro';
import { cache, cacheKey, TTL, KEY } from '../../../../lib/cache';

const PREDICTION_API = import.meta.env.PREDICTION_API;
const API_KEY = import.meta.env.PREDICTION_API_KEY;

interface PriceHistoryCache {
  prices: number[];
  trend: string;
}

export const GET: APIRoute = async ({ params, locals }) => {
  const itemId = parseInt(params.itemId || '');

  if (!itemId || isNaN(itemId)) {
    return new Response(JSON.stringify({
      success: false,
      error: 'Invalid item ID'
    }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    });
  }

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
    return new Response(JSON.stringify({
      success: false,
      error: 'Service misconfigured'
    }), {
      status: 503,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  try {
    const redisCacheKey = cacheKey(KEY.ITEM_PRICE, 'history', itemId.toString());

    // Try cache first
    let data: PriceHistoryCache | null = null;
    try {
      data = await cache.get<PriceHistoryCache>(redisCacheKey);
    } catch (err) {
      console.warn('[PriceHistory] Cache read failed:', (err as Error)?.message);
    }

    if (!data) {
      const engineRes = await fetch(
        `${PREDICTION_API}/api/v1/items/${itemId}/price-history?hours=24`,
        {
          headers: {
            ...(API_KEY ? { 'X-API-Key': API_KEY } : {})
          }
        }
      );

      if (!engineRes.ok) {
        throw new Error(`Engine returned ${engineRes.status}`);
      }

      const raw = await engineRes.json();

      // Simplify extended price points to midpoint array for sparklines
      const prices: number[] = (raw.history ?? []).map(
        (pt: { high: number; low: number }) => Math.round((pt.high + pt.low) / 2)
      );

      data = {
        prices,
        trend: raw.trend ?? 'Stable',
      };

      // Cache for 5 minutes (prices update every ~5 min)
      cache.set(redisCacheKey, data, TTL.PRICE_HISTORY).catch((err) => {
        console.warn('[PriceHistory] Cache write failed:', err?.message);
      });
    }

    return new Response(JSON.stringify({
      success: true,
      data
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error(`[PriceHistory/${itemId}] API Error:`, error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to fetch price history'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
