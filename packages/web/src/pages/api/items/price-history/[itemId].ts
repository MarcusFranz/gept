import type { APIRoute } from 'astro';
import { createHash } from 'crypto';
import { cache, cacheKey, TTL, KEY } from '../../../../lib/cache';

const trendToFrontend: Record<string, string> = { Rising: 'up', Falling: 'down', Stable: 'stable' };

const PREDICTION_API = process.env.PREDICTION_API ?? import.meta.env.PREDICTION_API;
const API_KEY = process.env.PREDICTION_API_KEY ?? import.meta.env.PREDICTION_API_KEY;
const API_KEY_SOURCE = process.env.PREDICTION_API_KEY
  ? 'process.env'
  : import.meta.env.PREDICTION_API_KEY
    ? 'import.meta.env'
    : 'missing';

const fingerprintKey = (key?: string) => {
  if (!key) return { keyLen: 0, keySuffix: null, keyHash: null };
  const keyHash = createHash('sha256').update(key).digest('hex').slice(0, 8);
  return {
    keyLen: key.length,
    keySuffix: key.slice(-4),
    keyHash,
  };
};

interface PriceHistoryCache {
  highs: number[];
  lows: number[];
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
    const redisCacheKey = cacheKey(KEY.ITEM_PRICE, 'history:v2', itemId.toString());

    // Try cache first
    let data: PriceHistoryCache | null = null;
    try {
      const cached = await cache.get<PriceHistoryCache>(redisCacheKey);
      if (
        cached &&
        Array.isArray(cached.highs) &&
        Array.isArray(cached.lows) &&
        cached.highs.length > 0 &&
        cached.lows.length > 0
      ) {
        data = cached;
      }
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
        if (engineRes.status === 401) {
          console.error(`[PriceHistory/${itemId}] Engine 401 fingerprint`, {
            apiKeySource: API_KEY_SOURCE,
            ...fingerprintKey(API_KEY)
          });
        }
        throw new Error(`Engine returned ${engineRes.status}`);
      }

      const raw = await engineRes.json();

      const history = raw.history ?? [];
      const highs: number[] = history
        .map((pt: { high: number }) => Math.round(pt.high))
        .filter((v: number) => !isNaN(v));
      const lows: number[] = history
        .map((pt: { low: number }) => Math.round(pt.low))
        .filter((v: number) => !isNaN(v));

      data = {
        highs,
        lows,
        trend: trendToFrontend[raw.trend] ?? 'stable',
      };

      // Cache for 5 minutes (prices update every ~5 min). Don't cache empty arrays:
      // empty history can be caused by transient engine/data issues.
      if (highs.length > 0 && lows.length > 0) {
        cache.set(redisCacheKey, data, TTL.PRICE_HISTORY).catch((err) => {
          console.warn('[PriceHistory] Cache write failed:', err?.message);
        });
      }
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
