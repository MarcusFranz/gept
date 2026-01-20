import type { APIRoute } from 'astro';
import { userRepo, activeTradesRepo } from '../../lib/repositories';
import { getRecommendations } from '../../lib/api';
import { cache, cacheKey, bucketCapital, TTL, KEY } from '../../lib/cache';

export const GET: APIRoute = async ({ locals, request }) => {
  // Use request.url to get params - Astro's url object can lose query params with ISR
  const url = new URL(request.url);

  try {
    // Get authenticated user from session
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

    // Get or create user record with trading settings
    const user = await userRepo.findOrCreate(userId, locals.user.email);

    // Parse query params from request.url (not Astro's url which can lose params on Vercel)
    const slotsParam = url.searchParams.get('slots');
    const excludeParam = url.searchParams.get('exclude');
    // Override settings from query params (for immediate refresh after settings change)
    const capitalOverride = url.searchParams.get('capital');
    const styleOverride = url.searchParams.get('style') as 'passive' | 'hybrid' | 'active' | null;
    const riskOverride = url.searchParams.get('risk') as 'low' | 'medium' | 'high' | null;
    const marginOverride = url.searchParams.get('margin') as 'conservative' | 'moderate' | 'aggressive' | null;
    // API limit is 8 slots max
    const requestedSlots = Math.min(slotsParam ? parseInt(slotsParam, 10) : 8, 8);

    // Rate limiting (skip for single-slot requests to allow "next" button)
    if (requestedSlots > 1) {
      const rateCheck = await userRepo.consumeQuery(userId);
      if (!rateCheck.allowed) {
        return new Response(JSON.stringify({
          success: false,
          error: 'Rate limit exceeded. Please try again tomorrow.',
          rateLimitExceeded: true
        }), {
          status: 429,
          headers: { 'Content-Type': 'application/json' }
        });
      }
    }

    // Get excluded items (active trades + any passed via query param)
    const activeTrades = await activeTradesRepo.findByUserId(userId);
    const excludedItems = activeTrades.map(t => t.item_id);

    // Add any additional excludes from query param
    if (excludeParam) {
      const additionalExcludes = excludeParam.split(',').map(id => parseInt(id, 10)).filter(id => !isNaN(id));
      excludedItems.push(...additionalExcludes);
    }

    // Get available capital (use override if provided)
    const tiedCapital = await activeTradesRepo.getTiedCapital(userId);
    const baseCapital = capitalOverride ? parseInt(capitalOverride, 10) : user.capital;
    const availableCapital = Math.max(0, baseCapital - tiedCapital);

    // Use override settings or fall back to user's saved settings
    const effectiveStyle = styleOverride || user.style;
    const effectiveRisk = riskOverride || user.risk;
    const effectiveMargin = marginOverride || user.margin;

    // Check if client requested fresh data (bypass cache)
    const skipCache = url.searchParams.get('fresh') === '1';

    // Build cache key based on effective settings
    const capitalBucket = bucketCapital(availableCapital);
    const redisCacheKey = cacheKey(KEY.RECS, userId, capitalBucket, effectiveStyle, effectiveRisk, effectiveMargin);

    // Try Redis cache first (unless fresh=1 requested)
    let recommendations: Awaited<ReturnType<typeof getRecommendations>> | null = null;

    if (!skipCache) {
      try {
        recommendations = await cache.get<Awaited<ReturnType<typeof getRecommendations>>>(redisCacheKey);
      } catch {
        // Continue without cache on Redis errors
      }
    }

    if (!recommendations) {
      // Fetch from prediction API
      recommendations = await getRecommendations(userId, {
        capital: availableCapital,
        style: effectiveStyle,
        risk: effectiveRisk,
        margin: effectiveMargin,
        excludedItems: [] // Don't exclude at API level - filter client-side for flexibility
      }, requestedSlots);

      // Cache for 30 seconds (fire and forget)
      cache.set(redisCacheKey, recommendations, TTL.RECOMMENDATIONS).catch(() => {});
    }

    // Filter out excluded items (active trades + skipped)
    if (excludedItems.length > 0) {
      const excludeSet = new Set(excludedItems);
      recommendations = recommendations.filter(r => !excludeSet.has(r.itemId));
    }

    return new Response(JSON.stringify({
      success: true,
      data: recommendations
    }), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0'
      }
    });
  } catch (error) {
    console.error('[Recommendations] API Error:', error);

    // Return mock data for development/demo
    return new Response(JSON.stringify({
      success: true,
      data: getMockRecommendations()
    }), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0'
      }
    });
  }
};

// Generate price history with realistic fluctuations
function generatePriceHistory(basePrice: number, trend: 'up' | 'down' | 'stable', points: number = 96): number[] {
  const history: number[] = [];
  let price = basePrice;
  const volatility = basePrice * 0.02; // 2% volatility

  // Work backwards from current price
  for (let i = 0; i < points; i++) {
    history.unshift(Math.round(price));
    const random = (Math.random() - 0.5) * volatility;
    const trendBias = trend === 'up' ? -volatility * 0.01 : trend === 'down' ? volatility * 0.01 : 0;
    price = price + random + trendBias;
  }
  return history;
}

function getMockRecommendations() {
  return [
    {
      id: 'rec-1',
      item: 'Armadyl Godsword',
      itemId: 11802,
      buyPrice: 10500000,
      sellPrice: 10850000,
      quantity: 1,
      capitalRequired: 10500000,
      expectedProfit: 350000,
      confidence: 0.85,
      fillProbability: 0.92,
      fillConfidence: 0.88,
      expectedHours: 2.5,
      trend: 'up' as const,
      reason: 'High demand due to recent PvM content update. Price trending upward with strong volume.',
      volume24h: 156,
      modelId: 'v2.1.0',
      isMultiLimitStrategy: false,
      priceHistory: generatePriceHistory(10500000, 'up')
    },
    {
      id: 'rec-2',
      item: 'Dragon Bones',
      itemId: 536,
      buyPrice: 2150,
      sellPrice: 2280,
      quantity: 5000,
      capitalRequired: 10750000,
      expectedProfit: 650000,
      confidence: 0.78,
      fillProbability: 0.95,
      fillConfidence: 0.91,
      expectedHours: 1.5,
      trend: 'stable' as const,
      reason: 'Consistent demand for prayer training. Good volume and reliable margins.',
      volume24h: 245000,
      modelId: 'v2.1.0',
      isMultiLimitStrategy: false,
      priceHistory: generatePriceHistory(2150, 'stable')
    },
    {
      id: 'rec-3',
      item: 'Twisted Bow',
      itemId: 20997,
      buyPrice: 1150000000,
      sellPrice: 1165000000,
      quantity: 1,
      capitalRequired: 1150000000,
      expectedProfit: 15000000,
      confidence: 0.72,
      fillProbability: 0.65,
      fillConfidence: 0.70,
      expectedHours: 8,
      trend: 'down' as const,
      reason: 'High-value item with decent margin. Lower fill probability but significant profit potential.',
      volume24h: 12,
      modelId: 'v2.1.0',
      isMultiLimitStrategy: false,
      priceHistory: generatePriceHistory(1150000000, 'down')
    },
    {
      id: 'rec-4',
      item: 'Bandos Chestplate',
      itemId: 11832,
      buyPrice: 18500000,
      sellPrice: 19100000,
      quantity: 1,
      capitalRequired: 18500000,
      expectedProfit: 600000,
      confidence: 0.81,
      fillProbability: 0.88,
      fillConfidence: 0.85,
      expectedHours: 3,
      trend: 'up' as const,
      reason: 'Popular melee armor. Steady demand from PvM community.',
      volume24h: 89,
      modelId: 'v2.1.0',
      isMultiLimitStrategy: false,
      priceHistory: generatePriceHistory(18500000, 'up')
    },
    {
      id: 'rec-5',
      item: 'Cannonballs',
      itemId: 2,
      buyPrice: 185,
      sellPrice: 198,
      quantity: 10000,
      capitalRequired: 1850000,
      expectedProfit: 130000,
      confidence: 0.92,
      fillProbability: 0.98,
      fillConfidence: 0.95,
      expectedHours: 0.5,
      trend: 'stable' as const,
      reason: 'High-volume consumable. Very reliable margins.',
      volume24h: 2500000,
      modelId: 'v2.1.0',
      isMultiLimitStrategy: false,
      priceHistory: generatePriceHistory(185, 'stable')
    },
    {
      id: 'rec-6',
      item: 'Abyssal Whip',
      itemId: 4151,
      buyPrice: 2850000,
      sellPrice: 2950000,
      quantity: 2,
      capitalRequired: 5700000,
      expectedProfit: 200000,
      confidence: 0.76,
      fillProbability: 0.91,
      fillConfidence: 0.84,
      expectedHours: 2,
      trend: 'down' as const,
      reason: 'Classic weapon with consistent demand.',
      volume24h: 420,
      modelId: 'v2.1.0',
      isMultiLimitStrategy: false,
      priceHistory: generatePriceHistory(2850000, 'down')
    },
    {
      id: 'rec-7',
      item: 'Saradomin Godsword',
      itemId: 11806,
      buyPrice: 28000000,
      sellPrice: 28900000,
      quantity: 1,
      capitalRequired: 28000000,
      expectedProfit: 900000,
      confidence: 0.79,
      fillProbability: 0.85,
      fillConfidence: 0.82,
      expectedHours: 4,
      trend: 'up' as const,
      reason: 'Strong spec weapon for slayer tasks.',
      volume24h: 67,
      modelId: 'v2.1.0',
      isMultiLimitStrategy: false,
      priceHistory: generatePriceHistory(28000000, 'up')
    },
    {
      id: 'rec-8',
      item: 'Black Chinchompa',
      itemId: 11959,
      buyPrice: 2450,
      sellPrice: 2580,
      quantity: 3000,
      capitalRequired: 7350000,
      expectedProfit: 390000,
      confidence: 0.83,
      fillProbability: 0.89,
      fillConfidence: 0.86,
      expectedHours: 1,
      trend: 'stable' as const,
      reason: 'Essential for efficient ranged training.',
      volume24h: 180000,
      modelId: 'v2.1.0',
      isMultiLimitStrategy: false,
      priceHistory: generatePriceHistory(2450, 'stable')
    }
  ];
}
