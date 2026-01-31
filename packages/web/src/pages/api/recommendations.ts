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

    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to fetch recommendations',
      recommendations: []
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
