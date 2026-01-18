import type { APIRoute } from 'astro';
import { userRepo } from '../../../lib/repositories';
import { getRecommendationByItemId } from '../../../lib/api';

// Mock item recommendations
const mockItemRecommendations: Record<number, object> = {
  11802: {
    id: 'rec-ags',
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
    trend: 'up',
    reason: 'High demand due to recent PvM content update. Price trending upward with strong volume.',
    volume24h: 156,
    modelId: 'v2.1.0'
  },
  536: {
    id: 'rec-dbones',
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
    trend: 'stable',
    reason: 'Consistent demand for prayer training. Good volume and reliable margins.',
    volume24h: 245000,
    modelId: 'v2.1.0'
  }
};

export const GET: APIRoute = async ({ params, locals }) => {
  const itemId = parseInt(params.id || '');

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

  const userId = locals.user.id;

  try {
    const user = await userRepo.findOrCreate(userId, locals.user.email);

    // Try API first
    const recommendation = await getRecommendationByItemId(itemId, userId, {
      capital: user.capital,
      style: user.style,
      risk: user.risk,
      margin: user.margin
    });

    if (recommendation) {
      return new Response(JSON.stringify({
        success: true,
        data: recommendation
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  } catch {
    // Fall through to mock data
  }

  // Check mock data
  const mockRec = mockItemRecommendations[itemId];
  if (mockRec) {
    return new Response(JSON.stringify({
      success: true,
      data: mockRec
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  // Item not recommended
  return new Response(JSON.stringify({
    success: false,
    notRecommended: true,
    error: 'This item is not currently recommended for flipping'
  }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' }
  });
};
