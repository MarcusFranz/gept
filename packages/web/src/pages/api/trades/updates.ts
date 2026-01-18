import type { APIRoute } from 'astro';
import type { UpdateRecommendation, UpdateCheckResponse } from '../../../lib/types';

// Mock update recommendations for testing
// In production, this would call the ML backend
function getMockUpdates(tradeIds: string[]): UpdateRecommendation[] {
  const updates: UpdateRecommendation[] = [];

  // For demo: 30% chance per trade to get an update
  tradeIds.forEach((tradeId, index) => {
    if (Math.random() < 0.3) {
      const updateTypes: Array<'SWITCH_ITEM' | 'SELL_NOW' | 'ADJUST_PRICE'> = ['SWITCH_ITEM', 'SELL_NOW', 'ADJUST_PRICE'];
      const randomType = updateTypes[Math.floor(Math.random() * updateTypes.length)];

      const baseUpdate: UpdateRecommendation = {
        id: `update-${Date.now()}-${index}`,
        tradeId,
        type: randomType,
        reason: getReasonForType(randomType),
        confidence: 0.75 + Math.random() * 0.2,
        urgency: Math.random() > 0.7 ? 'high' : Math.random() > 0.4 ? 'medium' : 'low',
        timestamp: new Date().toISOString(),
        profitDelta: Math.round((Math.random() * 2 - 1) * 500000) // -500k to +500k
      };

      if (randomType === 'SWITCH_ITEM') {
        baseUpdate.newItem = {
          itemId: 11832,
          item: 'Bandos Chestplate',
          buyPrice: 18500000,
          sellPrice: 19100000,
          quantity: 1,
          expectedProfit: 600000,
          confidence: 0.81
        };
      } else if (randomType === 'SELL_NOW') {
        baseUpdate.adjustedSellPrice = 10750000;
      } else if (randomType === 'ADJUST_PRICE') {
        baseUpdate.originalSellPrice = 19100000;
        baseUpdate.newSellPrice = 18800000;
      }

      updates.push(baseUpdate);
    }
  });

  return updates;
}

function getReasonForType(type: string): string {
  switch (type) {
    case 'SWITCH_ITEM':
      return 'Market conditions have shifted. A better opportunity has been identified with higher expected returns.';
    case 'SELL_NOW':
      return 'Buy order partially filled. Recommend selling current inventory to secure profits before price drops.';
    case 'ADJUST_PRICE':
      return 'Price movement detected. Reducing sell price will improve fill probability with minimal profit impact.';
    default:
      return 'Market conditions have changed.';
  }
}

export const GET: APIRoute = async ({ request, locals }) => {
  try {
    if (!locals.user) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Unauthorized'
      }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const url = new URL(request.url);
    const tradeIdsParam = url.searchParams.get('tradeIds');
    const tradeIds = tradeIdsParam ? tradeIdsParam.split(',').filter(Boolean) : [];
    // In production, this would call the ML backend
    // For now, return mock data occasionally
    const updates = getMockUpdates(tradeIds);

    const response: UpdateCheckResponse = {
      updates,
      nextCheckIn: updates.length > 0 ? 15 : 30 // Check more frequently if updates exist
    };

    return new Response(JSON.stringify({
      success: true,
      data: response
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Updates check error:', error);
    return new Response(JSON.stringify({
      success: true,
      data: { updates: [], nextCheckIn: 60 }
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
