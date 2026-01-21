import type { APIRoute } from 'astro';
import { activeTradesRepo } from '../../../lib/repositories';
import { dispatchWebhook } from '../../../lib/webhook';

export const GET: APIRoute = async ({ locals }) => {
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

    const userId = locals.user.id;
    const trades = await activeTradesRepo.findByUserId(userId);

    return new Response(JSON.stringify({
      success: true,
      data: trades
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Get active trades error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to load active trades'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};

export const POST: APIRoute = async ({ request, locals }) => {
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

    const userId = locals.user.id;
    const body = await request.json();
    const { itemId, itemName, buyPrice, sellPrice, quantity, recId, modelId, expectedHours } = body;

    // Validate required fields
    if (!itemId || !itemName || !buyPrice || !sellPrice || !quantity) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Missing required fields'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Validate types and ranges
    if (typeof itemId !== 'number' || itemId <= 0 || itemId > 100000) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid item ID'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (typeof itemName !== 'string' || itemName.length < 1 || itemName.length > 100) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid item name'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (typeof buyPrice !== 'number' || buyPrice <= 0 || buyPrice > 2147483647) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid buy price'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (typeof sellPrice !== 'number' || sellPrice <= 0 || sellPrice > 2147483647) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid sell price'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (typeof quantity !== 'number' || quantity <= 0 || quantity > 2147483647) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid quantity'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Optional: expected hours from recommendation
    if (expectedHours !== undefined && (typeof expectedHours !== 'number' || expectedHours <= 0 || expectedHours > 168)) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid expected hours'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Check trade limit
    const currentCount = await activeTradesRepo.count(userId);
    if (currentCount >= 8) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Maximum 8 active trades allowed'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Create trade
    const trade = await activeTradesRepo.create({
      user_id: userId,
      item_id: itemId,
      item_name: itemName,
      buy_price: buyPrice,
      sell_price: sellPrice,
      quantity,
      rec_id: recId || null,
      model_id: modelId || null,
      expected_hours: expectedHours || undefined
    });

    // Dispatch webhook to ML engine (fire-and-forget)
    dispatchWebhook('TRADE_CREATED', userId, trade.id, {
      itemId,
      itemName,
      buyPrice,
      sellPrice,
      quantity,
      recId: recId || null,
      modelId: modelId || null
    });

    return new Response(JSON.stringify({
      success: true,
      data: trade
    }), {
      status: 201,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Create trade error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: error instanceof Error && error.message.includes('Maximum 8')
        ? 'Maximum 8 active trades allowed'
        : 'Failed to create trade'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
