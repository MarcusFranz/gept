import type { APIRoute } from 'astro';
import { activeTradesRepo } from '../../../lib/repositories';
import { dispatchWebhook } from '../../../lib/webhook';
import { createMockTrade, getMockTrades } from '../../../lib/mock-data';

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
    const isDevUser = import.meta.env.DEV && userId === 'dev-user';
    if (isDevUser) {
      return new Response(JSON.stringify({
        success: true,
        data: getMockTrades()
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }
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
    const isDevUser = import.meta.env.DEV && userId === 'dev-user';
    const body = await request.json();
    const { itemId, itemName, buyPrice, sellPrice, quantity, recId, modelId, expectedHours, confidence, fillProbability, expectedProfit } = body;

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

    // Optional: prediction context fields
    if (confidence !== undefined && (typeof confidence !== 'string' || !['low', 'medium', 'high'].includes(confidence))) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid confidence (must be low, medium, or high)'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (fillProbability !== undefined && (typeof fillProbability !== 'number' || !Number.isFinite(fillProbability) || fillProbability < 0 || fillProbability > 1)) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid fill probability (must be 0-1)'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (expectedProfit !== undefined && (typeof expectedProfit !== 'number' || !Number.isFinite(expectedProfit))) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid expected profit'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (isDevUser) {
      const trade = createMockTrade({
        user_id: userId,
        item_id: itemId,
        item_name: itemName,
        buy_price: buyPrice,
        sell_price: sellPrice,
        quantity,
        rec_id: recId || null,
        model_id: modelId || null,
        expected_hours: expectedHours ?? undefined,
        confidence: confidence ?? null,
        fill_probability: fillProbability ?? null,
        expected_profit: expectedProfit ?? null,
      });
      return new Response(JSON.stringify({
        success: true,
        data: trade
      }), {
        status: 201,
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
      expected_hours: expectedHours ?? undefined,
      suggested_sell_price: null,
      confidence: confidence ?? null,
      fill_probability: fillProbability ?? null,
      expected_profit: expectedProfit ?? null
    });

    // Dispatch webhook to ML engine (fire-and-forget)
    dispatchWebhook('TRADE_CREATED', userId, trade.id, {
      itemId,
      itemName,
      buyPrice,
      sellPrice,
      quantity,
      recId: recId || null,
      modelId: modelId || null,
      expectedHours: trade.expected_hours ?? null,
      createdAt: trade.created_at ? new Date(trade.created_at).toISOString() : null
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
