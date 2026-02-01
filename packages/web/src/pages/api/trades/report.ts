import type { APIRoute } from 'astro';
import { tradeHistoryRepo } from '../../../lib/repositories';
import { reportTradeOutcome } from '../../../lib/api';

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
    const { itemId, itemName, buyPrice, sellPrice, quantity, profit, notes, recId, modelId, expectedProfit, confidence, fillProbability, expectedHours } = body;

    // Validate required fields
    if (!itemName || profit === undefined) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Item name and profit are required'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Validate types and ranges
    if (typeof itemName !== 'string' || itemName.length < 1 || itemName.length > 100) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid item name'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (typeof profit !== 'number' || profit < -2147483647 || profit > 2147483647) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid profit value'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (notes && (typeof notes !== 'string' || notes.length > 500)) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Notes too long (max 500 characters)'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Create history entry
    const entry = await tradeHistoryRepo.create({
      user_id: userId,
      item_id: itemId ?? null,
      item_name: itemName,
      buy_price: buyPrice ?? null,
      sell_price: sellPrice ?? null,
      quantity: quantity ?? null,
      profit,
      notes: notes || null,
      rec_id: recId || null,
      model_id: modelId || null,
      status: 'completed',
      expected_profit: expectedProfit ?? null,
      confidence: confidence ?? null,
      fill_probability: fillProbability ?? null,
      expected_hours: expectedHours ?? null
    });

    // Report to ML API (non-blocking) if we have all the data
    if (itemId && buyPrice && sellPrice && quantity) {
      reportTradeOutcome({
        userId,
        itemId,
        itemName,
        buyPrice,
        sellPrice,
        quantity,
        actualProfit: profit,
        recId,
        modelId
      }).catch(() => {
        // Silently fail - ML feedback is optional
      });
    }

    return new Response(JSON.stringify({
      success: true,
      data: entry
    }), {
      status: 201,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Report trade error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to report trade'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
