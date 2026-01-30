import type { APIRoute } from 'astro';
import { activeTradesRepo, tradeHistoryRepo } from '../../../../../lib/repositories';
import { dispatchWebhook } from '../../../../../lib/webhook';

export const DELETE: APIRoute = async ({ params, locals }) => {
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
    const tradeId = params.id;
    if (!tradeId) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Trade ID required'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const trade = await activeTradesRepo.findById(tradeId);
    if (!trade || trade.user_id !== userId) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Trade not found'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Create cancelled history entry
    await tradeHistoryRepo.create({
      user_id: userId,
      item_id: trade.item_id,
      item_name: trade.item_name,
      buy_price: trade.buy_price,
      sell_price: null,
      quantity: trade.quantity,
      profit: 0,
      notes: 'Trade cancelled',
      rec_id: trade.rec_id,
      model_id: trade.model_id,
      status: 'cancelled'
    });

    // Delete the trade
    await activeTradesRepo.delete(tradeId);

    // Dispatch webhook to ML engine (fire-and-forget)
    dispatchWebhook('TRADE_CANCELLED', userId, tradeId, {
      itemId: trade.item_id,
      itemName: trade.item_name,
      buyPrice: trade.buy_price,
      sellPrice: trade.sell_price,
      quantity: trade.quantity,
      recId: trade.rec_id,
      modelId: trade.model_id
    });

    return new Response(JSON.stringify({
      success: true
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Delete trade error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to cancel trade'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};

export const PATCH: APIRoute = async ({ params, request, locals }) => {
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
    const tradeId = params.id;
    if (!tradeId) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Trade ID required'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const trade = await activeTradesRepo.findById(tradeId);
    if (!trade || trade.user_id !== userId) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Trade not found'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const body = await request.json();
    const { quantity, sellPrice } = body;

    if (quantity !== undefined) {
      await activeTradesRepo.updateQuantity(tradeId, quantity);
    }

    if (sellPrice !== undefined) {
      await activeTradesRepo.updateSellPrice(tradeId, sellPrice);
    }

    const updated = await activeTradesRepo.findById(tradeId);

    // Dispatch webhook to ML engine (fire-and-forget)
    if (updated) {
      dispatchWebhook('TRADE_UPDATED', userId, tradeId, {
        itemId: updated.item_id,
        itemName: updated.item_name,
        buyPrice: updated.buy_price,
        sellPrice: updated.sell_price,
        quantity: updated.quantity,
        recId: updated.rec_id,
        modelId: updated.model_id
      });
    }

    return new Response(JSON.stringify({
      success: true,
      data: updated
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Update trade error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to update trade'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
