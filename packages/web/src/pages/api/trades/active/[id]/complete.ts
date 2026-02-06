import type { APIRoute } from 'astro';
import { activeTradesRepo, tradeHistoryRepo } from '../../../../../lib/repositories';
import { reportTradeOutcome } from '../../../../../lib/api';
import { dispatchWebhook } from '../../../../../lib/webhook';
import { deleteMockTrade, findMockTrade } from '../../../../../lib/mock-data';

export const POST: APIRoute = async ({ params, request, locals }) => {
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

    const isDevUser = import.meta.env.DEV && userId === 'dev-user';
    const trade = isDevUser ? findMockTrade(tradeId) : await activeTradesRepo.findById(tradeId);
    if (!trade || trade.user_id !== userId) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Trade not found'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (isDevUser) {
      deleteMockTrade(tradeId);
      return new Response(JSON.stringify({
        success: true
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const body = await request.json();
    const { sellPrice, quantity, profit, notes } = body;

    // Create history entry (preserve prediction context from active trade)
    await tradeHistoryRepo.create({
      user_id: userId,
      item_id: trade.item_id,
      item_name: trade.item_name,
      buy_price: trade.buy_price,
      sell_price: sellPrice || trade.sell_price,
      quantity: quantity || trade.quantity,
      profit,
      notes: notes || null,
      rec_id: trade.rec_id,
      model_id: trade.model_id,
      status: 'completed',
      expected_profit: trade.expected_profit,
      confidence: trade.confidence,
      fill_probability: trade.fill_probability,
      expected_hours: trade.expected_hours
    });

    // Report to ML API (non-blocking)
    reportTradeOutcome({
      userId,
      itemId: trade.item_id,
      itemName: trade.item_name,
      buyPrice: trade.buy_price,
      sellPrice: sellPrice || trade.sell_price,
      quantity: quantity || trade.quantity,
      actualProfit: profit,
      recId: trade.rec_id || undefined,
      modelId: trade.model_id || undefined
    }).catch(() => {
      // Silently fail - ML feedback is optional
    });

    // Delete active trade
    await activeTradesRepo.delete(tradeId);

    // Dispatch webhook to ML engine (fire-and-forget)
    dispatchWebhook('TRADE_COMPLETED', userId, tradeId, {
      itemId: trade.item_id,
      itemName: trade.item_name,
      buyPrice: trade.buy_price,
      sellPrice: sellPrice || trade.sell_price,
      quantity: quantity || trade.quantity,
      profit,
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
    console.error('Complete trade error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to complete trade'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
