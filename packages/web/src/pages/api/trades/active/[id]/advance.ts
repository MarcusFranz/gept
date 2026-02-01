import type { APIRoute } from 'astro';
import { activeTradesRepo, tradeHistoryRepo } from '../../../../../lib/repositories';
import { dispatchWebhook } from '../../../../../lib/webhook';

export const POST: APIRoute = async ({ params, locals }) => {
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

    const { id } = params;
    if (!id) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Trade ID required'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Verify trade belongs to user
    const trade = await activeTradesRepo.findById(id);
    if (!trade || trade.user_id !== locals.user.id) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Trade not found'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (trade.phase === 'buying') {
      // Advance to selling phase
      await activeTradesRepo.updatePhase(id, 'selling');
      const updatedTrade = await activeTradesRepo.findById(id);

      return new Response(JSON.stringify({
        success: true,
        trade: updatedTrade,
        message: 'Advanced to selling phase'
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    } else {
      // Complete the trade
      const actualBuyPrice = trade.actual_buy_price || trade.buy_price;
      const actualSellPrice = trade.actual_sell_price || trade.sell_price;
      const profit = (actualSellPrice - actualBuyPrice) * trade.quantity;

      // Create history record (preserve prediction context from active trade)
      await tradeHistoryRepo.create({
        user_id: trade.user_id,
        item_id: trade.item_id,
        item_name: trade.item_name,
        buy_price: actualBuyPrice,
        sell_price: actualSellPrice,
        quantity: trade.quantity,
        profit,
        rec_id: trade.rec_id,
        model_id: trade.model_id,
        status: 'completed',
        notes: null,
        expected_profit: trade.expected_profit,
        confidence: trade.confidence,
        fill_probability: trade.fill_probability,
        expected_hours: trade.expected_hours
      });

      // Delete active trade
      await activeTradesRepo.delete(id);

      // Dispatch webhook
      dispatchWebhook('TRADE_COMPLETED', trade.user_id, id, {
        itemId: trade.item_id,
        itemName: trade.item_name,
        buyPrice: actualBuyPrice,
        sellPrice: actualSellPrice,
        quantity: trade.quantity,
        profit,
        recId: trade.rec_id,
        modelId: trade.model_id
      });

      return new Response(JSON.stringify({
        success: true,
        profit,
        message: 'Trade completed'
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }
  } catch (error) {
    console.error('Advance error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to advance trade'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
