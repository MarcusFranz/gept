import type { APIRoute } from 'astro';
import { activeTradesRepo, tradeHistoryRepo } from '../../../../../lib/repositories';
import { submitEngineFeedback } from '../../../../../lib/api';
import { dispatchWebhook } from '../../../../../lib/webhook';
import { cache, cacheKey, KEY } from '../../../../../lib/cache';
import { deleteMockTrade, findMockTrade, updateMockTradeQuantity, updateMockTradeSellPrice } from '../../../../../lib/mock-data';

type CancelReason = 'changed_mind' | 'did_not_fill';

function parseCancelReason(request: Request): CancelReason {
  const reason = new URL(request.url).searchParams.get('reason');
  return reason === 'did_not_fill' ? 'did_not_fill' : 'changed_mind';
}

export const DELETE: APIRoute = async ({ params, locals, request }) => {
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

    const cancelReason = parseCancelReason(request);

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
    } else {
      // Create cancelled history entry
      await tradeHistoryRepo.create({
        user_id: userId,
        item_id: trade.item_id,
        item_name: trade.item_name,
        buy_price: trade.buy_price,
        sell_price: null,
        quantity: trade.quantity,
        profit: 0,
        notes: `Trade cancelled (${cancelReason})`,
        rec_id: trade.rec_id,
        model_id: trade.model_id,
        offset_pct: trade.offset_pct ?? null,
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
        offsetPct: trade.offset_pct == null ? null : Number(trade.offset_pct),
        quantity: trade.quantity,
        recId: trade.rec_id,
        modelId: trade.model_id
      });

      if (cancelReason === 'did_not_fill') {
        // Structured ML feedback (non-blocking)
        submitEngineFeedback({
          userId,
          itemId: trade.item_id,
          itemName: trade.item_name,
          recId: trade.rec_id || undefined,
          offsetPct: trade.offset_pct == null ? undefined : Number(trade.offset_pct),
          feedbackType: 'did_not_fill',
          side: trade.phase === 'selling' ? 'sell' : 'buy',
          notes: `auto:trade_cancelled reason=${cancelReason} phase=${trade.phase} progress=${trade.progress} expectedHours=${trade.expected_hours ?? 'null'}`,
          recommendedPrice: trade.phase === 'selling' ? trade.sell_price : trade.buy_price,
          actualPrice: trade.phase === 'selling'
            ? (trade.actual_sell_price ?? undefined)
            : (trade.actual_buy_price ?? undefined)
        }).catch(() => {
          // Silent fail - ML feedback is optional
        });
      }
    }

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

    const body = await request.json();
    const { quantity, sellPrice } = body;

    if (isDevUser) {
      if (quantity !== undefined) {
        updateMockTradeQuantity(tradeId, quantity);
      }
      if (sellPrice !== undefined) {
        updateMockTradeSellPrice(tradeId, sellPrice);
      }
    } else {
      if (quantity !== undefined) {
        await activeTradesRepo.updateQuantity(tradeId, quantity);
      }

      if (sellPrice !== undefined) {
        await activeTradesRepo.updateSellPrice(tradeId, sellPrice);
      }
    }

    const updated = isDevUser ? findMockTrade(tradeId) : await activeTradesRepo.findById(tradeId);

    // Invalidate short-lived cached update recommendations for this user.
    // Otherwise the UI can immediately re-show the same "Revise price" alert
    // for up to TTL.RECOMMENDATIONS even after the trade has been patched.
    cache.del(cacheKey(KEY.SSE, 'updates', userId)).catch(() => {});

    // Dispatch webhook to ML engine (fire-and-forget)
    if (updated && !isDevUser) {
      dispatchWebhook('TRADE_UPDATED', userId, tradeId, {
        itemId: updated.item_id,
        itemName: updated.item_name,
        buyPrice: updated.buy_price,
        sellPrice: updated.sell_price,
        offsetPct: updated.offset_pct == null ? null : Number(updated.offset_pct),
        quantity: updated.quantity,
        recId: updated.rec_id,
        modelId: updated.model_id
      });

      // If the user manually adjusted sell price, capture structured feedback.
      if (sellPrice !== undefined && typeof sellPrice === 'number' && sellPrice !== trade.sell_price) {
        const feedbackType = sellPrice < trade.sell_price ? 'price_too_high' : 'price_too_low';
        submitEngineFeedback({
          userId,
          itemId: updated.item_id,
          itemName: updated.item_name,
          recId: updated.rec_id || undefined,
          offsetPct: updated.offset_pct == null ? undefined : Number(updated.offset_pct),
          feedbackType,
          side: 'sell',
          notes: `auto:user_adjusted_sell_price from=${trade.sell_price} to=${sellPrice} phase=${updated.phase}`,
          recommendedPrice: trade.sell_price,
          actualPrice: sellPrice
        }).catch(() => {
          // Silent fail - ML feedback is optional
        });
      }
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
