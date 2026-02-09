import type { APIRoute } from 'astro';
import { activeTradesRepo, tradeHistoryRepo } from '../../../../../lib/repositories';
import { reportTradeOutcome, submitEngineFeedback } from '../../../../../lib/api';
import { dispatchWebhook } from '../../../../../lib/webhook';
import { deleteMockTrade, findMockTrade } from '../../../../../lib/mock-data';
import { calculateFlipProfit } from '../../../../../lib/ge-tax';

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
    const { sellPrice, quantity, notes } = body;
    const resolvedSellPriceRaw = typeof sellPrice === 'number' ? sellPrice : Number(sellPrice);
    const resolvedQtyRaw = typeof quantity === 'number' ? quantity : Number(quantity);
    const resolvedSellPrice = Number.isFinite(resolvedSellPriceRaw)
      ? Math.max(0, Math.round(resolvedSellPriceRaw))
      : trade.sell_price;
    const resolvedQty = Number.isFinite(resolvedQtyRaw)
      ? Math.max(0, Math.round(resolvedQtyRaw))
      : trade.quantity;
    const profit = calculateFlipProfit(trade.buy_price, resolvedSellPrice, resolvedQty);

    // Create history entry (preserve prediction context from active trade)
    await tradeHistoryRepo.create({
      user_id: userId,
      item_id: trade.item_id,
      item_name: trade.item_name,
      buy_price: trade.buy_price,
      sell_price: resolvedSellPrice,
      quantity: resolvedQty,
      profit,
      notes: notes || null,
      rec_id: trade.rec_id,
      model_id: trade.model_id,
      offset_pct: trade.offset_pct ?? null,
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
      sellPrice: resolvedSellPrice,
      quantity: resolvedQty,
      actualProfit: profit,
      recId: trade.rec_id || undefined,
      modelId: trade.model_id || undefined,
      offsetPct: trade.offset_pct === null ? undefined : Number(trade.offset_pct)
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
      sellPrice: resolvedSellPrice,
      // `DECIMAL` from Postgres can come back as string; normalize for the webhook payload.
      offsetPct: trade.offset_pct == null ? null : Number(trade.offset_pct),
      quantity: resolvedQty,
      profit,
      recId: trade.rec_id,
      modelId: trade.model_id
    });

    // Structured ML feedback: sell leg filled (non-blocking).
    const createdAtMs = trade.created_at ? new Date(trade.created_at).getTime() : null;
    const expectedHoursRaw = trade.expected_hours;
    const expectedHours =
      typeof expectedHoursRaw === 'number' ? expectedHoursRaw : Number(expectedHoursRaw);
    if (createdAtMs && Number.isFinite(expectedHours) && expectedHours > 0) {
      const elapsedHours = (Date.now() - createdAtMs) / (1000 * 60 * 60);
      const feedbackType = elapsedHours <= expectedHours ? 'filled_quickly' : 'filled_slowly';
        submitEngineFeedback({
          userId,
          itemId: trade.item_id,
          itemName: trade.item_name,
          recId: trade.rec_id || undefined,
          offsetPct: trade.offset_pct === null ? undefined : Number(trade.offset_pct),
          feedbackType,
          side: 'sell',
          notes: `auto:trade_completed elapsedHours=${elapsedHours.toFixed(2)} expectedHours=${expectedHours}`,
          recommendedPrice: trade.sell_price,
          actualPrice: resolvedSellPrice
        }).catch(() => {
          // Silent fail - ML feedback is optional
        });
      }

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
