import type { APIRoute } from 'astro';
import { activeTradesRepo, tradeHistoryRepo } from '../../../../../lib/repositories';
import { reportTradeOutcome, submitEngineFeedback } from '../../../../../lib/api';
import { dispatchWebhook } from '../../../../../lib/webhook';
import { advanceMockTrade, findMockTrade } from '../../../../../lib/mock-data';
import { calculateFlipProfit } from '../../../../../lib/ge-tax';

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

    const userId = locals.user.id;
    const isDevUser = import.meta.env.DEV && userId === 'dev-user';

    // Verify trade belongs to user
    const trade = isDevUser ? findMockTrade(id) : await activeTradesRepo.findById(id);
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
      const result = advanceMockTrade(id);
      return new Response(JSON.stringify({
        success: true,
        trade: result.trade,
        profit: result.profit,
        message: result.message
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    if (trade.phase === 'buying') {
      // Advance to selling phase
      await activeTradesRepo.updatePhase(id, 'selling');
      const updatedTrade = await activeTradesRepo.findById(id);

      // Dispatch webhook to ML engine (fire-and-forget) so the engine can learn from phase transitions.
      if (updatedTrade) {
        dispatchWebhook('TRADE_UPDATED', userId, id, {
          itemId: updatedTrade.item_id,
          itemName: updatedTrade.item_name,
          buyPrice: updatedTrade.buy_price,
          sellPrice: updatedTrade.sell_price,
          // `DECIMAL` from Postgres can come back as string; normalize for the webhook payload.
          offsetPct: updatedTrade.offset_pct == null ? null : Number(updatedTrade.offset_pct),
          quantity: updatedTrade.quantity,
          recId: updatedTrade.rec_id,
          modelId: updatedTrade.model_id,
          expectedHours: updatedTrade.expected_hours ?? null,
          createdAt: updatedTrade.created_at ? new Date(updatedTrade.created_at).toISOString() : null
        });
      }

      // Structured ML feedback: buy leg filled (non-blocking).
      // We use a simple heuristic: compare time-to-fill against ~50% of expectedHours.
      const createdAtMs = trade.created_at ? new Date(trade.created_at).getTime() : null;
      const expectedHoursRaw = trade.expected_hours;
      const expectedHours =
        typeof expectedHoursRaw === 'number' ? expectedHoursRaw : Number(expectedHoursRaw);
      if (createdAtMs && Number.isFinite(expectedHours) && expectedHours > 0) {
        const elapsedHours = (Date.now() - createdAtMs) / (1000 * 60 * 60);
        const expectedBuyHours = expectedHours * 0.5;
        const feedbackType = elapsedHours <= expectedBuyHours ? 'filled_quickly' : 'filled_slowly';
        submitEngineFeedback({
          userId,
          itemId: trade.item_id,
          itemName: trade.item_name,
          recId: trade.rec_id || undefined,
          offsetPct: trade.offset_pct === null ? undefined : Number(trade.offset_pct),
          feedbackType,
          side: 'buy',
          notes: `auto:phase_advance buy_filled elapsedHours=${elapsedHours.toFixed(2)} expectedBuyHours=${expectedBuyHours.toFixed(2)} expectedHours=${expectedHours}`,
          recommendedPrice: trade.buy_price,
          actualPrice: trade.actual_buy_price ?? undefined
        }).catch(() => {
          // Silent fail - ML feedback is optional
        });
      }

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
      const profit = calculateFlipProfit(actualBuyPrice, actualSellPrice, trade.quantity);

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
        offset_pct: trade.offset_pct ?? null,
        status: 'completed',
        notes: null,
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
        buyPrice: actualBuyPrice,
        sellPrice: actualSellPrice,
        quantity: trade.quantity,
        actualProfit: profit,
        recId: trade.rec_id || undefined,
        modelId: trade.model_id || undefined,
        offsetPct: trade.offset_pct === null ? undefined : Number(trade.offset_pct)
      }).catch(() => {
        // Silently fail - ML feedback is optional
      });

      // Delete active trade
      await activeTradesRepo.delete(id);

      // Dispatch webhook
      dispatchWebhook('TRADE_COMPLETED', trade.user_id, id, {
        itemId: trade.item_id,
        itemName: trade.item_name,
        buyPrice: actualBuyPrice,
        sellPrice: actualSellPrice,
        // `DECIMAL` from Postgres can come back as string; normalize for the webhook payload.
        offsetPct: trade.offset_pct == null ? null : Number(trade.offset_pct),
        quantity: trade.quantity,
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
          actualPrice: actualSellPrice
        }).catch(() => {
          // Silent fail - ML feedback is optional
        });
      }

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
