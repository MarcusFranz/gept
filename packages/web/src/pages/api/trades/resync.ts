import type { APIRoute } from 'astro';
import { sql, type ActiveTrade } from '../../../lib/db';
import { dispatchWebhook } from '../../../lib/webhook';

/**
 * POST /api/trades/resync
 *
 * Re-dispatches TRADE_CREATED webhooks for all active trades to the engine.
 * Use after engine restart or when ENGINE_WEBHOOK_URL is first configured.
 *
 * Auth: WEBHOOK_SECRET in Authorization header (internal use only).
 */
export const POST: APIRoute = async ({ request }) => {
  try {
    const secret = import.meta.env.WEBHOOK_SECRET || '';
    const authHeader = request.headers.get('Authorization') || '';

    if (!secret || authHeader !== `Bearer ${secret}`) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Unauthorized'
      }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Fetch all active trades across all users
    const trades = await sql<ActiveTrade>`
      SELECT * FROM active_trades ORDER BY created_at ASC
    `;

    let dispatched = 0;
    for (const trade of trades) {
      dispatchWebhook('TRADE_CREATED', trade.user_id, trade.id, {
        itemId: trade.item_id,
        itemName: trade.item_name,
        buyPrice: trade.buy_price,
        sellPrice: trade.sell_price,
        quantity: trade.quantity,
        recId: trade.rec_id || null,
        modelId: trade.model_id || null,
        expectedHours: trade.expected_hours || null,
        createdAt: trade.created_at ? new Date(trade.created_at).toISOString() : null
      });
      dispatched++;
    }

    return new Response(JSON.stringify({
      success: true,
      data: { dispatched }
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Trade resync error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to resync trades'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
