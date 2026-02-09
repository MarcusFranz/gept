import type { APIRoute } from 'astro';
import { sql, type ActiveTrade } from '../../../lib/db';
import { dispatchWebhook, getWebhookSecret } from '../../../lib/webhook';

/**
 * POST /api/trades/resync
 * GET  /api/trades/resync
 *
 * Re-dispatches TRADE_CREATED webhooks for all active trades to the engine.
 * Use after engine restart or when ENGINE_WEBHOOK_URL is first configured.
 *
 * Auth: WEBHOOK_SECRET in Authorization header (internal use only).
 * Note: Some WAF/CDN setups strip Authorization headers. As a fallback, this
 * endpoint also accepts `X-Gept-Webhook-Secret: <WEBHOOK_SECRET>`.
 */
async function handleResync(request: Request): Promise<Response> {
  try {
    const secret = getWebhookSecret();
    const authHeader = request.headers.get('Authorization') || '';
    const altSecret = request.headers.get('X-Gept-Webhook-Secret') || '';

    const bearerOk = authHeader === `Bearer ${secret}`;
    const headerOk = altSecret === secret;

    if (!secret || (!bearerOk && !headerOk)) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Unauthorized',
        // Debug-only signals (no secret values) to help diagnose CDN/WAF header stripping
        // in production. If this endpoint is stable, we can remove later.
        debug: {
          secretConfigured: Boolean(secret),
          secretLen: secret.length,
          authorizationPresent: Boolean(authHeader),
          authorizationLen: authHeader.length,
          xGeptWebhookSecretPresent: Boolean(altSecret),
          xGeptWebhookSecretLen: altSecret.length,
          bearerOk,
          headerOk,
        }
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
        offsetPct: trade.offset_pct == null ? null : Number(trade.offset_pct),
        quantity: trade.quantity,
        recId: trade.rec_id || null,
        modelId: trade.model_id || null,
        expectedHours: trade.expected_hours ?? null,
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
}

export const POST: APIRoute = async ({ request }) => {
  return handleResync(request);
};

export const GET: APIRoute = async ({ request }) => {
  return handleResync(request);
};
