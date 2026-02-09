import type { APIRoute } from 'astro';
import { verifyWebhookSignature } from '../../../lib/webhook';
import { sseManager } from '../../../lib/sse-manager';
import { activeTradesRepo } from '../../../lib/repositories';
import type { UpdateRecommendation } from '../../../lib/types';

interface IncomingAlert {
  userId: string;
  alert: UpdateRecommendation;
}

export const POST: APIRoute = async ({ request }) => {
  try {
    // Get raw body for signature verification
    const body = await request.text();
    const timestamp = request.headers.get('X-Webhook-Timestamp') || '';
    const signature = request.headers.get('X-Webhook-Signature') || '';

    // Verify webhook signature
    if (!verifyWebhookSignature(body, timestamp, signature)) {
      console.warn('Webhook signature verification failed');
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid signature'
      }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Parse the alert payload
    let payload: IncomingAlert;
    try {
      payload = JSON.parse(body);
    } catch {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid JSON payload'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Validate required fields
    if (!payload.userId || !payload.alert) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Missing userId or alert'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Validate alert structure
    const alert = payload.alert;
    if (!alert.id || !alert.tradeId || !alert.type || !alert.reason) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid alert structure'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Persist suggested sell price to DB for durable display
    const suggestedPrice = alert.newSellPrice ?? alert.adjustedSellPrice;
    if (suggestedPrice && alert.tradeId) {
      // Suppress redundant alerts:
      // - If the user already revised their sell price to the suggested price.
      // - If we've already stored this exact suggestion (avoid re-pushing the same alert).
      //
      // This helps prevent the "accept -> immediate new alert" loop when the engine
      // hasn't yet received the TRADE_UPDATED webhook.
      try {
        const trade = await activeTradesRepo.findById(alert.tradeId);
        if (trade && trade.user_id === payload.userId) {
          if (Number(trade.sell_price) === Number(suggestedPrice)) {
            return new Response(JSON.stringify({
              success: true,
              data: { delivered: 0, queued: false, suppressed: true }
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json' }
            });
          }

          if (
            trade.suggested_sell_price != null
            && Number(trade.suggested_sell_price) === Number(suggestedPrice)
          ) {
            return new Response(JSON.stringify({
              success: true,
              data: { delivered: 0, queued: false, suppressed: true }
            }), {
              status: 200,
              headers: { 'Content-Type': 'application/json' }
            });
          }
        }
      } catch (err) {
        console.warn('Failed to load trade for alert suppression:', (err as Error)?.message);
      }

      await activeTradesRepo.setSuggestedSellPrice(alert.tradeId, suggestedPrice);
    }

    // Route to user's SSE connections (or queue for later)
    const sentCount = sseManager.sendToUser(payload.userId, alert);

    return new Response(JSON.stringify({
      success: true,
      data: {
        delivered: sentCount,
        queued: sentCount === 0
      }
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Webhook handler error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Internal server error'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
