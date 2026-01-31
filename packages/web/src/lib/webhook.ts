import { createHmac, timingSafeEqual } from 'crypto';
import type { WebhookEvent, WebhookEventType, TradeEventPayload } from './types';

const ENGINE_WEBHOOK_URL = import.meta.env.ENGINE_WEBHOOK_URL || 'http://localhost:8000/webhooks/trades';
const WEBHOOK_SECRET = import.meta.env.WEBHOOK_SECRET || '';
const TIMESTAMP_TOLERANCE_MS = 5 * 60 * 1000; // 5 minutes

/**
 * Dispatch a webhook event to the ML engine (fire-and-forget)
 */
export async function dispatchWebhook(
  eventType: WebhookEventType,
  userId: string,
  tradeId: string,
  payload: TradeEventPayload
): Promise<void> {
  if (!ENGINE_WEBHOOK_URL) {
    console.warn('ENGINE_WEBHOOK_URL not configured, skipping webhook dispatch');
    return;
  }

  const event: WebhookEvent = {
    eventType,
    timestamp: new Date().toISOString(),
    userId,
    tradeId,
    payload
  };

  const body = JSON.stringify(event);
  const timestamp = Date.now().toString();
  const signature = generateSignature(body, timestamp);

  // Fire-and-forget: don't await, catch errors silently
  fetch(ENGINE_WEBHOOK_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Webhook-Timestamp': timestamp,
      'X-Webhook-Signature': signature
    },
    body,
    signal: AbortSignal.timeout(5000) // 5s timeout
  }).catch(error => {
    console.error('Webhook dispatch failed:', error.message);
  });
}

/**
 * Generate HMAC-SHA256 signature for webhook payload
 */
export function generateSignature(body: string, timestamp: string): string {
  if (!WEBHOOK_SECRET) {
    console.error('WEBHOOK_SECRET not configured — cannot sign outbound webhooks');
    return '';
  }
  const payload = `${timestamp}.${body}`;
  return createHmac('sha256', WEBHOOK_SECRET).update(payload).digest('hex');
}

/**
 * Verify incoming webhook signature from ML engine
 * Returns true if signature is valid, false otherwise
 */
export function verifyWebhookSignature(
  body: string,
  timestamp: string,
  signature: string
): boolean {
  if (!WEBHOOK_SECRET) {
    console.error('WEBHOOK_SECRET not configured — rejecting webhook (fail closed)');
    return false;
  }

  // Check timestamp to prevent replay attacks
  const timestampMs = parseInt(timestamp, 10);
  if (isNaN(timestampMs)) {
    return false;
  }

  const now = Date.now();
  if (Math.abs(now - timestampMs) > TIMESTAMP_TOLERANCE_MS) {
    console.warn('Webhook timestamp outside tolerance window');
    return false;
  }

  // Verify signature
  const expectedSignature = generateSignature(body, timestamp);
  if (!expectedSignature) {
    return false;
  }

  try {
    const sigBuffer = Buffer.from(signature, 'hex');
    const expectedBuffer = Buffer.from(expectedSignature, 'hex');
    return sigBuffer.length === expectedBuffer.length &&
      timingSafeEqual(sigBuffer, expectedBuffer);
  } catch {
    return false;
  }
}
