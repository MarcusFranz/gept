import { createHmac, timingSafeEqual } from 'crypto';
import type { WebhookEvent, WebhookEventType, TradeEventPayload } from './types';

function getEnv(name: string): string {
  // Prefer runtime env for serverless (Vercel injects at runtime).
  // Fall back to Vite/Astro build-time env for local dev/build usage.
  const fromProcess =
    typeof process !== 'undefined' && process?.env ? process.env[name] : undefined;
  const fromVite = (import.meta as unknown as { env?: Record<string, string | undefined> })
    ?.env?.[name];
  return String(fromProcess ?? fromVite ?? '').trim();
}

export function getWebhookSecret(): string {
  return getEnv('WEBHOOK_SECRET');
}

export function getEngineWebhookUrl(): string {
  return getEnv('ENGINE_WEBHOOK_URL') || 'http://localhost:8000/webhooks/trades';
}

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
  const engineWebhookUrl = getEngineWebhookUrl();
  if (!engineWebhookUrl) {
    console.warn('ENGINE_WEBHOOK_URL not configured, skipping webhook dispatch');
    return;
  }

  const secret = getWebhookSecret();
  if (!secret) {
    console.error('WEBHOOK_SECRET not configured — cannot dispatch webhooks to engine');
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
  const signature = generateSignature(body, timestamp, secret);

  // Fire-and-forget: don't await, catch errors silently
  fetch(engineWebhookUrl, {
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
export function generateSignature(body: string, timestamp: string, secret?: string): string {
  const webhookSecret = (secret ?? getWebhookSecret()).trim();
  if (!webhookSecret) {
    console.error('WEBHOOK_SECRET not configured — cannot sign outbound webhooks');
    return '';
  }
  const payload = `${timestamp}.${body}`;
  return createHmac('sha256', webhookSecret).update(payload).digest('hex');
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
  const webhookSecret = getWebhookSecret();
  if (!webhookSecret) {
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
  const expectedSignature = generateSignature(body, timestamp, webhookSecret);
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
