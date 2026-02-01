import type { APIRoute } from 'astro';
import { auth } from '../../../lib/auth';
import {
  checkLoginRateLimit,
  checkSignupRateLimit,
  checkPasswordResetRateLimit,
  rateLimitResponse,
  getClientIp,
} from '../../../lib/rate-limit';

export const ALL: APIRoute = async (ctx) => {
  const { request } = ctx;
  const url = new URL(request.url);
  const pathname = url.pathname;

  // Only apply rate limiting to POST requests (login, signup, reset)
  if (request.method === 'POST') {
    try {
      // Clone request to read body (request body can only be read once)
      const clonedRequest = request.clone();

      // Check if this is a login attempt
      if (pathname.endsWith('/sign-in/email')) {
        const body = await clonedRequest.json().catch(() => ({}));
        const email = body.email?.toLowerCase() || '';

        if (email) {
          const result = await checkLoginRateLimit(email);
          if (!result.success) {
            return rateLimitResponse(result);
          }
        }
      }

      // Check if this is a signup attempt
      if (pathname.endsWith('/sign-up/email')) {
        const ip = getClientIp(request);
        const result = await checkSignupRateLimit(ip);
        if (!result.success) {
          return rateLimitResponse(result);
        }
      }

      // Check if this is a password reset request
      if (pathname.endsWith('/request-password-reset')) {
        const body = await clonedRequest.json().catch(() => ({}));
        const email = body.email?.toLowerCase() || '';

        if (email) {
          const result = await checkPasswordResetRateLimit(email);
          if (!result.success) {
            return rateLimitResponse(result);
          }
        }
      }
    } catch (err) {
      // If rate limiting fails, log and continue (fail open for availability)
      console.error('[Auth] Rate limit check failed:', err);
    }
  }

  return auth.handler(request);
};
