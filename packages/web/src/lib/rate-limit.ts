import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';

// Initialize Redis client for rate limiting
let redis: Redis | null = null;

function getRedis(): Redis | null {
  if (redis) return redis;

  const url = (process.env.UPSTASH_REDIS_REST_URL || '').trim();
  const token = (process.env.UPSTASH_REDIS_REST_TOKEN || '').trim();

  if (!url || !token) {
    return null;
  }

  redis = new Redis({ url, token });
  return redis;
}

// Rate limiters for different auth actions
let loginLimiter: Ratelimit | null = null;
let signupLimiter: Ratelimit | null = null;
let passwordResetLimiter: Ratelimit | null = null;

/**
 * Get or create the login rate limiter
 * 5 attempts per email per 15 minutes
 */
function getLoginLimiter(): Ratelimit | null {
  if (loginLimiter) return loginLimiter;

  const client = getRedis();
  if (!client) return null;

  loginLimiter = new Ratelimit({
    redis: client,
    limiter: Ratelimit.slidingWindow(5, '15 m'),
    prefix: 'ratelimit:login:',
    analytics: false,
  });

  return loginLimiter;
}

/**
 * Get or create the signup rate limiter
 * 3 signups per IP per hour
 */
function getSignupLimiter(): Ratelimit | null {
  if (signupLimiter) return signupLimiter;

  const client = getRedis();
  if (!client) return null;

  signupLimiter = new Ratelimit({
    redis: client,
    limiter: Ratelimit.slidingWindow(3, '1 h'),
    prefix: 'ratelimit:signup:',
    analytics: false,
  });

  return signupLimiter;
}

/**
 * Get or create the password reset rate limiter
 * 3 requests per email per hour
 */
function getPasswordResetLimiter(): Ratelimit | null {
  if (passwordResetLimiter) return passwordResetLimiter;

  const client = getRedis();
  if (!client) return null;

  passwordResetLimiter = new Ratelimit({
    redis: client,
    limiter: Ratelimit.slidingWindow(3, '1 h'),
    prefix: 'ratelimit:reset:',
    analytics: false,
  });

  return passwordResetLimiter;
}

export interface RateLimitResult {
  success: boolean;
  limit: number;
  remaining: number;
  reset: number;
}

/**
 * Check rate limit for login attempts
 * @param identifier - Email address or IP
 */
export async function checkLoginRateLimit(identifier: string): Promise<RateLimitResult> {
  const limiter = getLoginLimiter();

  // If rate limiting is not available, allow the request
  if (!limiter) {
    return { success: true, limit: 5, remaining: 5, reset: 0 };
  }

  try {
    const result = await limiter.limit(identifier);
    return {
      success: result.success,
      limit: result.limit,
      remaining: result.remaining,
      reset: result.reset,
    };
  } catch (err) {
    console.error('[RateLimit] Login check error:', err);
    // On error, allow the request
    return { success: true, limit: 5, remaining: 5, reset: 0 };
  }
}

/**
 * Check rate limit for signup attempts
 * @param ip - Client IP address
 */
export async function checkSignupRateLimit(ip: string): Promise<RateLimitResult> {
  const limiter = getSignupLimiter();

  if (!limiter) {
    return { success: true, limit: 3, remaining: 3, reset: 0 };
  }

  try {
    const result = await limiter.limit(ip);
    return {
      success: result.success,
      limit: result.limit,
      remaining: result.remaining,
      reset: result.reset,
    };
  } catch (err) {
    console.error('[RateLimit] Signup check error:', err);
    return { success: true, limit: 3, remaining: 3, reset: 0 };
  }
}

/**
 * Check rate limit for password reset attempts
 * @param email - Email address requesting reset
 */
export async function checkPasswordResetRateLimit(email: string): Promise<RateLimitResult> {
  const limiter = getPasswordResetLimiter();

  if (!limiter) {
    return { success: true, limit: 3, remaining: 3, reset: 0 };
  }

  try {
    const result = await limiter.limit(email.toLowerCase());
    return {
      success: result.success,
      limit: result.limit,
      remaining: result.remaining,
      reset: result.reset,
    };
  } catch (err) {
    console.error('[RateLimit] Password reset check error:', err);
    return { success: true, limit: 3, remaining: 3, reset: 0 };
  }
}

/**
 * Create rate limit exceeded response
 */
export function rateLimitResponse(result: RateLimitResult): Response {
  const retryAfter = Math.ceil((result.reset - Date.now()) / 1000);

  return new Response(
    JSON.stringify({
      success: false,
      error: 'Too many requests. Please try again later.',
      retryAfter: retryAfter > 0 ? retryAfter : 60,
    }),
    {
      status: 429,
      headers: {
        'Content-Type': 'application/json',
        'X-RateLimit-Limit': result.limit.toString(),
        'X-RateLimit-Remaining': result.remaining.toString(),
        'X-RateLimit-Reset': result.reset.toString(),
        'Retry-After': (retryAfter > 0 ? retryAfter : 60).toString(),
      },
    }
  );
}

/**
 * Extract client IP from request
 */
export function getClientIp(request: Request): string {
  // Check various headers for the real IP
  const forwardedFor = request.headers.get('x-forwarded-for');
  if (forwardedFor) {
    // x-forwarded-for can be a comma-separated list, take the first one
    return forwardedFor.split(',')[0].trim();
  }

  const realIp = request.headers.get('x-real-ip');
  if (realIp) {
    return realIp;
  }

  // Fallback to a default (this shouldn't happen in production with Vercel)
  return '127.0.0.1';
}
