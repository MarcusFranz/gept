import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';

// ---------------------------------------------------------------------------
// In-memory fallback rate limiter (activates when Redis is unavailable)
// Uses a simple sliding window approach with a Map. Limits are more aggressive
// (roughly half the Redis limits) since this is per-instance, not distributed.
// ---------------------------------------------------------------------------

interface MemoryBucket {
  timestamps: number[];
}

class InMemoryRateLimiter {
  private buckets: Map<string, MemoryBucket> = new Map();
  private readonly maxRequests: number;
  private readonly windowMs: number;
  private readonly prefix: string;
  private cleanupTimer: ReturnType<typeof setInterval> | null = null;

  constructor(maxRequests: number, windowMs: number, prefix: string) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.prefix = prefix;

    // Periodic cleanup of expired entries every 60 seconds
    this.cleanupTimer = setInterval(() => this.cleanup(), 60_000);
    // Allow the timer to not keep the process alive
    if (this.cleanupTimer && typeof this.cleanupTimer === 'object' && 'unref' in this.cleanupTimer) {
      this.cleanupTimer.unref();
    }
  }

  check(identifier: string): RateLimitResult {
    const key = `${this.prefix}${identifier}`;
    const now = Date.now();
    const windowStart = now - this.windowMs;

    let bucket = this.buckets.get(key);
    if (!bucket) {
      bucket = { timestamps: [] };
      this.buckets.set(key, bucket);
    }

    // Remove timestamps outside the window
    bucket.timestamps = bucket.timestamps.filter((t) => t > windowStart);

    const remaining = Math.max(0, this.maxRequests - bucket.timestamps.length);
    const success = bucket.timestamps.length < this.maxRequests;

    if (success) {
      bucket.timestamps.push(now);
    }

    return {
      success,
      limit: this.maxRequests,
      remaining: success ? Math.max(0, this.maxRequests - bucket.timestamps.length) : 0,
      reset: now + this.windowMs,
    };
  }

  private cleanup(): void {
    const now = Date.now();
    const windowStart = now - this.windowMs;
    this.buckets.forEach((bucket, key) => {
      bucket.timestamps = bucket.timestamps.filter((t) => t > windowStart);
      if (bucket.timestamps.length === 0) {
        this.buckets.delete(key);
      }
    });
  }
}

// In-memory fallback instances (created lazily, one per action type)
let memoryLoginLimiter: InMemoryRateLimiter | null = null;
let memorySignupLimiter: InMemoryRateLimiter | null = null;
let memoryPasswordResetLimiter: InMemoryRateLimiter | null = null;

function getMemoryLoginLimiter(): InMemoryRateLimiter {
  if (!memoryLoginLimiter) {
    // Half of Redis limit: 2 attempts per 15 minutes
    memoryLoginLimiter = new InMemoryRateLimiter(2, 15 * 60 * 1000, 'mem:login:');
  }
  return memoryLoginLimiter;
}

function getMemorySignupLimiter(): InMemoryRateLimiter {
  if (!memorySignupLimiter) {
    // Half of Redis limit: 1 signup per hour
    memorySignupLimiter = new InMemoryRateLimiter(1, 60 * 60 * 1000, 'mem:signup:');
  }
  return memorySignupLimiter;
}

function getMemoryPasswordResetLimiter(): InMemoryRateLimiter {
  if (!memoryPasswordResetLimiter) {
    // Half of Redis limit: 1 reset per hour
    memoryPasswordResetLimiter = new InMemoryRateLimiter(1, 60 * 60 * 1000, 'mem:reset:');
  }
  return memoryPasswordResetLimiter;
}

// ---------------------------------------------------------------------------
// Redis-based primary rate limiting
// ---------------------------------------------------------------------------

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

  // If Redis rate limiting is not available, fall back to in-memory limiter
  if (!limiter) {
    console.warn('[RateLimit] Redis unavailable for login check, using in-memory fallback');
    return getMemoryLoginLimiter().check(identifier);
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
    console.warn('[RateLimit] Falling back to in-memory limiter for login');
    return getMemoryLoginLimiter().check(identifier);
  }
}

/**
 * Check rate limit for signup attempts
 * @param ip - Client IP address
 */
export async function checkSignupRateLimit(ip: string): Promise<RateLimitResult> {
  const limiter = getSignupLimiter();

  // If Redis rate limiting is not available, fall back to in-memory limiter
  if (!limiter) {
    console.warn('[RateLimit] Redis unavailable for signup check, using in-memory fallback');
    return getMemorySignupLimiter().check(ip);
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
    console.warn('[RateLimit] Falling back to in-memory limiter for signup');
    return getMemorySignupLimiter().check(ip);
  }
}

/**
 * Check rate limit for password reset attempts
 * @param email - Email address requesting reset
 */
export async function checkPasswordResetRateLimit(email: string): Promise<RateLimitResult> {
  const limiter = getPasswordResetLimiter();

  // If Redis rate limiting is not available, fall back to in-memory limiter
  if (!limiter) {
    console.warn('[RateLimit] Redis unavailable for password reset check, using in-memory fallback');
    return getMemoryPasswordResetLimiter().check(email.toLowerCase());
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
    console.warn('[RateLimit] Falling back to in-memory limiter for password reset');
    return getMemoryPasswordResetLimiter().check(email.toLowerCase());
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
