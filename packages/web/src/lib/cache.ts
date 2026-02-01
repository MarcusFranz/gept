import { Redis } from '@upstash/redis';

// Initialize Redis client (lazy - only connects when used)
let redis: Redis | null = null;

function getRedis(): Redis | null {
  if (redis) return redis;

  // Server-side: use process.env directly (Vercel injects at runtime)
  const url = (process.env.UPSTASH_REDIS_REST_URL || '').trim();
  const token = (process.env.UPSTASH_REDIS_REST_TOKEN || '').trim();

  if (!url || !token) {
    console.warn('[Cache] Redis not configured - caching disabled');
    return null;
  }

  redis = new Redis({ url, token });
  return redis;
}

// Cache TTLs (in seconds)
export const TTL = {
  ITEM_PRICES: 120,        // 2 minutes - prices update every ~5 min
  ITEM_SEARCH: 300,        // 5 minutes - item names rarely change
  ITEM_METADATA: 3600,     // 1 hour - item IDs/names very stable
  RECOMMENDATIONS: 30,     // 30 seconds - personalized, short cache
  OPPORTUNITIES: 45,       // 45 seconds - shared data, changes every 5 min
  USER_SETTINGS: 300,      // 5 minutes - user settings
  PRICE_HISTORY: 300,      // 5 minutes - sparkline price charts
} as const;

// Key prefixes for organization
export const KEY = {
  ITEM_PRICE: 'price:',
  ITEM_SEARCH: 'search:',
  ITEM_META: 'meta:',
  RECS: 'recs:',
  OPPS: 'opps:',
  USER: 'user:',
  SSE: 'sse:',
} as const;

/**
 * Cache utility with graceful fallback when Redis is unavailable
 */
export const cache = {
  /**
   * Get a value from cache
   */
  async get<T>(key: string): Promise<T | null> {
    const client = getRedis();
    if (!client) return null;

    try {
      return await client.get<T>(key);
    } catch (err) {
      console.error('[Cache] Get error:', err);
      return null;
    }
  },

  /**
   * Set a value in cache with TTL
   */
  async set<T>(key: string, value: T, ttlSeconds: number): Promise<boolean> {
    const client = getRedis();
    if (!client) return false;

    try {
      await client.set(key, value, { ex: ttlSeconds });
      return true;
    } catch (err) {
      console.error('[Cache] Set error:', err);
      return false;
    }
  },

  /**
   * Delete a key from cache
   */
  async del(key: string): Promise<boolean> {
    const client = getRedis();
    if (!client) return false;

    try {
      await client.del(key);
      return true;
    } catch (err) {
      console.error('[Cache] Del error:', err);
      return false;
    }
  },

  /**
   * Delete multiple keys by pattern (use sparingly)
   */
  async delPattern(pattern: string): Promise<number> {
    const client = getRedis();
    if (!client) return 0;

    try {
      const keys = await client.keys(pattern);
      if (keys.length === 0) return 0;
      await client.del(...keys);
      return keys.length;
    } catch (err) {
      console.error('[Cache] DelPattern error:', err);
      return 0;
    }
  },

  /**
   * Get or fetch pattern - returns cached value or fetches fresh
   */
  async getOrFetch<T>(
    key: string,
    fetcher: () => Promise<T>,
    ttlSeconds: number
  ): Promise<T> {
    // Try cache first
    const cached = await this.get<T>(key);
    if (cached !== null) {
      return cached;
    }

    // Fetch fresh data
    const fresh = await fetcher();

    // Cache it (don't await - fire and forget)
    this.set(key, fresh, ttlSeconds);

    return fresh;
  },

  /**
   * Check if Redis is available
   */
  isAvailable(): boolean {
    return getRedis() !== null;
  }
};

/**
 * Pub/Sub for SSE alerts across serverless functions
 */
export const pubsub = {
  /**
   * Publish an alert to a user channel
   */
  async publish(userId: string, message: object): Promise<boolean> {
    const client = getRedis();
    if (!client) return false;

    try {
      await client.publish(`alerts:${userId}`, JSON.stringify(message));
      return true;
    } catch (err) {
      console.error('[PubSub] Publish error:', err);
      return false;
    }
  },

  /**
   * Store a pending alert for a user (fallback for when they're not connected)
   */
  async queueAlert(userId: string, alert: object): Promise<boolean> {
    const client = getRedis();
    if (!client) return false;

    try {
      const key = `${KEY.SSE}pending:${userId}`;
      // Add to list, keep max 50 alerts
      await client.lpush(key, JSON.stringify(alert));
      await client.ltrim(key, 0, 49);
      await client.expire(key, 86400); // 24 hour expiry
      return true;
    } catch (err) {
      console.error('[PubSub] Queue error:', err);
      return false;
    }
  },

  /**
   * Get and clear pending alerts for a user
   */
  async getPendingAlerts(userId: string): Promise<object[]> {
    const client = getRedis();
    if (!client) return [];

    try {
      const key = `${KEY.SSE}pending:${userId}`;
      const alerts = await client.lrange(key, 0, -1);
      if (alerts.length > 0) {
        await client.del(key);
      }
      return alerts.map(a => typeof a === 'string' ? JSON.parse(a) : a);
    } catch (err) {
      console.error('[PubSub] GetPending error:', err);
      return [];
    }
  }
};

/**
 * Helper to create cache keys
 */
export function cacheKey(...parts: (string | number)[]): string {
  return parts.join(':');
}

export default cache;
