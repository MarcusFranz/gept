import type { SSEMessage, UpdateRecommendation } from './types';
import { pubsub } from './cache';

// Connection entry with controller for sending messages
interface SSEConnection {
  controller: ReadableStreamDefaultController<Uint8Array>;
  userId: string;
  connectedAt: number;
  lastHeartbeat: number;
}

// In-memory connection store for active connections
// Note: Active connections must be in-memory (they're live streams)
// But pending alerts are stored in Redis to persist across serverless invocations
const connections = new Map<string, Set<SSEConnection>>();

// Local pending alerts (fallback when Redis is unavailable)
const localPendingAlerts = new Map<string, UpdateRecommendation[]>();

const encoder = new TextEncoder();

/**
 * Format data as SSE event
 */
function formatSSEMessage(message: SSEMessage): string {
  return `data: ${JSON.stringify(message)}\n\n`;
}

/**
 * Send a message to a specific connection
 */
function sendToConnection(conn: SSEConnection, message: SSEMessage): boolean {
  try {
    const data = encoder.encode(formatSSEMessage(message));
    conn.controller.enqueue(data);
    return true;
  } catch {
    return false;
  }
}

/**
 * SSE Connection Manager
 *
 * Manages active SSE connections and routes messages to users.
 * Supports multiple tabs per user.
 */
export const sseManager = {
  /**
   * Register a new SSE connection
   */
  addConnection(userId: string, controller: ReadableStreamDefaultController<Uint8Array>): SSEConnection {
    const conn: SSEConnection = {
      controller,
      userId,
      connectedAt: Date.now(),
      lastHeartbeat: Date.now()
    };

    if (!connections.has(userId)) {
      connections.set(userId, new Set());
    }
    connections.get(userId)!.add(conn);

    // Send connected message
    sendToConnection(conn, {
      type: 'CONNECTED',
      timestamp: new Date().toISOString()
    });

    // Send any pending alerts from Redis (async, don't block connection)
    pubsub.getPendingAlerts(userId).then(alerts => {
      for (const alert of alerts) {
        sendToConnection(conn, {
          type: 'ALERT',
          data: alert as UpdateRecommendation,
          timestamp: new Date().toISOString()
        });
      }
    });

    // Also check local pending alerts (fallback)
    const localPending = localPendingAlerts.get(userId);
    if (localPending && localPending.length > 0) {
      for (const alert of localPending) {
        sendToConnection(conn, {
          type: 'ALERT',
          data: alert,
          timestamp: new Date().toISOString()
        });
      }
      localPendingAlerts.delete(userId);
    }

    return conn;
  },

  /**
   * Remove an SSE connection
   */
  removeConnection(userId: string, conn: SSEConnection): void {
    const userConns = connections.get(userId);
    if (userConns) {
      userConns.delete(conn);
      if (userConns.size === 0) {
        connections.delete(userId);
      }
    }
  },

  /**
   * Send an alert to all of a user's connections
   * If no connections exist, queue the alert in Redis for later delivery
   */
  sendToUser(userId: string, alert: UpdateRecommendation): number {
    const userConns = connections.get(userId);

    if (!userConns || userConns.size === 0) {
      // No active connections in this instance - queue alert in Redis
      // This allows the alert to be delivered when user reconnects (possibly to different instance)
      pubsub.queueAlert(userId, alert);

      // Also queue locally as fallback
      if (!localPendingAlerts.has(userId)) {
        localPendingAlerts.set(userId, []);
      }
      const queue = localPendingAlerts.get(userId)!;
      if (queue.length < 50) {
        queue.push(alert);
      }
      return 0;
    }

    const message: SSEMessage = {
      type: 'ALERT',
      data: alert,
      timestamp: new Date().toISOString()
    };

    let sent = 0;
    const deadConns: SSEConnection[] = [];

    for (const conn of userConns) {
      if (sendToConnection(conn, message)) {
        sent++;
      } else {
        deadConns.push(conn);
      }
    }

    // Clean up dead connections
    for (const conn of deadConns) {
      userConns.delete(conn);
    }
    if (userConns.size === 0) {
      connections.delete(userId);
    }

    return sent;
  },

  /**
   * Send heartbeat to a specific connection
   */
  sendHeartbeat(conn: SSEConnection): boolean {
    conn.lastHeartbeat = Date.now();
    return sendToConnection(conn, {
      type: 'HEARTBEAT',
      timestamp: new Date().toISOString()
    });
  },

  /**
   * Get connection count for a user
   */
  getConnectionCount(userId: string): number {
    return connections.get(userId)?.size || 0;
  },

  /**
   * Get total active connections (for monitoring)
   */
  getTotalConnections(): number {
    let total = 0;
    for (const userConns of connections.values()) {
      total += userConns.size;
    }
    return total;
  },

  /**
   * Get pending alerts for a user
   * Checks both Redis and local storage, clears after retrieval
   */
  async getPendingAlerts(userId: string): Promise<UpdateRecommendation[]> {
    // Get from Redis first
    const redisAlerts = await pubsub.getPendingAlerts(userId);

    // Also get local alerts
    const localAlerts = localPendingAlerts.get(userId) || [];
    localPendingAlerts.delete(userId);

    // Combine and return
    return [...(redisAlerts as UpdateRecommendation[]), ...localAlerts];
  },

  /**
   * Queue an alert for a user (for serverless webhook handler)
   * Stores in Redis for persistence across serverless invocations
   */
  queueAlert(userId: string, alert: UpdateRecommendation): void {
    // Try to send immediately if connections exist in this instance
    const sent = this.sendToUser(userId, alert);
    if (sent > 0) return;

    // Already queued in sendToUser, but if that failed, queue in Redis directly
    pubsub.queueAlert(userId, alert);
  }
};

export default sseManager;
