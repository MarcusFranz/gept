import type { APIRoute } from 'astro';
import { sseManager } from '../../../lib/sse-manager';

const HEARTBEAT_INTERVAL = 30000; // 30 seconds

export const GET: APIRoute = async ({ locals }) => {
  // Require authentication
  if (!locals.user) {
    return new Response(JSON.stringify({
      success: false,
      error: 'Unauthorized'
    }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  const userId = locals.user.id;

  // Create a readable stream for SSE
  const stream = new ReadableStream({
    start(controller) {
      // Register this connection
      const conn = sseManager.addConnection(userId, controller);

      // Set up heartbeat interval
      const heartbeatTimer = setInterval(() => {
        const success = sseManager.sendHeartbeat(conn);
        if (!success) {
          clearInterval(heartbeatTimer);
          sseManager.removeConnection(userId, conn);
          try {
            controller.close();
          } catch {
            // Already closed
          }
        }
      }, HEARTBEAT_INTERVAL);

      // Store cleanup function on controller for cancel handling
      (controller as unknown as { cleanup: () => void }).cleanup = () => {
        clearInterval(heartbeatTimer);
        sseManager.removeConnection(userId, conn);
      };
    },
    cancel(controller) {
      // Clean up on client disconnect
      const cleanup = (controller as unknown as { cleanup?: () => void }).cleanup;
      if (cleanup) cleanup();
    }
  });

  return new Response(stream, {
    status: 200,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Connection': 'keep-alive',
      'X-Accel-Buffering': 'no' // Disable nginx buffering
    }
  });
};
