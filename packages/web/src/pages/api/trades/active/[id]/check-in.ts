// packages/web/src/pages/api/trades/active/[id]/check-in.ts
import type { APIRoute } from 'astro';
import { activeTradesRepo } from '../../../../../lib/repositories';
import { findMockTrade, updateMockTradeProgress } from '../../../../../lib/mock-data';

export const PUT: APIRoute = async ({ params, request, locals }) => {
  try {
    if (!locals.user) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Unauthorized'
      }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const { id } = params;
    if (!id) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Trade ID required'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const userId = locals.user.id;
    const isDevUser = import.meta.env.DEV && userId === 'dev-user';

    // Verify trade belongs to user
    const trade = isDevUser ? findMockTrade(id) : await activeTradesRepo.findById(id);
    if (!trade || trade.user_id !== userId) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Trade not found'
      }), {
        status: 404,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    const body = await request.json();
    const { progress } = body;

    // Validate progress
    if (typeof progress !== 'number' || progress < 0 || progress > 100) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Progress must be a number between 0 and 100'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Calculate next check-in based on progress and expected time
    const expectedHours = trade.expected_hours || 4;
    const elapsedMs = Date.now() - new Date(trade.created_at).getTime();
    const elapsedHours = elapsedMs / (1000 * 60 * 60);
    const expectedProgress = Math.min(100, (elapsedHours / expectedHours) * 100);

    // If behind schedule, check in more frequently
    const behindSchedule = progress < expectedProgress - 10;
    const checkInIntervalHours = behindSchedule
      ? Math.max(0.5, expectedHours * 0.1) // 10% of expected time, min 30 min
      : Math.max(1, expectedHours * 0.25);  // 25% of expected time, min 1 hour

    const nextCheckIn = new Date();
    nextCheckIn.setMinutes(nextCheckIn.getMinutes() + (checkInIntervalHours * 60));

    // Update trade progress
    const updatedTrade = isDevUser
      ? updateMockTradeProgress(id, progress, nextCheckIn)
      : await activeTradesRepo.updateProgress(id, progress, nextCheckIn);

    // TODO: Add guidance logic here based on progress vs expected
    // For now, return simple response
    let guidance = undefined;
    if (behindSchedule && progress < 30) {
      guidance = {
        action: 'relist' as const,
        reason: 'Filling slower than expected',
        params: {
          newPrice: Math.round(trade.buy_price * 1.005), // Suggest 0.5% higher
          priceDelta: Math.round(trade.buy_price * 0.005),
          expectedSpeedup: '2x faster'
        }
      };
    }

    return new Response(JSON.stringify({
      success: true,
      trade: updatedTrade,
      guidance,
      nextCheckIn: nextCheckIn.toISOString()
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Check-in error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to record check-in'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
