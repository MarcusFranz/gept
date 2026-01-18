import type { APIRoute } from 'astro';
import { tradeHistoryRepo } from '../../lib/repositories';

export const GET: APIRoute = async ({ url, locals }) => {
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

    const userId = locals.user.id;
    const timeframe = url.searchParams.get('timeframe') as 'today' | 'week' | 'month' | 'all' || 'all';

    const stats = await tradeHistoryRepo.getStats(userId, timeframe);

    return new Response(JSON.stringify({
      success: true,
      data: stats
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Get portfolio error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to load portfolio stats'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
