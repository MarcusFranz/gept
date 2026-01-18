import type { APIRoute } from 'astro';
import { tradeHistoryRepo } from '../../../lib/repositories';

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
    const filter = url.searchParams.get('filter') as 'all' | 'wins' | 'losses' | 'cancelled' || 'all';
    const rawPage = parseInt(url.searchParams.get('page') || '1');
    const rawLimit = parseInt(url.searchParams.get('limit') || '10');
    const itemId = url.searchParams.get('itemId');

    // Validate pagination params
    const page = Math.max(1, rawPage || 1);
    const limit = Math.min(Math.max(1, rawLimit || 10), 100); // Cap at 100
    const offset = (page - 1) * limit;

    const trades = await tradeHistoryRepo.findByUserId(userId, {
      filter,
      limit,
      offset,
      itemId: itemId ? parseInt(itemId) : undefined
    });

    const total = await tradeHistoryRepo.count(userId, filter);

    return new Response(JSON.stringify({
      success: true,
      data: trades,
      total,
      page,
      limit,
      totalPages: Math.ceil(total / limit)
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error('Get history error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to load trade history'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
