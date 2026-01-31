import type { APIRoute } from 'astro';
import { userRepo } from '../../../lib/repositories';
import { getRecommendationByItemId } from '../../../lib/api';

export const GET: APIRoute = async ({ params, locals }) => {
  const itemId = parseInt(params.id || '');

  if (!itemId || isNaN(itemId)) {
    return new Response(JSON.stringify({
      success: false,
      error: 'Invalid item ID'
    }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    });
  }

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

  try {
    const user = await userRepo.findOrCreate(userId, locals.user.email);

    const recommendation = await getRecommendationByItemId(itemId, userId, {
      capital: user.capital,
      style: user.style,
      risk: user.risk,
      margin: user.margin
    });

    if (recommendation) {
      return new Response(JSON.stringify({
        success: true,
        data: recommendation
      }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Item not recommended
    return new Response(JSON.stringify({
      success: false,
      notRecommended: true,
      error: 'This item is not currently recommended for flipping'
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    console.error(`[Items/${itemId}] API Error:`, error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to fetch item recommendation'
    }), {
      status: 502,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
