import type { APIRoute } from 'astro';
import { searchItems } from '../../../lib/api';

export const GET: APIRoute = async ({ url }) => {
  const query = url.searchParams.get('q')?.toLowerCase() || '';
  const rawLimit = parseInt(url.searchParams.get('limit') || '15');
  const limit = Math.min(Math.max(1, rawLimit || 15), 50); // Cap between 1-50

  // Common cache headers for search results (5 min cache, 10 min stale-while-revalidate)
  const cacheHeaders = {
    'Content-Type': 'application/json',
    'Cache-Control': 'public, max-age=300, stale-while-revalidate=600'
  };

  if (query.length < 2) {
    return new Response(JSON.stringify({
      success: true,
      data: []
    }), {
      status: 200,
      headers: cacheHeaders
    });
  }

  try {
    const results = await searchItems(query, limit);
    return new Response(JSON.stringify({
      success: true,
      data: results
    }), {
      status: 200,
      headers: cacheHeaders
    });
  } catch (error) {
    console.error('[Search] API Error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Search unavailable'
    }), {
      status: 502,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
