import type { APIRoute } from 'astro';
import type { UpdateCheckResponse } from '../../../lib/types';

// Trade updates are delivered via SSE (pushed from engine webhooks).
// This polling endpoint returns empty until a dedicated backend is implemented.

export const GET: APIRoute = async ({ locals }) => {
  if (!locals.user) {
    return new Response(JSON.stringify({
      success: false,
      error: 'Unauthorized'
    }), {
      status: 401,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  const response: UpdateCheckResponse = {
    updates: [],
    nextCheckIn: 60
  };

  return new Response(JSON.stringify({
    success: true,
    data: response
  }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' }
  });
};
