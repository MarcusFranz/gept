// packages/web/src/pages/api/opportunities.ts
import type { APIRoute } from 'astro';

const PREDICTION_API = import.meta.env.PREDICTION_API || 'http://150.136.170.128:8000';
const API_KEY = import.meta.env.PREDICTION_API_KEY;

export const POST: APIRoute = async ({ request, locals }) => {
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

  try {
    // Parse filter parameters from body
    const filters = await request.json();

    // Call engine opportunities endpoint
    const engineRes = await fetch(`${PREDICTION_API}/api/v1/opportunities`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(API_KEY ? { 'X-API-Key': API_KEY } : {})
      },
      body: JSON.stringify({
        min_profit: filters.profitMin,
        max_profit: filters.profitMax,
        min_hours: filters.timeMin,
        max_hours: filters.timeMax,
        confidence: filters.confidence, // array of levels
        max_capital: filters.capitalMax,
        categories: filters.categories,
        limit: filters.limit || 50,
        offset: filters.offset || 0
      })
    });

    if (!engineRes.ok) {
      throw new Error(`Engine returned ${engineRes.status}`);
    }

    const data = await engineRes.json();

    // Transform to frontend format
    return new Response(JSON.stringify({
      success: true,
      data: {
        items: data.items.map((item: any) => ({
          id: item.id,
          itemId: item.item_id,
          item: item.item_name,
          iconUrl: item.icon_url,
          buyPrice: item.buy_price,
          sellPrice: item.sell_price,
          quantity: item.quantity,
          capitalRequired: item.capital_required,
          expectedProfit: item.expected_profit,
          expectedHours: item.expected_hours,
          confidence: item.confidence,
          fillProbability: item.fill_probability,
          volume24h: item.volume_24h,
          trend: item.trend,
          whyChips: item.why_chips,
          category: item.category
        })),
        total: data.total,
        hasMore: data.has_more
      }
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });

  } catch (error) {
    console.error('[Opportunities] API Error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to fetch opportunities'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
