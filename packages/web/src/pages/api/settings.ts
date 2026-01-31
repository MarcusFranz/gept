import type { APIRoute } from 'astro';
import { userRepo } from '../../lib/repositories';
import { cache, KEY } from '../../lib/cache';

export const GET: APIRoute = async ({ locals }) => {
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
    const user = await userRepo.findOrCreate(userId, locals.user.email);
    const rateLimit = await userRepo.getRateLimitInfo(userId);

    return new Response(JSON.stringify({
      success: true,
      data: {
        user: {
          capital: user.capital,
          style: user.style,
          risk: user.risk,
          margin: user.margin,
          slots: user.slots,
          min_roi: user.min_roi,
          tier: user.tier,
          tutorialCompleted: user.tutorial_completed
        },
        rateLimit
      }
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch {
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to load settings'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};

export const PUT: APIRoute = async ({ request, locals }) => {
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
    const body = await request.json();
    const { capital, style, risk, margin, slots, min_roi, tutorialCompleted } = body;

    const updates: Record<string, unknown> = {};

    if (capital !== undefined && typeof capital === 'number' && capital > 0 && capital <= 2147483647) {
      updates.capital = capital;
    }

    if (style && ['passive', 'hybrid', 'active'].includes(style)) {
      updates.style = style;
    }

    if (risk && ['low', 'medium', 'high'].includes(risk)) {
      updates.risk = risk;
    }

    if (margin && ['conservative', 'moderate', 'aggressive'].includes(margin)) {
      updates.margin = margin;
    }

    if (slots !== undefined && typeof slots === 'number' && slots >= 1 && slots <= 8) {
      updates.slots = slots;
    }

    if (min_roi !== undefined && typeof min_roi === 'number') {
      updates.min_roi = min_roi;
    }

    if (tutorialCompleted !== undefined && typeof tutorialCompleted === 'boolean') {
      updates.tutorial_completed = tutorialCompleted;
    }

    if (Object.keys(updates).length > 0) {
      await userRepo.update(userId, updates);

      // Clear user's recommendation cache so they get fresh data with new settings
      // Key format: recs:userId:capital:style:risk:margin
      cache.delPattern(`${KEY.RECS}:${userId}:*`).catch(() => {});
    }

    const user = await userRepo.findById(userId);

    return new Response(JSON.stringify({
      success: true,
      data: {
        capital: user?.capital,
        style: user?.style,
        risk: user?.risk,
        margin: user?.margin,
        slots: user?.slots,
        min_roi: user?.min_roi,
        tier: user?.tier,
        tutorialCompleted: user?.tutorial_completed
      }
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch {
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to update settings'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
