import type { APIRoute } from 'astro';
import { feedbackRepo } from '../../../lib/repositories';

export const POST: APIRoute = async ({ request, locals }) => {
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
    const { type, rating, message, email, recommendation } = body;

    // Validate type
    if (!type || !['bug', 'feature', 'general', 'recommendation'].includes(type)) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Invalid feedback type'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // For recommendation feedback, rating is required
    if (type === 'recommendation' && !rating) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Rating is required for recommendation feedback'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // For general feedback, message is required
    if (type !== 'recommendation' && !message?.trim()) {
      return new Response(JSON.stringify({
        success: false,
        error: 'Message is required for this feedback type'
      }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Validate recommendation data if provided
    if (type === 'recommendation' && recommendation) {
      const required = ['id', 'itemId', 'item', 'buyPrice', 'sellPrice', 'quantity', 'expectedProfit', 'confidence', 'modelId'];
      const missing = required.filter(key => recommendation[key] === undefined);
      if (missing.length > 0) {
        return new Response(JSON.stringify({
          success: false,
          error: `Missing recommendation fields: ${missing.join(', ')}`
        }), {
          status: 400,
          headers: { 'Content-Type': 'application/json' }
        });
      }
    }

    const feedback = await feedbackRepo.create({
      userId,
      type,
      rating,
      message: message?.trim(),
      email: email?.trim(),
      recommendation
    });

    return new Response(JSON.stringify({
      success: true,
      data: { id: feedback.id }
    }), {
      status: 201,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error) {
    return new Response(JSON.stringify({
      success: false,
      error: 'Failed to submit feedback'
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
