import { defineMiddleware } from 'astro:middleware';

// Security headers for production
const securityHeaders: Record<string, string> = {
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'X-XSS-Protection': '1; mode=block',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
};

const isDev = (import.meta as { env?: { DEV?: boolean } }).env?.DEV ?? process.env.NODE_ENV === 'development';

const createDevSession = () => ({
  user: {
    id: 'dev-user',
    email: 'dev@gept.local',
    name: 'Dev User',
    emailVerified: false,
    createdAt: new Date(),
    updatedAt: new Date(),
    image: null,
  },
  session: {
    id: 'dev-session',
    userId: 'dev-user',
    expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
    createdAt: new Date(),
    updatedAt: new Date(),
    token: 'dev-token',
    ipAddress: null,
    userAgent: null,
  },
});

const getAuth = async () => {
  try {
    const mod = await import('./lib/auth');
    return mod.auth;
  } catch {
    return null;
  }
};

export const onRequest = defineMiddleware(async (context, next) => {
  // Get session from Better Auth with error handling
  let session = null;
  try {
    const auth = await getAuth();
    if (auth) {
      session = await auth.api.getSession({
        headers: context.request.headers,
      });
    }
  } catch {
    session = null;
  }

  if (!session && isDev) {
    session = createDevSession();
  }

  if (session?.user) {
    context.locals.user = session.user;
    context.locals.session = session.session;
  } else {
    context.locals.user = null;
    context.locals.session = null;
  }

  // Protect routes that require authentication
  // Pages that require auth (redirect to /login)
  const protectedPages = ['/', '/account', '/settings', '/history', '/portfolio', '/stats', '/report'];
  // API routes that require auth (return 401)
  // Note: /api/items/search is public (not in this list)
  const protectedApiRoutes = ['/api/trades', '/api/user', '/api/settings', '/api/portfolio', '/api/recommendations', '/api/events', '/api/feedback'];
  const { pathname } = context.url;

  // Check if route requires authentication
  const isProtectedPage = protectedPages.includes(pathname);
  const isProtectedApi = protectedApiRoutes.some(route => pathname.startsWith(route));

  // Check if unauthenticated
  const isUnauthenticated = !session || !session.user;

  if (isUnauthenticated) {
    // For protected API routes, return 401
    if (isProtectedApi) {
      return new Response(
        JSON.stringify({ success: false, error: 'Unauthorized' }),
        {
          status: 401,
          headers: {
            'Content-Type': 'application/json',
            ...securityHeaders
          }
        }
      );
    }
    // For protected pages, redirect to welcome (landing page)
    if (isProtectedPage) {
      return context.redirect('/welcome');
    }
  }

  // Redirect authenticated users from welcome page to home
  if (!isUnauthenticated && pathname === '/welcome') {
    return context.redirect('/');
  }

  const response = await next();

  // Add security headers to all responses
  Object.entries(securityHeaders).forEach(([key, value]) => {
    response.headers.set(key, value);
  });

  return response;
});
