import { betterAuth } from 'better-auth';
import { Pool } from 'pg';
import { Resend } from 'resend';

// Validate required environment variables
if (!process.env.DATABASE_URL) {
  throw new Error('DATABASE_URL environment variable is required');
}

if (!process.env.BETTER_AUTH_SECRET) {
  throw new Error('BETTER_AUTH_SECRET environment variable is required');
}

const baseURL = process.env.BETTER_AUTH_URL || process.env.PUBLIC_APP_URL || 'http://localhost:3000';

// Build trusted origins list
const trustedOrigins = ['http://localhost:3000', 'http://localhost:4321'];
if (baseURL && !baseURL.includes('localhost')) {
  trustedOrigins.push(baseURL);
}

export const auth = betterAuth({
  baseURL,
  secret: process.env.BETTER_AUTH_SECRET,
  trustedOrigins,

  database: new Pool({
    connectionString: process.env.DATABASE_URL,
  }),

  // Email/password authentication
  emailAndPassword: {
    enabled: true,
    minPasswordLength: 8,
    sendResetPassword: async ({ user, url }) => {
      const isProd = process.env.NODE_ENV === 'production';

      if (isProd && url.includes('localhost')) {
        console.error('[Auth] Password reset URL points at localhost; check BETTER_AUTH_URL / PUBLIC_APP_URL configuration', {
          url,
        });
        throw new Error('Password reset is misconfigured');
      }

      // Dev fallback: log to console when no Resend API key is configured.
      // In prod, fail loudly so password reset isn't silently broken.
      if (!process.env.RESEND_API_KEY) {
        if (isProd) {
          console.error('[Auth] RESEND_API_KEY is not configured; cannot send password reset email');
          throw new Error('Password reset email is not configured');
        }
        console.log('\n=== PASSWORD RESET LINK ===');
        console.log(url);
        console.log('===========================\n');
        return;
      }

      if (isProd && !process.env.EMAIL_FROM) {
        console.error('[Auth] EMAIL_FROM is not configured; cannot send password reset email');
        throw new Error('Password reset email is not configured');
      }

      const resend = new Resend(process.env.RESEND_API_KEY);
      const fromAddress = process.env.EMAIL_FROM || 'GePT <noreply@gept.gg>';
      const subject = 'Reset your GePT password';
      const text = `Reset your GePT password:\n\n${url}\n\nIf you didn't request this, you can ignore this email.`;

      const { error } = await resend.emails.send({
        from: fromAddress,
        to: user.email,
        subject,
        text,
        html: `
          <div style="font-family: sans-serif; max-width: 480px; margin: 0 auto;">
            <h2>Password Reset</h2>
            <p>You requested a password reset for your GePT account.</p>
            <p>
              <a href="${url}" style="display: inline-block; padding: 12px 24px; background: #2563eb; color: #fff; text-decoration: none; border-radius: 6px;">
                Reset Password
              </a>
            </p>
            <p style="font-size: 14px; color: #666;">
              If the button doesn't work, copy and paste this URL into your browser:
              <br />
              <span style="word-break: break-all;">${url}</span>
            </p>
            <p style="color: #666; font-size: 14px;">
              If you didn't request this, you can safely ignore this email.
              This link will expire shortly.
            </p>
          </div>
        `,
      });

      if (error) {
        console.error(`[Auth] Failed to send password reset email to ${user.email}:`, error);
        throw new Error('Failed to send password reset email');
      }
    },
  },

  // Map to our table names and snake_case columns
  user: {
    modelName: 'users',
    fields: {
      emailVerified: 'email_verified',
      createdAt: 'created_at',
      updatedAt: 'updated_at',
    },
  },
  session: {
    modelName: 'sessions',
    fields: {
      userId: 'user_id',
      expiresAt: 'expires_at',
      ipAddress: 'ip_address',
      userAgent: 'user_agent',
      createdAt: 'created_at',
      updatedAt: 'updated_at',
    },
    expiresIn: 60 * 60 * 24 * 30, // 30 days
    updateAge: 60 * 60 * 24,
    cookieCache: {
      enabled: true,
      maxAge: 60 * 5,
    },
  },
  account: {
    modelName: 'accounts',
    fields: {
      userId: 'user_id',
      accountId: 'account_id',
      providerId: 'provider_id',
      accessToken: 'access_token',
      refreshToken: 'refresh_token',
      accessTokenExpiresAt: 'access_token_expires_at',
      refreshTokenExpiresAt: 'refresh_token_expires_at',
      idToken: 'id_token',
      createdAt: 'created_at',
      updatedAt: 'updated_at',
    },
  },
  verification: {
    modelName: 'verifications',
    fields: {
      expiresAt: 'expires_at',
      createdAt: 'created_at',
      updatedAt: 'updated_at',
    },
  },
});

export type Session = typeof auth.$Infer.Session;
export type User = typeof auth.$Infer.Session.user;
