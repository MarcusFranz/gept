import { betterAuth } from 'better-auth';
import { Pool } from 'pg';
import { Resend } from 'resend';

const env = import.meta.env as Record<string, string | undefined>;

const DATABASE_URL = process.env.DATABASE_URL ?? env.DATABASE_URL;
const BETTER_AUTH_SECRET = process.env.BETTER_AUTH_SECRET ?? env.BETTER_AUTH_SECRET;
const BETTER_AUTH_URL = process.env.BETTER_AUTH_URL ?? env.BETTER_AUTH_URL;
const RESEND_API_KEY = process.env.RESEND_API_KEY ?? env.RESEND_API_KEY;
const EMAIL_FROM = process.env.EMAIL_FROM ?? env.EMAIL_FROM;

// Validate required environment variables
if (!DATABASE_URL) {
  throw new Error('DATABASE_URL environment variable is required');
}

if (!BETTER_AUTH_SECRET) {
  throw new Error('BETTER_AUTH_SECRET environment variable is required');
}

// Build trusted origins list
const trustedOrigins = ['http://localhost:3000', 'http://localhost:4321'];
if (BETTER_AUTH_URL && !BETTER_AUTH_URL.includes('localhost')) {
  trustedOrigins.push(BETTER_AUTH_URL);
}

export const auth = betterAuth({
  baseURL: BETTER_AUTH_URL || 'http://localhost:3000',
  secret: BETTER_AUTH_SECRET,
  trustedOrigins,

  database: new Pool({
    connectionString: DATABASE_URL,
  }),

  // Email/password authentication
  emailAndPassword: {
    enabled: true,
    minPasswordLength: 8,
    sendResetPassword: async ({ user, url }) => {
      // Dev fallback: log to console when no Resend API key is configured
      if (!RESEND_API_KEY) {
        if (process.env.NODE_ENV === 'development') {
          console.log('\n=== PASSWORD RESET LINK ===');
          console.log(url);
          console.log('===========================\n');
        } else {
          console.warn(`[Auth] Password reset requested for ${user.email} but RESEND_API_KEY is not configured`);
        }
        return;
      }

      const resend = new Resend(RESEND_API_KEY);
      const fromAddress = EMAIL_FROM || 'GePT <noreply@gept.gg>';

      const { error } = await resend.emails.send({
        from: fromAddress,
        to: user.email,
        subject: 'Reset your GePT password',
        html: `
          <div style="font-family: sans-serif; max-width: 480px; margin: 0 auto;">
            <h2>Password Reset</h2>
            <p>You requested a password reset for your GePT account.</p>
            <p>
              <a href="${url}" style="display: inline-block; padding: 12px 24px; background: #2563eb; color: #fff; text-decoration: none; border-radius: 6px;">
                Reset Password
              </a>
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
