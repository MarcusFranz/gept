import { createAuthClient } from 'better-auth/solid';

export const authClient = createAuthClient({
  baseURL: import.meta.env.PUBLIC_APP_URL || 'http://localhost:3000',
});

// Export convenience methods
export const { signIn, signUp, signOut, useSession, requestPasswordReset, resetPassword } = authClient;
