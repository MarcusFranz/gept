// @ts-check
import { defineConfig } from 'astro/config';
import solidJs from '@astrojs/solid-js';
import vercel from '@astrojs/vercel';

// https://astro.build/config
export default defineConfig({
  integrations: [solidJs()],
  output: 'server',
  adapter: vercel({
    webAnalytics: { enabled: true },
    maxDuration: 30,
    imageService: true, // Enable Vercel Image Optimization
    isr: {
      // Enable ISR for pages that can be cached
      expiration: 60, // Revalidate every 60 seconds
      bypassToken: process.env.ISR_BYPASS_TOKEN // Optional bypass token
    }
  }),
  image: {
    // Configure image optimization
    domains: ['gept.gg'],
    remotePatterns: [{ protocol: 'https' }]
  },
  vite: {
    build: {
      // Improve chunk splitting for better caching
      rollupOptions: {
        output: {
          manualChunks: {
            'solid': ['solid-js', 'solid-js/web'],
            'auth': ['better-auth']
          }
        }
      }
    }
  }
});
