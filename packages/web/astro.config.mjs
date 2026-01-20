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
    // ISR disabled - was causing API routes to cache and ignore query params
    // isr: {
    //   expiration: 60,
    //   bypassToken: process.env.ISR_BYPASS_TOKEN
    // }
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
