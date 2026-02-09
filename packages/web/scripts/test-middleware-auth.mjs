import { spawn } from 'node:child_process';
import path from 'node:path';
import { createHmac } from 'node:crypto';
import { rm } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';

function run(cmd, args, opts = {}) {
  return spawn(cmd, args, {
    stdio: ['ignore', 'pipe', 'pipe'],
    ...opts,
  });
}

async function sleep(ms) {
  return await new Promise((r) => setTimeout(r, ms));
}

async function waitForHttpOk(url, { timeoutMs = 15_000 } = {}) {
  const started = Date.now();
  let lastErr = null;
  while (Date.now() - started < timeoutMs) {
    try {
      const res = await fetch(url, { redirect: 'manual' });
      if (res.status >= 200 && res.status < 500) return;
      lastErr = new Error(`unexpected status ${res.status}`);
    } catch (e) {
      lastErr = e;
    }
    await sleep(200);
  }
  throw lastErr ?? new Error(`timed out waiting for ${url}`);
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

async function main() {
  // Ensure webhook verification uses a runtime env var (server-side best practice).
  // This is a test-only value (not a secret).
  process.env.WEBHOOK_SECRET = 'test-webhook-secret-min-32-chars-xxxxxxxx';

  // Build first to ensure we're exercising production-mode output.
  // Astro's Vercel adapter build hook can error if `.vercel/output` already exists
  // from a previous run, so we clean it to keep this test idempotent.
  const here = path.dirname(fileURLToPath(import.meta.url));
  await rm(path.resolve(here, '..', '.vercel', 'output'), { recursive: true, force: true });

  const build = run('npm', ['run', 'build'], {
    env: { ...process.env, NODE_ENV: 'production' },
  });

  let buildStdout = '';
  let buildStderr = '';
  build.stdout.on('data', (d) => (buildStdout += d.toString()));
  build.stderr.on('data', (d) => (buildStderr += d.toString()));

  const buildExitCode = await new Promise((resolve) => build.on('close', resolve));
  if (buildExitCode !== 0) {
    // Avoid dumping env; include only command output.
    throw new Error(`web build failed (exit ${buildExitCode})\n${buildStdout}\n${buildStderr}`);
  }

  // The Vercel adapter does not support `astro preview`. Instead, we validate
  // production behavior by importing the built internal middleware module from
  // `.vercel/output/...` and invoking it directly with mocked contexts.
  const builtMiddlewarePath = path.resolve(
    here,
    '..',
    '.vercel',
    'output',
    'functions',
    '_render.func',
    'packages',
    'web',
    'dist',
    'server',
    '_astro-internal_middleware.mjs'
  );

  const { onRequest } = await import(pathToFileURL(builtMiddlewarePath).href);
  assert(typeof onRequest === 'function', 'expected built middleware to export onRequest');

  const next = async () => new Response('ok', { status: 200 });
  const redirect = (to) => new Response(null, { status: 302, headers: { location: to } });

  // 1) Protected API route: should be blocked in middleware before route handling.
  {
    const url = new URL('http://example.test/api/trades/does-not-exist');
    const context = {
      request: new Request(url),
      url,
      locals: {},
      redirect,
    };

    const res = await onRequest(context, next);
    assert(res.status === 401, `expected 401 from protected API route, got ${res.status}`);
    const json = await res.json().catch(() => null);
    assert(json && json.success === false && json.error === 'Unauthorized', 'expected {success:false,error:"Unauthorized"}');
  }

  // 2) Protected page: should redirect to /welcome when unauthenticated.
  {
    const url = new URL('http://example.test/');
    const context = {
      request: new Request(url),
      url,
      locals: {},
      redirect,
    };

    const res = await onRequest(context, next);
    assert(res.status >= 300 && res.status < 400, `expected redirect from protected page, got ${res.status}`);
    const loc = res.headers.get('location') ?? '';
    assert(loc === '/welcome', `expected Location: /welcome, got ${loc || '(missing)'}`);
  }

  // 3) Alerts webhook: should accept a correctly-signed payload when WEBHOOK_SECRET
  // is provided via process.env at runtime (not just import.meta.env).
  {
    const builtAlertsPath = path.resolve(
      here,
      '..',
      '.vercel',
      'output',
      'functions',
      '_render.func',
      'packages',
      'web',
      'dist',
      'server',
      'pages',
      'api',
      'webhooks',
      'alerts.astro.mjs'
    );

    const mod = await import(pathToFileURL(builtAlertsPath).href);
    assert(typeof mod.page === 'function', 'expected built alerts route to export page()');
    const route = mod.page();
    const POST = route.POST;
    assert(typeof POST === 'function', 'expected built alerts route to expose POST');

    const payload = {
      userId: 'user_test_1',
      alert: {
        id: 'alert_test_1',
        tradeId: 'trade_test_1',
        type: 'ADJUST_PRICE',
        reason: 'test',
        confidence: 0.5,
        urgency: 'low',
        timestamp: new Date().toISOString(),
      },
    };

    const body = JSON.stringify(payload);
    const timestamp = Date.now().toString();
    const signedPayload = `${timestamp}.${body}`;
    const signature = createHmac('sha256', process.env.WEBHOOK_SECRET)
      .update(signedPayload)
      .digest('hex');

    const req = new Request('http://example.test/api/webhooks/alerts', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Webhook-Timestamp': timestamp,
        'X-Webhook-Signature': signature,
      },
      body,
    });

    const res = await POST({ request: req });
    assert(res.status === 200, `expected 200 from alerts webhook, got ${res.status}`);
    const json = await res.json().catch(() => null);
    assert(json && json.success === true, 'expected {success:true} from alerts webhook');
  }

  // 4) Trades resync endpoint: should not be blocked by middleware, and should
  // accept auth via the fallback header even if Authorization is stripped.
  {
    const builtResyncPath = path.resolve(
      here,
      '..',
      '.vercel',
      'output',
      'functions',
      '_render.func',
      'packages',
      'web',
      'dist',
      'server',
      'pages',
      'api',
      'trades',
      'resync.astro.mjs'
    );

    const mod = await import(pathToFileURL(builtResyncPath).href);
    assert(typeof mod.page === 'function', 'expected built resync route to export page()');
    const route = mod.page();
    const GET = route.GET;
    assert(typeof GET === 'function', 'expected built resync route to expose GET');

    // No auth should fail closed.
    {
      const req = new Request('http://example.test/api/trades/resync', { method: 'GET' });
      const res = await GET({ request: req });
      assert(res.status === 401, `expected 401 from resync without auth, got ${res.status}`);
    }

    // Fallback header should pass auth. The handler will likely 500 in this test
    // environment because DATABASE_URL isn't set; we only assert it's not 401.
    {
      const req = new Request('http://example.test/api/trades/resync', {
        method: 'GET',
        headers: { 'X-Gept-Webhook-Secret': process.env.WEBHOOK_SECRET },
      });
      const res = await GET({ request: req });
      assert(res.status !== 401, `expected resync auth to pass via X-Gept-Webhook-Secret, got ${res.status}`);
    }
  }
}

main().catch((e) => {
  console.error(String(e?.stack || e));
  process.exitCode = 1;
});
