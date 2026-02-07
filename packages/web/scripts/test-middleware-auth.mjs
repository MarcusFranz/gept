import { spawn } from 'node:child_process';
import path from 'node:path';
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
  // Build first to ensure we're exercising production-mode output.
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
  const here = path.dirname(fileURLToPath(import.meta.url));
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
}

main().catch((e) => {
  console.error(String(e?.stack || e));
  process.exitCode = 1;
});
