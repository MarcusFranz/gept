import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { neon } from '@neondatabase/serverless';

function parseArgs(argv) {
  const out = { day: null, dryRun: false };
  for (const arg of argv.slice(2)) {
    if (arg === '--dry-run') out.dryRun = true;
    if (arg.startsWith('--day=')) out.day = arg.slice('--day='.length);
  }
  return out;
}

async function fileExists(p) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

function parseDotEnv(contents) {
  const env = {};
  for (const rawLine of contents.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq <= 0) continue;
    const key = line.slice(0, eq).trim();
    let val = line.slice(eq + 1).trim();
    // Remove surrounding quotes if present.
    if (
      (val.startsWith('"') && val.endsWith('"')) ||
      (val.startsWith("'") && val.endsWith("'"))
    ) {
      val = val.slice(1, -1);
    }
    env[key] = val;
  }
  return env;
}

async function loadEnvFallbacks() {
  // Prefer the already-set environment first, but allow local dev env files as a fallback.
  const candidates = [
    path.resolve(process.cwd(), '.env.local'),
    path.resolve(process.cwd(), '.env'),
    path.resolve(process.cwd(), 'packages/web/.env.local'),
    path.resolve(process.cwd(), 'packages/web/.env'),
  ];

  for (const p of candidates) {
    if (process.env.DATABASE_URL) return;
    if (!(await fileExists(p))) continue;
    const parsed = parseDotEnv(await fs.readFile(p, 'utf8'));
    for (const [k, v] of Object.entries(parsed)) {
      if (process.env[k] == null) process.env[k] = v;
    }
  }
}

function utcDayWindow(dayOverride) {
  // Window is [start, end) for a specific UTC day.
  // Default: previous UTC day to make re-runs deterministic.
  const now = new Date();
  const end = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  const start = new Date(end.getTime() - 24 * 60 * 60 * 1000);

  if (!dayOverride) return { day: start.toISOString().slice(0, 10), start, end };

  if (!/^\d{4}-\d{2}-\d{2}$/.test(dayOverride)) {
    throw new Error(`Invalid --day format. Expected YYYY-MM-DD, got: ${dayOverride}`);
  }
  const [y, m, d] = dayOverride.split('-').map(Number);
  const start2 = new Date(Date.UTC(y, m - 1, d));
  const end2 = new Date(Date.UTC(y, m - 1, d + 1));
  return { day: dayOverride, start: start2, end: end2 };
}

function csvEscape(v) {
  const s = String(v ?? '');
  if (/[,"\n]/.test(s)) return `"${s.replaceAll('"', '""')}"`;
  return s;
}

function makeCsvLine(values) {
  return values.map(csvEscape).join(',');
}

async function upsertCsvRow(csvPath, headerCols, dayKey, rowValuesByCol) {
  const header = makeCsvLine(headerCols);
  const dayColIdx = headerCols.indexOf(dayKey);
  if (dayColIdx === -1) throw new Error(`Missing day column "${dayKey}" in header`);

  const desiredLine = makeCsvLine(headerCols.map((c) => rowValuesByCol[c] ?? ''));

  await fs.mkdir(path.dirname(csvPath), { recursive: true });
  const exists = await fileExists(csvPath);
  if (!exists) {
    await fs.writeFile(csvPath, `${header}\n${desiredLine}\n`, 'utf8');
    return { action: 'created', replaced: false };
  }

  const raw = await fs.readFile(csvPath, 'utf8');
  const lines = raw.split('\n').filter((l) => l.trim().length > 0);
  const hasHeader = lines.length > 0 && lines[0].trim() === header.trim();

  const outLines = [];
  outLines.push(header);

  let replaced = false;
  const dataLines = hasHeader ? lines.slice(1) : lines;

  for (const line of dataLines) {
    const cols = line.split(',');
    const dayVal = (cols[dayColIdx] ?? '').replace(/^"|"$/g, '');
    if (dayVal === rowValuesByCol[dayKey]) {
      outLines.push(desiredLine);
      replaced = true;
    } else {
      outLines.push(line);
    }
  }

  if (!replaced) outLines.push(desiredLine);

  await fs.writeFile(csvPath, `${outLines.join('\n')}\n`, 'utf8');
  return { action: 'updated', replaced };
}

async function main() {
  const args = parseArgs(process.argv);

  await loadEnvFallbacks();
  const connectionString = process.env.DATABASE_URL;
  if (!connectionString) {
    throw new Error('DATABASE_URL is not set (and no .env/.env.local fallback provided it).');
  }

  const outputDir = process.env.USAGE_DATA_DIR || '/Users/marcus/code/usage-data';
  const csvPath = process.env.USAGE_CSV_PATH || path.join(outputDir, 'user_activity.csv');

  const { day, start, end } = utcDayWindow(args.day);
  const runAt = new Date();

  const sql = neon(connectionString);
  const startIso = start.toISOString();
  const endIso = end.toISOString();

  const [
    totalUsersRow,
    newUsersRow,
    activeUsersRow,
    sessionsCreatedRow,
    sessionsUpdatedRow,
    tradesCreatedRow,
    completedRow,
    cancelledRow,
    feedbackRow,
    activeTradesNowRow,
    sellingNowRow,
    buyingNowRow,
  ] = await Promise.all([
    sql`SELECT COUNT(*)::int AS n FROM users`,
    sql`SELECT COUNT(*)::int AS n FROM users WHERE created_at >= ${startIso} AND created_at < ${endIso}`,
    sql`SELECT COUNT(DISTINCT user_id)::int AS n FROM sessions WHERE updated_at >= ${startIso} AND updated_at < ${endIso}`,
    sql`SELECT COUNT(*)::int AS n FROM sessions WHERE created_at >= ${startIso} AND created_at < ${endIso}`,
    sql`SELECT COUNT(*)::int AS n FROM sessions WHERE updated_at >= ${startIso} AND updated_at < ${endIso}`,
    sql`SELECT COUNT(*)::int AS n FROM active_trades WHERE created_at >= ${startIso} AND created_at < ${endIso}`,
    sql`SELECT COUNT(*)::int AS n FROM trade_history WHERE status = 'completed' AND created_at >= ${startIso} AND created_at < ${endIso}`,
    sql`SELECT COUNT(*)::int AS n FROM trade_history WHERE status = 'cancelled' AND created_at >= ${startIso} AND created_at < ${endIso}`,
    sql`SELECT COUNT(*)::int AS n FROM feedback WHERE created_at >= ${startIso} AND created_at < ${endIso}`,
    sql`SELECT COUNT(*)::int AS n FROM active_trades`,
    sql`SELECT COUNT(*)::int AS n FROM active_trades WHERE phase = 'selling'`,
    sql`SELECT COUNT(*)::int AS n FROM active_trades WHERE phase = 'buying'`,
  ]);

  const row = {
    day_utc: day,
    window_start_utc: startIso,
    window_end_utc: endIso,
    run_at_utc: runAt.toISOString(),
    total_users: totalUsersRow?.[0]?.n ?? 0,
    new_users: newUsersRow?.[0]?.n ?? 0,
    active_users: activeUsersRow?.[0]?.n ?? 0,
    sessions_created: sessionsCreatedRow?.[0]?.n ?? 0,
    sessions_updated: sessionsUpdatedRow?.[0]?.n ?? 0,
    trades_created: tradesCreatedRow?.[0]?.n ?? 0,
    trades_completed: completedRow?.[0]?.n ?? 0,
    trades_cancelled: cancelledRow?.[0]?.n ?? 0,
    feedback_count: feedbackRow?.[0]?.n ?? 0,
    active_trades_now: activeTradesNowRow?.[0]?.n ?? 0,
    selling_trades_now: sellingNowRow?.[0]?.n ?? 0,
    buying_trades_now: buyingNowRow?.[0]?.n ?? 0,
  };

  const headerCols = Object.keys(row);

  if (args.dryRun) {
    // Intentionally prints only non-sensitive aggregated counts.
    // eslint-disable-next-line no-console
    console.log(`[usage-export] dry-run ${day}:`, row);
    return;
  }

  const res = await upsertCsvRow(csvPath, headerCols, 'day_utc', row);
  // eslint-disable-next-line no-console
  console.log(`[usage-export] ${res.action} ${path.relative(process.cwd(), csvPath)} (${day}${res.replaced ? ', replaced' : ''})`);
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error('[usage-export] failed:', err?.message || err);
  process.exitCode = 1;
});

