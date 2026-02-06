// Shared price history fetcher with module-level cache

export interface PriceHistoryData {
  highs: number[];
  lows: number[];
}

const priceHistoryCache = new Map<number, PriceHistoryData>();

const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

const getSparklineDemoDelayMs = () => {
  const isDev = typeof import.meta !== 'undefined' && (import.meta as any).env?.DEV;
  if (!isDev) return 0;
  if (typeof window === 'undefined') return 0;
  try {
    const params = new URLSearchParams(window.location.search);
    const raw = params.get('sparklineDelay') ?? params.get('sparklineDemo');
    if (!raw) return 0;
    if (raw === '1' || raw.toLowerCase() === 'true') return 1200;
    const n = Number(raw);
    if (!Number.isFinite(n) || n <= 0) return 0;
    return Math.min(Math.round(n), 10_000);
  } catch {
    return 0;
  }
};

const generateMockHistory = (itemId: number, points = 28): PriceHistoryData => {
  const highs: number[] = [];
  const lows: number[] = [];
  const base = 90 + (itemId % 40) * 3;
  const phase = (itemId % 7) * 0.6;
  for (let i = 0; i < points; i += 1) {
    const t = i / (points - 1);
    const wave = Math.sin(t * Math.PI * 2 + phase) * 0.06;
    const noise = Math.sin(i * 1.8 + itemId * 0.17) * 0.025;
    const mid = base * (1 + wave + noise);
    highs.push(Number((mid * 1.02).toFixed(2)));
    lows.push(Number((mid * 0.98).toFixed(2)));
  }
  return { highs, lows };
};

export async function fetchPriceHistory(itemId: number): Promise<PriceHistoryData | null> {
  if (priceHistoryCache.has(itemId)) return priceHistoryCache.get(itemId)!;
  try {
    const res = await fetch(`/api/items/price-history/${itemId}`);
    if (!res.ok) return null;
    const data = await res.json();
    if (data.success && data.data?.highs?.length && data.data?.lows?.length) {
      const result: PriceHistoryData = {
        highs: data.data.highs,
        lows: data.data.lows,
      };
      priceHistoryCache.set(itemId, result);
      return result;
    }
    return null;
  } catch {
    return null;
  }
}

export async function fetchPriceHistoryWithFallback(itemId: number): Promise<PriceHistoryData | null> {
  const isDev = typeof import.meta !== 'undefined' && (import.meta as any).env?.DEV;
  const delayMs = getSparklineDemoDelayMs();
  const data = await fetchPriceHistory(itemId);
  if (data) {
    if (delayMs) await sleep(delayMs);
    return data;
  }
  if (isDev) {
    const mock = generateMockHistory(itemId);
    priceHistoryCache.set(itemId, mock);
    if (delayMs) await sleep(delayMs);
    return mock;
  }
  return null;
}
