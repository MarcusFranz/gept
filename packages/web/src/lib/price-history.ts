// Shared price history fetcher with module-level cache

export interface PriceHistoryData {
  highs: number[];
  lows: number[];
}

const priceHistoryCache = new Map<number, PriceHistoryData>();

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
