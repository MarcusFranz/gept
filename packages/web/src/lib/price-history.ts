// Shared price history fetcher with module-level cache

const priceHistoryCache = new Map<number, number[]>();

export async function fetchPriceHistory(itemId: number): Promise<number[] | null> {
  if (priceHistoryCache.has(itemId)) return priceHistoryCache.get(itemId)!;
  try {
    const res = await fetch(`/api/items/price-history/${itemId}`);
    if (!res.ok) return null;
    const data = await res.json();
    if (data.success && data.data?.prices?.length) {
      priceHistoryCache.set(itemId, data.data.prices);
      return data.data.prices;
    }
    return null;
  } catch {
    return null;
  }
}
