import type { ActiveTrade } from './db';
import type { Opportunity } from './trade-types';

const now = Date.now();

let mockTrades: ActiveTrade[] = [
  {
    id: 'mock-trade-1',
    user_id: 'dev-user',
    item_id: 560,
    item_name: 'Death rune',
    buy_price: 196,
    sell_price: 212,
    quantity: 7000,
    rec_id: 'opp-1',
    model_id: 'v2',
    created_at: new Date(now - 6 * 60 * 60 * 1000),
    phase: 'buying',
    progress: 35,
    last_check_in: new Date(now - 30 * 60 * 1000),
    next_check_in: new Date(now + 60 * 60 * 1000),
    actual_buy_price: null,
    actual_sell_price: null,
    expected_hours: 4,
    suggested_sell_price: null,
    confidence: 'high',
    fill_probability: 0.42,
    expected_profit: 112000,
  },
  {
    id: 'mock-trade-2',
    user_id: 'dev-user',
    item_id: 563,
    item_name: 'Law rune',
    buy_price: 178,
    sell_price: 192,
    quantity: 9000,
    rec_id: 'opp-2',
    model_id: 'v2',
    created_at: new Date(now - 4.5 * 60 * 60 * 1000),
    phase: 'selling',
    progress: 70,
    last_check_in: new Date(now - 90 * 60 * 1000),
    next_check_in: new Date(now + 45 * 60 * 1000),
    actual_buy_price: 179,
    actual_sell_price: null,
    expected_hours: 6,
    suggested_sell_price: 194,
    confidence: 'medium',
    fill_probability: 0.33,
    expected_profit: 126000,
  },
  {
    id: 'mock-trade-3',
    user_id: 'dev-user',
    item_id: 445,
    item_name: 'Saradomin brew(4)',
    buy_price: 7260,
    sell_price: 7520,
    quantity: 220,
    rec_id: 'opp-3',
    model_id: 'v3',
    created_at: new Date(now - 1.2 * 60 * 60 * 1000),
    phase: 'buying',
    progress: 55,
    last_check_in: new Date(now - 25 * 60 * 1000),
    next_check_in: new Date(now + 50 * 60 * 1000),
    actual_buy_price: null,
    actual_sell_price: null,
    expected_hours: 3,
    suggested_sell_price: 7480,
    confidence: 'high',
    fill_probability: 0.51,
    expected_profit: 57200,
  },
];

let mockOpportunities: Opportunity[] = [
  {
    id: 'opp-1',
    itemId: 562,
    item: 'Chaos rune',
    buyPrice: 92,
    sellPrice: 104,
    quantity: 12000,
    capitalRequired: 1104000,
    expectedProfit: 144000,
    expectedHours: 3,
    confidence: 'high',
    fillProbability: 0.42,
    volume24h: 1450000,
    trend: 'up',
    whyChips: [
      { icon: '⚡', label: 'Fast fill', type: 'positive' },
      { icon: '🔥', label: 'High volume', type: 'positive' },
      { icon: '🎯', label: 'High confidence', type: 'positive' },
    ],
    category: 'Runes',
    modelId: 'v2',
  },
  {
    id: 'opp-2',
    itemId: 2366,
    item: 'Shield left half',
    buyPrice: 67700,
    sellPrice: 72200,
    quantity: 14,
    capitalRequired: 947800,
    expectedProfit: 63000,
    expectedHours: 6,
    confidence: 'medium',
    fillProbability: 0.28,
    volume24h: 52000,
    trend: 'stable',
    whyChips: [
      { icon: '📊', label: 'Good volume', type: 'neutral' },
      { icon: '🕐', label: 'Longer hold', type: 'neutral' },
    ],
    category: 'Rare',
    modelId: 'v2',
  },
  {
    id: 'opp-3',
    itemId: 1513,
    item: 'Magic logs',
    buyPrice: 995,
    sellPrice: 1065,
    quantity: 1600,
    capitalRequired: 1592000,
    expectedProfit: 112000,
    expectedHours: 2,
    confidence: 'high',
    fillProbability: 0.36,
    volume24h: 840000,
    trend: 'up',
    whyChips: [
      { icon: '🎯', label: 'High confidence', type: 'positive' },
      { icon: '⏱', label: 'Quick flip', type: 'positive' },
    ],
    category: 'Resources',
    modelId: 'v3',
  },
  {
    id: 'opp-4',
    itemId: 1123,
    item: 'Rune platebody',
    buyPrice: 38800,
    sellPrice: 40200,
    quantity: 60,
    capitalRequired: 2328000,
    expectedProfit: 84000,
    expectedHours: 4,
    confidence: 'medium',
    fillProbability: 0.3,
    volume24h: 190000,
    trend: 'stable',
    whyChips: [
      { icon: '📊', label: 'Good volume', type: 'neutral' },
      { icon: '⚡', label: 'Fast fill', type: 'positive' },
    ],
    category: 'Armor',
    modelId: 'v3',
  },
  {
    id: 'opp-5',
    itemId: 9244,
    item: 'Dragon bolts (e)',
    buyPrice: 5050,
    sellPrice: 5320,
    quantity: 450,
    capitalRequired: 2272500,
    expectedProfit: 121500,
    expectedHours: 5,
    confidence: 'low',
    fillProbability: 0.22,
    volume24h: 120000,
    trend: 'down',
    whyChips: [
      { icon: '📉', label: 'Trending down', type: 'negative' },
      { icon: '🕐', label: 'Longer hold', type: 'neutral' },
    ],
    category: 'Ammo',
    modelId: 'v2',
  },
];

const generateId = () => `mock-${Date.now()}-${Math.floor(Math.random() * 1000)}`;

export const getMockTrades = () => mockTrades;

export const findMockTrade = (id: string) => mockTrades.find((trade) => trade.id === id);

export const createMockTrade = (input: {
  user_id: string;
  item_id: number;
  item_name: string;
  buy_price: number;
  sell_price: number;
  quantity: number;
  rec_id?: string | null;
  model_id?: string | null;
  expected_hours?: number;
  confidence?: string | null;
  fill_probability?: number | null;
  expected_profit?: number | null;
}) => {
  const expectedHours = input.expected_hours ?? 4;
  const nextCheckIn = new Date(Date.now() + expectedHours * 0.25 * 60 * 60 * 1000);
  const trade: ActiveTrade = {
    id: generateId(),
    user_id: input.user_id,
    item_id: input.item_id,
    item_name: input.item_name,
    buy_price: input.buy_price,
    sell_price: input.sell_price,
    quantity: input.quantity,
    rec_id: input.rec_id ?? null,
    model_id: input.model_id ?? null,
    created_at: new Date(),
    phase: 'buying',
    progress: 0,
    last_check_in: null,
    next_check_in: nextCheckIn,
    actual_buy_price: null,
    actual_sell_price: null,
    expected_hours: expectedHours,
    suggested_sell_price: null,
    confidence: input.confidence ?? null,
    fill_probability: input.fill_probability ?? null,
    expected_profit: input.expected_profit ?? null,
  };
  mockTrades = [trade, ...mockTrades];
  return trade;
};

export const updateMockTradeProgress = (id: string, progress: number, nextCheckIn: Date) => {
  mockTrades = mockTrades.map((trade) =>
    trade.id === id
      ? { ...trade, progress, last_check_in: new Date(), next_check_in: nextCheckIn }
      : trade
  );
  return findMockTrade(id);
};

export const updateMockTradeSellPrice = (id: string, sellPrice: number) => {
  mockTrades = mockTrades.map((trade) =>
    trade.id === id ? { ...trade, sell_price: sellPrice } : trade
  );
  return findMockTrade(id);
};

export const updateMockTradeQuantity = (id: string, quantity: number) => {
  mockTrades = mockTrades.map((trade) =>
    trade.id === id ? { ...trade, quantity } : trade
  );
  return findMockTrade(id);
};

export const deleteMockTrade = (id: string) => {
  mockTrades = mockTrades.filter((trade) => trade.id !== id);
};

export const advanceMockTrade = (id: string) => {
  const trade = findMockTrade(id);
  if (!trade) return { trade: null, message: 'Trade not found' };
  if (trade.phase === 'buying') {
    const updated = { ...trade, phase: 'selling' as const, progress: 0, next_check_in: new Date(Date.now() + 60 * 60 * 1000) };
    mockTrades = mockTrades.map((t) => (t.id === id ? updated : t));
    return { trade: updated, message: 'Advanced to selling phase' };
  }
  deleteMockTrade(id);
  return { trade: null, message: 'Trade completed', profit: (trade.sell_price - trade.buy_price) * trade.quantity };
};

export const getMockOpportunities = (filters: {
  profitMin?: number;
  profitMax?: number;
  timeMax?: number;
  confidence?: string[] | string;
  capitalMax?: number;
  category?: string;
  limit?: number;
  offset?: number;
}) => {
  let items = [...mockOpportunities];

  if (filters.profitMin !== undefined) {
    items = items.filter((item) => item.expectedProfit >= filters.profitMin!);
  }
  if (filters.profitMax !== undefined) {
    items = items.filter((item) => item.expectedProfit <= filters.profitMax!);
  }
  if (filters.timeMax !== undefined) {
    items = items.filter((item) => item.expectedHours <= filters.timeMax!);
  }
  if (filters.capitalMax !== undefined) {
    items = items.filter((item) => item.capitalRequired <= filters.capitalMax!);
  }
  if (filters.category) {
    items = items.filter((item) => item.category === filters.category);
  }
  if (filters.confidence) {
    const allowed = Array.isArray(filters.confidence) ? filters.confidence : [filters.confidence];
    items = items.filter((item) => allowed.includes(item.confidence));
  }

  const total = items.length;
  const offset = filters.offset ?? 0;
  const limit = filters.limit ?? 30;
  const sliced = items.slice(offset, offset + limit);

  return { items: sliced, total, hasMore: offset + limit < total };
};
