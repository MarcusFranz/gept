import { neon, type NeonQueryFunction } from '@neondatabase/serverless';

// Get connection - lazily initialized
let _sql: NeonQueryFunction<false, false> | null = null;

export function sql<T = Record<string, unknown>>(strings: TemplateStringsArray, ...values: unknown[]): Promise<T[]> {
  if (!_sql) {
    const connectionString = process.env.DATABASE_URL;
    if (!connectionString) {
      throw new Error('DATABASE_URL environment variable is not set');
    }
    _sql = neon(connectionString);
  }
  return _sql(strings, ...values) as Promise<T[]>;
}

export default sql;

// Helper types - updated for Better Auth + Neon
export interface User {
  id: string;
  name: string | null;
  email: string;
  email_verified: boolean;
  image: string | null;
  created_at: Date;
  updated_at: Date;
  // GePT custom fields
  capital: number;
  style: 'passive' | 'hybrid' | 'active';
  risk: 'low' | 'medium' | 'high';
  margin: 'conservative' | 'moderate' | 'aggressive';
  slots: number;
  min_roi: number;
  tier: 'beta' | 'free' | 'premium';
  tutorial_completed: boolean;
  excluded_items: string[];
  use_beta_model: boolean;
  daily_query_count: number;
  daily_reset_date: string | null;
  weekly_query_count: number;
  weekly_reset_date: string | null;
  lifetime_queries: number;
}

export interface ActiveTrade {
  id: string;
  user_id: string;
  item_id: number;
  item_name: string;
  buy_price: number;
  sell_price: number;
  quantity: number;
  rec_id: string | null;
  model_id: string | null;
  created_at: Date;
  // Phase tracking fields for trade-centric UI
  phase: 'buying' | 'selling';
  progress: number;
  last_check_in: Date | null;
  next_check_in: Date | null;
  actual_buy_price: number | null;
  actual_sell_price: number | null;
  expected_hours: number | null;
  suggested_sell_price: number | null;
  confidence: string | null;
  fill_probability: number | null;
  expected_profit: number | null;
}

export interface TradeHistory {
  id: string;
  user_id: string;
  item_id: number | null;
  item_name: string;
  buy_price: number | null;
  sell_price: number | null;
  quantity: number | null;
  profit: number;
  notes: string | null;
  rec_id: string | null;
  model_id: string | null;
  status: 'completed' | 'cancelled';
  expected_profit?: number | null;
  confidence?: string | null;
  fill_probability?: number | null;
  expected_hours?: number | null;
  created_at: Date;
}

export interface Session {
  id: string;
  user_id: string;
  token: string;
  expires_at: Date;
  ip_address: string | null;
  user_agent: string | null;
  created_at: Date;
  updated_at: Date;
}

export interface Account {
  id: string;
  user_id: string;
  account_id: string;
  provider_id: string;
  access_token: string | null;
  refresh_token: string | null;
  access_token_expires_at: Date | null;
  refresh_token_expires_at: Date | null;
  scope: string | null;
  id_token: string | null;
  created_at: Date;
  updated_at: Date;
}

export interface Feedback {
  id: string;
  user_id: string;
  type: 'bug' | 'feature' | 'general' | 'recommendation';
  rating: 'positive' | 'negative' | null;
  message: string | null;
  email: string | null;
  // Recommendation context (only for type='recommendation')
  rec_id: string | null;
  rec_item_id: number | null;
  rec_item_name: string | null;
  rec_buy_price: number | null;
  rec_sell_price: number | null;
  rec_quantity: number | null;
  rec_expected_profit: number | null;
  rec_confidence: number | null;
  rec_model_id: string | null;
  created_at: Date;
}
