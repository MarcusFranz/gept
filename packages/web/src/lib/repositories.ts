import { sql, type User, type ActiveTrade, type TradeHistory, type Session, type Feedback } from './db';
import { generateId, type PortfolioStats, rateLimits } from './types';

// User repository
export const userRepo = {
  async findById(id: string): Promise<User | undefined> {
    const result = await sql<User>`SELECT * FROM users WHERE id = ${id}`;
    return result[0];
  },

  async findByEmail(email: string): Promise<User | undefined> {
    const result = await sql<User>`SELECT * FROM users WHERE email = ${email}`;
    return result[0];
  },

  async create(id: string, email: string, name?: string): Promise<User> {
    const today = new Date().toISOString().split('T')[0];
    await sql`
      INSERT INTO users (id, email, name, daily_reset_date, weekly_reset_date)
      VALUES (${id}, ${email}, ${name || null}, ${today}, ${today})
    `;
    return (await this.findById(id))!;
  },

  async findOrCreate(id: string, email: string, name?: string): Promise<User> {
    const existing = await this.findById(id);
    if (existing) return existing;
    return this.create(id, email, name);
  },

  async update(id: string, data: Partial<Omit<User, 'id' | 'created_at' | 'updated_at'>>): Promise<boolean> {
    const keys = Object.keys(data) as (keyof typeof data)[];
    if (keys.length === 0) return false;

    // Use COALESCE-based update for safe dynamic updates
    await sql`
      UPDATE users SET
        capital = COALESCE(${(data as Record<string, unknown>).capital ?? null}, capital),
        style = COALESCE(${(data as Record<string, unknown>).style ?? null}, style),
        risk = COALESCE(${(data as Record<string, unknown>).risk ?? null}, risk),
        margin = COALESCE(${(data as Record<string, unknown>).margin ?? null}, margin),
        slots = COALESCE(${(data as Record<string, unknown>).slots ?? null}, slots),
        min_roi = COALESCE(${(data as Record<string, unknown>).min_roi ?? null}, min_roi),
        tier = COALESCE(${(data as Record<string, unknown>).tier ?? null}, tier),
        tutorial_completed = COALESCE(${(data as Record<string, unknown>).tutorial_completed ?? null}, tutorial_completed),
        updated_at = NOW()
      WHERE id = ${id}
    `;
    return true;
  },

  async updateSettings(id: string, settings: {
    capital?: number;
    style?: 'passive' | 'hybrid' | 'active';
    risk?: 'low' | 'medium' | 'high';
    margin?: 'conservative' | 'moderate' | 'aggressive';
    slots?: number;
    min_roi?: number;
    excluded_items?: string[];
  }): Promise<boolean> {
    // Build dynamic update
    const updates: string[] = [];
    const values: unknown[] = [];
    let idx = 1;

    if (settings.capital !== undefined) {
      updates.push(`capital = $${idx++}`);
      values.push(settings.capital);
    }
    if (settings.style !== undefined) {
      updates.push(`style = $${idx++}`);
      values.push(settings.style);
    }
    if (settings.risk !== undefined) {
      updates.push(`risk = $${idx++}`);
      values.push(settings.risk);
    }
    if (settings.margin !== undefined) {
      updates.push(`margin = $${idx++}`);
      values.push(settings.margin);
    }
    if (settings.slots !== undefined) {
      updates.push(`slots = $${idx++}`);
      values.push(settings.slots);
    }
    if (settings.min_roi !== undefined) {
      updates.push(`min_roi = $${idx++}`);
      values.push(settings.min_roi);
    }
    if (settings.excluded_items !== undefined) {
      updates.push(`excluded_items = $${idx++}`);
      values.push(JSON.stringify(settings.excluded_items));
    }

    if (updates.length === 0) return false;

    await sql`
      UPDATE users SET
        capital = COALESCE(${settings.capital ?? null}, capital),
        style = COALESCE(${settings.style ?? null}, style),
        risk = COALESCE(${settings.risk ?? null}, risk),
        margin = COALESCE(${settings.margin ?? null}, margin),
        slots = COALESCE(${settings.slots ?? null}, slots),
        min_roi = COALESCE(${settings.min_roi ?? null}, min_roi),
        excluded_items = COALESCE(${settings.excluded_items ? JSON.stringify(settings.excluded_items) : null}, excluded_items)
      WHERE id = ${id}
    `;
    return true;
  },

  async consumeQuery(id: string): Promise<{ allowed: boolean; remaining: number }> {
    const user = await this.findById(id);
    if (!user) return { allowed: false, remaining: 0 };

    const today = new Date().toISOString().split('T')[0];
    const limit = rateLimits[user.tier]?.daily || rateLimits.beta.daily;

    // Reset daily count if new day
    if (user.daily_reset_date !== today) {
      await sql`
        UPDATE users SET daily_query_count = 0, daily_reset_date = ${today} WHERE id = ${id}
      `;
    }

    const currentCount = user.daily_reset_date === today ? user.daily_query_count : 0;

    if (currentCount >= limit) {
      return { allowed: false, remaining: 0 };
    }

    await sql`
      UPDATE users SET
        daily_query_count = daily_query_count + 1,
        lifetime_queries = lifetime_queries + 1
      WHERE id = ${id}
    `;

    return { allowed: true, remaining: limit - currentCount - 1 };
  },

  async getRateLimitInfo(id: string) {
    const user = await this.findById(id);
    if (!user) return null;

    const limits = rateLimits[user.tier] || rateLimits.beta;
    const today = new Date().toISOString().split('T')[0];
    const dailyUsed = user.daily_reset_date === today ? user.daily_query_count : 0;

    return {
      daily: {
        used: dailyUsed,
        limit: limits.daily,
        remaining: Math.max(0, limits.daily - dailyUsed),
        resetAt: new Date(today + 'T24:00:00Z').toISOString()
      },
      tier: user.tier
    };
  }
};

// Active trades repository
export const activeTradesRepo = {
  async findByUserId(userId: string): Promise<ActiveTrade[]> {
    const result = await sql<ActiveTrade>`
      SELECT * FROM active_trades WHERE user_id = ${userId} ORDER BY created_at DESC
    `;
    return result;
  },

  async findById(id: string): Promise<ActiveTrade | undefined> {
    const result = await sql<ActiveTrade>`SELECT * FROM active_trades WHERE id = ${id}`;
    return result[0];
  },

  async count(userId: string): Promise<number> {
    const result = await sql`SELECT COUNT(*) as count FROM active_trades WHERE user_id = ${userId}`;
    return Number(result[0]?.count || 0);
  },

  async create(trade: Omit<ActiveTrade, 'id' | 'created_at' | 'phase' | 'progress' | 'last_check_in' | 'next_check_in' | 'actual_buy_price' | 'actual_sell_price'> & { expected_hours?: number }): Promise<ActiveTrade> {
    const id = generateId();
    // Calculate first check-in at 25% of expected time (or 1 hour if not specified)
    const expectedHours = trade.expected_hours || 4;
    const firstCheckInMs = expectedHours * 0.25 * 60 * 60 * 1000;
    const nextCheckIn = new Date(Date.now() + firstCheckInMs);

    await sql`
      INSERT INTO active_trades (id, user_id, item_id, item_name, buy_price, sell_price, quantity, rec_id, model_id, phase, progress, next_check_in, expected_hours, confidence, fill_probability, expected_profit)
      VALUES (${id}, ${trade.user_id}, ${trade.item_id}, ${trade.item_name}, ${trade.buy_price}, ${trade.sell_price}, ${trade.quantity}, ${trade.rec_id || null}, ${trade.model_id || null}, 'buying', 0, ${nextCheckIn.toISOString()}, ${expectedHours}, ${trade.confidence || null}, ${trade.fill_probability ?? null}, ${trade.expected_profit ?? null})
    `;
    return (await this.findById(id))!;
  },

  async updateQuantity(id: string, quantity: number): Promise<boolean> {
    await sql`UPDATE active_trades SET quantity = ${quantity} WHERE id = ${id}`;
    return true;
  },

  async updateSellPrice(id: string, sellPrice: number): Promise<boolean> {
    await sql`UPDATE active_trades SET sell_price = ${sellPrice}, suggested_sell_price = NULL WHERE id = ${id}`;
    return true;
  },

  async setSuggestedSellPrice(id: string, price: number): Promise<boolean> {
    await sql`UPDATE active_trades SET suggested_sell_price = ${price} WHERE id = ${id}`;
    return true;
  },

  async clearSuggestedSellPrice(id: string): Promise<boolean> {
    await sql`UPDATE active_trades SET suggested_sell_price = NULL WHERE id = ${id}`;
    return true;
  },

  async delete(id: string): Promise<boolean> {
    await sql`DELETE FROM active_trades WHERE id = ${id}`;
    return true;
  },

  async updatePhase(id: string, phase: 'buying' | 'selling'): Promise<boolean> {
    await sql`
      UPDATE active_trades
      SET phase = ${phase}, progress = 0, last_check_in = NOW()
      WHERE id = ${id}
    `;
    return true;
  },

  async updateProgress(id: string, progress: number, nextCheckIn: Date): Promise<ActiveTrade | undefined> {
    await sql`
      UPDATE active_trades
      SET progress = ${progress}, last_check_in = NOW(), next_check_in = ${nextCheckIn.toISOString()}
      WHERE id = ${id}
    `;
    return this.findById(id);
  },

  async updatePrice(id: string, priceType: 'buy' | 'sell', price: number): Promise<boolean> {
    if (priceType === 'buy') {
      await sql`UPDATE active_trades SET actual_buy_price = ${price} WHERE id = ${id}`;
    } else {
      await sql`UPDATE active_trades SET actual_sell_price = ${price} WHERE id = ${id}`;
    }
    return true;
  },

  async findNeedingCheckIn(userId: string): Promise<ActiveTrade[]> {
    const result = await sql<ActiveTrade>`
      SELECT * FROM active_trades
      WHERE user_id = ${userId}
        AND next_check_in IS NOT NULL
        AND next_check_in <= NOW()
      ORDER BY next_check_in ASC
    `;
    return result;
  },

  async getTiedCapital(userId: string): Promise<number> {
    const result = await sql`
      SELECT COALESCE(SUM(buy_price * quantity), 0) as total
      FROM active_trades WHERE user_id = ${userId}
    `;
    return Number(result[0]?.total || 0);
  }
};

// Trade history repository
export const tradeHistoryRepo = {
  async findByUserId(userId: string, options?: {
    limit?: number;
    offset?: number;
    filter?: 'all' | 'wins' | 'losses' | 'cancelled';
    itemId?: number;
  }): Promise<TradeHistory[]> {
    let result;

    if (options?.filter === 'wins') {
      result = await sql`
        SELECT * FROM trade_history
        WHERE user_id = ${userId} AND profit > 0 AND status = 'completed'
        ORDER BY created_at DESC
        LIMIT ${options?.limit || 100} OFFSET ${options?.offset || 0}
      `;
    } else if (options?.filter === 'losses') {
      result = await sql`
        SELECT * FROM trade_history
        WHERE user_id = ${userId} AND profit < 0 AND status = 'completed'
        ORDER BY created_at DESC
        LIMIT ${options?.limit || 100} OFFSET ${options?.offset || 0}
      `;
    } else if (options?.filter === 'cancelled') {
      result = await sql`
        SELECT * FROM trade_history
        WHERE user_id = ${userId} AND status = 'cancelled'
        ORDER BY created_at DESC
        LIMIT ${options?.limit || 100} OFFSET ${options?.offset || 0}
      `;
    } else if (options?.itemId) {
      result = await sql`
        SELECT * FROM trade_history
        WHERE user_id = ${userId} AND item_id = ${options.itemId}
        ORDER BY created_at DESC
        LIMIT ${options?.limit || 100} OFFSET ${options?.offset || 0}
      `;
    } else {
      result = await sql`
        SELECT * FROM trade_history
        WHERE user_id = ${userId}
        ORDER BY created_at DESC
        LIMIT ${options?.limit || 100} OFFSET ${options?.offset || 0}
      `;
    }

    return result as unknown as TradeHistory[];
  },

  async count(userId: string, filter?: 'all' | 'wins' | 'losses' | 'cancelled'): Promise<number> {
    let result;

    if (filter === 'wins') {
      result = await sql`
        SELECT COUNT(*) as count FROM trade_history
        WHERE user_id = ${userId} AND profit > 0 AND status = 'completed'
      `;
    } else if (filter === 'losses') {
      result = await sql`
        SELECT COUNT(*) as count FROM trade_history
        WHERE user_id = ${userId} AND profit < 0 AND status = 'completed'
      `;
    } else if (filter === 'cancelled') {
      result = await sql`
        SELECT COUNT(*) as count FROM trade_history
        WHERE user_id = ${userId} AND status = 'cancelled'
      `;
    } else {
      result = await sql`
        SELECT COUNT(*) as count FROM trade_history WHERE user_id = ${userId}
      `;
    }

    return Number(result[0]?.count || 0);
  },

  async create(trade: Omit<TradeHistory, 'id' | 'created_at'>): Promise<TradeHistory> {
    const id = generateId();
    await sql`
      INSERT INTO trade_history (id, user_id, item_id, item_name, buy_price, sell_price, quantity, profit, notes, rec_id, model_id, status, expected_profit, confidence, fill_probability, expected_hours)
      VALUES (${id}, ${trade.user_id}, ${trade.item_id ?? null}, ${trade.item_name}, ${trade.buy_price ?? null}, ${trade.sell_price ?? null}, ${trade.quantity ?? null}, ${trade.profit}, ${trade.notes || null}, ${trade.rec_id || null}, ${trade.model_id || null}, ${trade.status || 'completed'}, ${trade.expected_profit ?? null}, ${trade.confidence || null}, ${trade.fill_probability ?? null}, ${trade.expected_hours ?? null})
    `;
    const result = await sql<TradeHistory>`SELECT * FROM trade_history WHERE id = ${id}`;
    return result[0];
  },

  async getStats(userId: string, timeframe?: 'today' | 'week' | 'month' | 'all'): Promise<PortfolioStats> {
    let dateFilter = '';
    const now = new Date();

    if (timeframe === 'today') {
      dateFilter = `AND created_at >= CURRENT_DATE`;
    } else if (timeframe === 'week') {
      dateFilter = `AND created_at >= CURRENT_DATE - INTERVAL '7 days'`;
    } else if (timeframe === 'month') {
      dateFilter = `AND created_at >= CURRENT_DATE - INTERVAL '30 days'`;
    }

    // Basic stats
    let stats;
    if (timeframe === 'today') {
      stats = await sql`
        SELECT
          COUNT(*) as total_trades,
          COALESCE(SUM(profit), 0) as total_profit,
          SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as win_count,
          SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) as loss_count,
          COALESCE(AVG(profit), 0) as avg_profit
        FROM trade_history
        WHERE user_id = ${userId} AND status = 'completed' AND created_at >= CURRENT_DATE
      `;
    } else if (timeframe === 'week') {
      stats = await sql`
        SELECT
          COUNT(*) as total_trades,
          COALESCE(SUM(profit), 0) as total_profit,
          SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as win_count,
          SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) as loss_count,
          COALESCE(AVG(profit), 0) as avg_profit
        FROM trade_history
        WHERE user_id = ${userId} AND status = 'completed' AND created_at >= CURRENT_DATE - INTERVAL '7 days'
      `;
    } else if (timeframe === 'month') {
      stats = await sql`
        SELECT
          COUNT(*) as total_trades,
          COALESCE(SUM(profit), 0) as total_profit,
          SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as win_count,
          SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) as loss_count,
          COALESCE(AVG(profit), 0) as avg_profit
        FROM trade_history
        WHERE user_id = ${userId} AND status = 'completed' AND created_at >= CURRENT_DATE - INTERVAL '30 days'
      `;
    } else {
      stats = await sql`
        SELECT
          COUNT(*) as total_trades,
          COALESCE(SUM(profit), 0) as total_profit,
          SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as win_count,
          SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) as loss_count,
          COALESCE(AVG(profit), 0) as avg_profit
        FROM trade_history
        WHERE user_id = ${userId} AND status = 'completed'
      `;
    }

    const statsRow = stats[0] || { total_trades: 0, total_profit: 0, win_count: 0, loss_count: 0, avg_profit: 0 };

    // Best trade
    const bestTrade = await sql`
      SELECT item_name as item, profit
      FROM trade_history
      WHERE user_id = ${userId} AND status = 'completed'
      ORDER BY profit DESC LIMIT 1
    `;

    // Worst trade
    const worstTrade = await sql`
      SELECT item_name as item, profit
      FROM trade_history
      WHERE user_id = ${userId} AND status = 'completed'
      ORDER BY profit ASC LIMIT 1
    `;

    // Most traded item
    const mostTraded = await sql`
      SELECT item_name as item, COUNT(*) as count
      FROM trade_history
      WHERE user_id = ${userId} AND status = 'completed'
      GROUP BY item_name ORDER BY count DESC LIMIT 1
    `;

    // Current streak
    const recentTrades = await sql`
      SELECT profit FROM trade_history
      WHERE user_id = ${userId} AND status = 'completed'
      ORDER BY created_at DESC LIMIT 100
    `;

    let streakType: 'win' | 'loss' = 'win';
    let streakCount = 0;

    if (recentTrades.length > 0) {
      streakType = Number(recentTrades[0].profit) > 0 ? 'win' : 'loss';
      for (const trade of recentTrades) {
        const isWin = Number(trade.profit) > 0;
        if ((streakType === 'win' && isWin) || (streakType === 'loss' && !isWin)) {
          streakCount++;
        } else {
          break;
        }
      }
    }

    return {
      totalTrades: Number(statsRow.total_trades),
      totalProfit: Number(statsRow.total_profit),
      winCount: Number(statsRow.win_count) || 0,
      lossCount: Number(statsRow.loss_count) || 0,
      averageProfit: Math.round(Number(statsRow.avg_profit)),
      bestTrade: bestTrade[0] ? { item: String(bestTrade[0].item), profit: Number(bestTrade[0].profit) } : null,
      worstTrade: worstTrade[0] ? { item: String(worstTrade[0].item), profit: Number(worstTrade[0].profit) } : null,
      mostTradedItem: mostTraded[0] ? { item: String(mostTraded[0].item), count: Number(mostTraded[0].count) } : null,
      currentStreak: { type: streakType, count: streakCount },
      winRate: Number(statsRow.total_trades) > 0 ? (Number(statsRow.win_count) || 0) / Number(statsRow.total_trades) : 0
    };
  }
};

// Session repository (for legacy compatibility - Better Auth handles sessions)
export const sessionRepo = {
  async findById(id: string): Promise<Session | undefined> {
    const result = await sql<Session>`SELECT * FROM sessions WHERE id = ${id}`;
    const session = result[0];
    if (session && new Date(session.expires_at) < new Date()) {
      await this.delete(id);
      return undefined;
    }
    return session;
  },

  async findByToken(token: string): Promise<Session | undefined> {
    const result = await sql<Session>`SELECT * FROM sessions WHERE token = ${token}`;
    const session = result[0];
    if (session && new Date(session.expires_at) < new Date()) {
      await this.delete(session.id);
      return undefined;
    }
    return session;
  },

  async create(userId: string, durationDays = 30): Promise<Session> {
    const id = generateId();
    const token = generateId() + generateId();
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + durationDays);

    await sql`
      INSERT INTO sessions (id, user_id, token, expires_at)
      VALUES (${id}, ${userId}, ${token}, ${expiresAt.toISOString()})
    `;

    return (await this.findById(id))!;
  },

  async delete(id: string): Promise<boolean> {
    await sql`DELETE FROM sessions WHERE id = ${id}`;
    return true;
  },

  async deleteByUserId(userId: string): Promise<number> {
    await sql`DELETE FROM sessions WHERE user_id = ${userId}`;
    return 1;
  },

  async cleanup(): Promise<number> {
    await sql`DELETE FROM sessions WHERE expires_at < NOW()`;
    return 1;
  }
};

// Feedback repository
export const feedbackRepo = {
  async create(feedback: {
    userId: string;
    type: 'bug' | 'feature' | 'general' | 'recommendation';
    rating?: 'positive' | 'negative';
    message?: string;
    email?: string;
    recommendation?: {
      id: string;
      itemId: number;
      item: string;
      buyPrice: number;
      sellPrice: number;
      quantity: number;
      expectedProfit: number;
      confidence: number;
      modelId: string;
    };
  }): Promise<Feedback> {
    const id = generateId();
    const rec = feedback.recommendation;

    await sql`
      INSERT INTO feedback (
        id, user_id, type, rating, message, email,
        rec_id, rec_item_id, rec_item_name, rec_buy_price, rec_sell_price,
        rec_quantity, rec_expected_profit, rec_confidence, rec_model_id
      )
      VALUES (
        ${id},
        ${feedback.userId},
        ${feedback.type},
        ${feedback.rating || null},
        ${feedback.message || null},
        ${feedback.email || null},
        ${rec?.id || null},
        ${rec?.itemId ?? null},
        ${rec?.item || null},
        ${rec?.buyPrice ?? null},
        ${rec?.sellPrice ?? null},
        ${rec?.quantity ?? null},
        ${rec?.expectedProfit ?? null},
        ${rec?.confidence ?? null},
        ${rec?.modelId || null}
      )
    `;

    const result = await sql<Feedback>`SELECT * FROM feedback WHERE id = ${id}`;
    return result[0];
  },

  async findByUserId(userId: string, limit = 50): Promise<Feedback[]> {
    const result = await sql<Feedback>`
      SELECT * FROM feedback
      WHERE user_id = ${userId}
      ORDER BY created_at DESC
      LIMIT ${limit}
    `;
    return result;
  },

  async countByType(type?: 'bug' | 'feature' | 'general' | 'recommendation'): Promise<number> {
    let result;
    if (type) {
      result = await sql`SELECT COUNT(*) as count FROM feedback WHERE type = ${type}`;
    } else {
      result = await sql`SELECT COUNT(*) as count FROM feedback`;
    }
    return Number(result[0]?.count || 0);
  }
};
