# Data Exploration Report - OSRS GE Price Data

## Executive Summary

This report documents the exploration of OSRS Grand Exchange price data stored in PostgreSQL, used for building a flipping prediction system.

**Key Findings:**
- 426M rows of 5-minute price data spanning 5 years (2021-2026)
- 4,384 tradeable items with price data
- 16 high-quality items identified for Tier 1 model training
- Data completeness averages 54% for actively traded items

---

## Database Overview

### Connection Details
- **Host**: localhost:5432 (via SSH tunnel to 150.136.170.128)
- **Database**: osrs_data
- **User**: osrs_user

### Table Inventory

| Table | Row Count | Size | Purpose |
|-------|-----------|------|---------|
| price_data_5min | 426,431,785 | Large | 5-minute OHLCV candles |
| prices_latest | 3,021,954 | 384 MB | 1-minute real-time prices |
| prices_1h | 1,580,003 | 177 MB | Hourly aggregations |
| items | 4,500 | 768 KB | Item metadata |
| reddit_posts | 158,000 | 89 MB | Community sentiment |
| youtube_videos | 7,500 | 7.6 MB | Video content |
| osrs_news | 1,417 | 1.7 MB | News articles |
| monsters | 1,310 | 352 KB | Monster data |
| drop_table | 12,926 | 2.2 MB | Monster drops |

---

## Schema Analysis

### price_data_5min (Primary Training Data)

```sql
CREATE TABLE price_data_5min (
    item_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    avg_high_price INTEGER,     -- Average instant-buy price
    high_price_volume BIGINT,   -- Volume at high price
    avg_low_price INTEGER,      -- Average instant-sell price
    low_price_volume BIGINT     -- Volume at low price
);
```

**Indexes:**
- `idx_price_5min_item` on (item_id)
- `idx_price_5min_unique` on (item_id, timestamp) - UNIQUE
- `price_data_5min_timestamp_idx` on (timestamp)

**Date Range:** 2021-03-08 to 2026-01-07 (1,766 days)

### prices_latest (Real-time Inference Data)

```sql
CREATE TABLE prices_latest (
    item_id INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    high_price INTEGER,     -- Most recent instant-buy price
    high_time TIMESTAMPTZ,  -- When high was recorded
    low_price INTEGER,      -- Most recent instant-sell price
    low_time TIMESTAMPTZ    -- When low was recorded
);
```

**Date Range:** Last 12 hours (continuously updating)

### items (Item Metadata)

```sql
CREATE TABLE items (
    item_id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    examine TEXT,
    members BOOLEAN,
    tradeable BOOLEAN,
    lowalch INTEGER,
    highalch INTEGER,
    cost INTEGER,
    wiki_url TEXT
);
```

---

## Data Quality Analysis

### Completeness Metrics

For the 1,766-day span at 5-minute intervals:
- **Expected intervals per item**: 508,608 (288 per day × 1,766 days)
- **Actual observed**: ~275,000 rows per high-volume item
- **Completeness rate**: ~54%

The 54% completeness is expected because:
1. API data collection started mid-2021
2. Occasional API outages
3. Some items have sporadic trading activity

### High-Volume Item Analysis

Top 15 items by average trading volume:

| Item ID | Name | Avg Volume/5min | Rows | Completeness |
|---------|------|-----------------|------|--------------|
| 556 | Air rune | 240,562 | 275,543 | 54.2% |
| 554 | Fire rune | 300,517 | 275,634 | 54.2% |
| 7936 | Pure essence | 338,300 | 275,475 | 54.2% |
| 21820 | Revenant ether | 180,507 | 275,708 | 54.2% |
| 565 | Blood rune | 218,933 | 275,716 | 54.2% |
| 555 | Water rune | 201,622 | 275,514 | 54.2% |
| 560 | Death rune | 168,288 | 275,719 | 54.2% |
| 27616 | Ancient essence | 244,590 | 263,599 | 83.8% |
| 562 | Chaos rune | 192,577 | 275,696 | 54.2% |
| 2 | Steel cannonball | 110,487 | 275,682 | 54.2% |
| 314 | Feather | 199,551 | 275,535 | 54.2% |
| 12934 | Zulrah's scales | 199,272 | 275,687 | 54.2% |
| 557 | Earth rune | 134,302 | 275,439 | 54.2% |
| 566 | Soul rune | 107,725 | 275,663 | 54.2% |
| 561 | Nature rune | 114,056 | 275,670 | 54.2% |

### Price Completeness

Most intervals have both high and low prices filled (~95% of rows).
Some intervals have only one side due to:
- No trades at that price level during the interval
- Very low volume items

---

## Item Tiering System

### Tier Definitions

| Tier | Completeness | Avg Volume | Min Days | Price Fill |
|------|--------------|------------|----------|------------|
| 1 | ≥40% | ≥50,000 | ≥365 | ≥70% |
| 2 | ≥30% | ≥10,000 | ≥180 | ≥50% |
| 3 | ≥15% | ≥1,000 | ≥30 | ≥30% |
| 4 | Below thresholds | - | - | No model |

### Tier 1 Items (16 total)

These items are suitable for high-confidence model training:

1. **Air rune (556)** - Most traded rune
2. **Fire rune (554)** - High volume
3. **Water rune (555)** - High volume
4. **Earth rune (557)** - Base rune
5. **Death rune (560)** - Combat essential
6. **Blood rune (565)** - High-level magic
7. **Chaos rune (562)** - Combat rune
8. **Nature rune (561)** - Alchemy rune
9. **Soul rune (566)** - High-level magic
10. **Pure essence (7936)** - Runecrafting material
11. **Revenant ether (21820)** - Wilderness drops
12. **Ancient essence (27616)** - Newer item, 83.8% complete
13. **Steel cannonball (2)** - Combat consumable
14. **Feather (314)** - Fishing/fletching material
15. **Zulrah's scales (12934)** - Boss drops
16. **Coal (453)** - Smithing material

### Tier Distribution

| Tier | Count | Description |
|------|-------|-------------|
| 1 | 16 | High-quality, full model training |
| 2 | ~50 | Medium quality, may train |
| 3 | ~200 | Low quality, limited training |
| 4 | ~4,000 | Insufficient data |

---

## Sample Data Exploration

### Recent Price Sample (Blood rune)

```sql
SELECT timestamp, avg_high_price, avg_low_price,
       high_price_volume, low_price_volume
FROM price_data_5min
WHERE item_id = 565
ORDER BY timestamp DESC
LIMIT 5;
```

| Timestamp | High Price | Low Price | High Vol | Low Vol |
|-----------|------------|-----------|----------|---------|
| 2026-01-07 11:05 | 212 | 205 | 15,432 | 12,891 |
| 2026-01-07 11:00 | 211 | 204 | 18,221 | 14,102 |
| 2026-01-07 10:55 | 213 | 206 | 12,890 | 11,234 |
| 2026-01-07 10:50 | 212 | 205 | 16,543 | 13,678 |
| 2026-01-07 10:45 | 211 | 204 | 14,321 | 12,456 |

**Observations:**
- Spread: ~7-8 gp (~3.4%)
- Volume: 25,000-32,000 trades per 5-min interval
- Prices are stable with small fluctuations

---

## Data Gaps Analysis

### Large Gaps (>1 hour)

Investigated gaps in high-volume items:
- Most gaps occur during OSRS maintenance windows
- Some gaps from API collection issues
- Average gap duration: 2-4 hours
- Maximum gap: ~24 hours (rare)

### Handling Strategy

1. **Training**: Forward-fill NaN prices, skip rows with extended gaps
2. **Inference**: Use most recent available data, flag stale predictions
3. **Features**: Rolling windows handle gaps gracefully (min_periods=1)

---

## Recommendations

1. **Focus on Tier 1 items** for initial deployment
2. **Monitor data freshness** - alert if prices_latest > 1 hour old
3. **Consider extended hours** - OSRS trading patterns vary by timezone
4. **Expand to Tier 2** after validating Tier 1 models
5. **Add data quality monitoring** for production

---

## Appendix: Useful Queries

### Check Recent Data Quality
```sql
SELECT
    MAX(timestamp) as latest,
    COUNT(DISTINCT item_id) as active_items,
    AVG(COALESCE(high_price_volume,0) + COALESCE(low_price_volume,0)) as avg_vol
FROM price_data_5min
WHERE timestamp > NOW() - INTERVAL '1 hour';
```

### Find Items with Best Spreads
```sql
SELECT
    item_id,
    AVG((avg_high_price - avg_low_price)::float /
        ((avg_high_price + avg_low_price)/2) * 100) as avg_spread_pct
FROM price_data_5min
WHERE timestamp > NOW() - INTERVAL '7 days'
  AND avg_high_price > 0 AND avg_low_price > 0
GROUP BY item_id
HAVING COUNT(*) > 1000
ORDER BY avg_spread_pct DESC
LIMIT 20;
```

### Check Data Completeness
```sql
WITH expected AS (
    SELECT item_id, COUNT(*) as actual,
           EXTRACT(EPOCH FROM MAX(timestamp) - MIN(timestamp))/300 as expected
    FROM price_data_5min
    GROUP BY item_id
)
SELECT item_id, actual, expected::int,
       ROUND(actual/GREATEST(expected,1)*100, 1) as completeness_pct
FROM expected
WHERE expected > 10000
ORDER BY completeness_pct DESC
LIMIT 50;
```
