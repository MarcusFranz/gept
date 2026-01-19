#!/usr/bin/env python3
"""Create database tables for ML ranker training data."""

import psycopg2
import os

def main():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 5432)),
        dbname=os.getenv('DB_NAME', 'osrs_data'),
        user=os.getenv('DB_USER', 'osrs_user'),
        password=os.getenv('DB_PASS', 'osrs_price_data_2024'),
    )
    cur = conn.cursor()

    # Create training_trades table for Flipping Copilot imports
    cur.execute('''
    CREATE TABLE IF NOT EXISTS training_trades (
        id SERIAL PRIMARY KEY,

        -- Trade identification
        first_buy_time TIMESTAMPTZ NOT NULL,
        last_sell_time TIMESTAMPTZ,
        account TEXT NOT NULL,

        -- Item reference
        item_name TEXT NOT NULL,
        item_id INTEGER,  -- Populated after name mapping

        -- Trade status
        status TEXT NOT NULL,  -- FINISHED, BUYING, SELLING

        -- Quantities
        bought INTEGER NOT NULL,
        sold INTEGER NOT NULL,

        -- Prices
        avg_buy_price DECIMAL(12,2) NOT NULL,
        avg_sell_price DECIMAL(12,2),

        -- Outcome
        tax DECIMAL(14,2),
        profit DECIMAL(14,2),
        profit_each DECIMAL(10,2),

        -- Metadata
        has_price_data BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMPTZ DEFAULT NOW(),

        -- Constraints
        CHECK (status IN ('FINISHED', 'BUYING', 'SELLING'))
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_training_trades_item_id ON training_trades(item_id);
    CREATE INDEX IF NOT EXISTS idx_training_trades_first_buy_time ON training_trades(first_buy_time);
    CREATE INDEX IF NOT EXISTS idx_training_trades_status ON training_trades(status);
    CREATE INDEX IF NOT EXISTS idx_training_trades_has_price_data ON training_trades(has_price_data);

    COMMENT ON TABLE training_trades IS 'Imported trades from Flipping Copilot for ML ranker training';
    ''')

    conn.commit()
    print("Created training_trades table successfully")

    # Create historical_predictions table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS historical_predictions (
        id SERIAL PRIMARY KEY,

        -- Link to training trade
        training_trade_id INTEGER REFERENCES training_trades(id),

        -- Item reference
        item_id INTEGER NOT NULL,

        -- Prediction timestamp (when trade started)
        prediction_time TIMESTAMPTZ NOT NULL,

        -- Model reference
        model_month TEXT NOT NULL,  -- e.g., '2025-08' for Aug 2025 model

        -- Prediction outputs (multiple hour_offset/offset_pct combinations)
        hour_offset INTEGER NOT NULL,
        offset_pct DECIMAL(5,4) NOT NULL,
        fill_probability DECIMAL(7,6),
        expected_value DECIMAL(10,6),
        buy_price DECIMAL(12,2),
        sell_price DECIMAL(12,2),
        confidence TEXT,

        -- Metadata
        created_at TIMESTAMPTZ DEFAULT NOW(),

        UNIQUE (training_trade_id, hour_offset, offset_pct)
    );

    -- Indexes
    CREATE INDEX IF NOT EXISTS idx_historical_predictions_trade_id ON historical_predictions(training_trade_id);
    CREATE INDEX IF NOT EXISTS idx_historical_predictions_item_id ON historical_predictions(item_id);
    CREATE INDEX IF NOT EXISTS idx_historical_predictions_model_month ON historical_predictions(model_month);

    COMMENT ON TABLE historical_predictions IS 'Recreated predictions for historical trades using monthly model snapshots';
    ''')

    conn.commit()
    print("Created historical_predictions table successfully")
    conn.close()


if __name__ == '__main__':
    main()
