import duckdb
import pandas as pd
import numpy as np
import logging
import sys
import os
import signal
import time
from datetime import datetime, timedelta

# Configuration
RAW_DB_PATH = "data/osrs_high_frequency.duckdb"
PROCESSED_DB_PATH = "data/processed_tensors.duckdb"
ITEM_PROPS_PATH = "data/item_properties.parquet"
ENCODER_LEN = 168 # 7 days of hours
DECODER_LEN = 24  # 1 day of hours
WINDOW_SIZE = ENCODER_LEN + DECODER_LEN

# Feature Columns (Must match training code)
STATIC_CATS = ['equipment_slot', 'combat_style', 'content_type', 'content_source', 'price_tier', 'volume_tier', 'buy_limit_tier']
STATIC_CONTS = ['ge_limit', 'high_alch', 'is_members', 'is_stackable', 'is_equipable', 'is_consumable', 'is_raid_drop', 'is_boss_drop', 'is_skilling_supply']

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("transformer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class DataTransformer:
    def __init__(self):
        self.running = True
        self.con_raw = None
        self.con_proc = None
        
        # Load item properties once
        if os.path.exists(ITEM_PROPS_PATH):
            self.item_props = pd.read_parquet(ITEM_PROPS_PATH)
        else:
            logging.warning("Item properties not found. Tensors will lack static features until populated.")
            self.item_props = pd.DataFrame()

    def _init_db(self):
        """Initialize the processed database schema."""
        self.con_proc = duckdb.connect(PROCESSED_DB_PATH)
        self.con_proc.execute("""
            CREATE TABLE IF NOT EXISTS tensors_1h (
                item_id INTEGER,
                window_start TIMESTAMP,
                window_end TIMESTAMP,
                encoder_cat BLOB,       -- Serialized Numpy/Torch tensor
                encoder_cont BLOB,      -- Serialized Numpy/Torch tensor
                decoder_cat BLOB,       -- Serialized Numpy/Torch tensor
                decoder_cont BLOB,      -- Serialized Numpy/Torch tensor
                static_cat BLOB,        -- Serialized Numpy/Torch tensor
                static_cont BLOB,       -- Serialized Numpy/Torch tensor
                targets BLOB,           -- Serialized Numpy/Torch tensor
                base_prices BLOB,       -- Serialized Numpy/Torch tensor
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (item_id, window_end)
            )
        """)
        logging.info("Processed DB schema initialized.")

    def compute_features(self, df):
        """Compute technical indicators (RSI, MA, Volatility) matching dataset.py logic."""
        # This function replicates the logic inside dataset.py
        # For brevity, I'll implement the core vectorizations
        
        high = df['avg_high_price'].values.astype(np.float32)
        low = df['avg_low_price'].values.astype(np.float32)
        
        # Log transforms
        log_high = np.log1p(np.maximum(high, 1))
        log_low = np.log1p(np.maximum(low, 1))
        
        # Returns
        returns_1h = np.zeros_like(high)
        returns_1h[1:] = np.clip((high[1:] - high[:-1]) / (high[:-1] + 1e-8), -0.3, 0.3)
        
        # Volatility (24h)
        vol_24h = np.zeros_like(high)
        for i in range(24, len(high)):
            vol_24h[i] = np.std(returns_1h[max(0, i-24):i])
            
        # Moving Averages
        ma_24h = np.zeros_like(high)
        for i in range(24, len(high)):
            ma_24h[i] = high[max(0, i-24):i].mean()
            
        # Price vs MA
        price_vs_ma = np.zeros_like(high)
        price_vs_ma[24:] = np.clip((high[24:] - ma_24h[24:]) / (ma_24h[24:] + 1e-8), -0.3, 0.3)

        return np.stack([log_high, log_low, returns_1h, vol_24h, price_vs_ma], axis=1)

    def process_item(self, item_id, df):
        """Process a single item's dataframe into tensors."""
        if len(df) < WINDOW_SIZE:
            return None

        # Take the latest window
        window = df.iloc[-WINDOW_SIZE:].copy()
        
        # Compute dynamic features
        features = self.compute_features(window)
        
        # Split Encoder/Decoder
        enc_feats = features[:ENCODER_LEN]
        dec_feats = features[ENCODER_LEN:] # Decoder typically only knows 'known' inputs, this needs alignment with your model
        
        # Targets (Log Returns)
        base_high = window['avg_high_price'].iloc[ENCODER_LEN-1]
        future_highs = window['avg_high_price'].iloc[ENCODER_LEN:].values
        targets = np.log(np.maximum(future_highs, 1) / np.maximum(base_high, 1))

        # Serialization (Simple Bytes for now, could be Pickle or Arrow)
        return {
            'encoder_cont': enc_feats.tobytes(),
            'targets': targets.tobytes(),
            'window_end': window['timestamp'].iloc[-1]
        }

    def run_batch(self):
        """Run a batch processing job with retry logic."""
        logging.info("Starting batch transformation...")
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.con_raw = duckdb.connect(RAW_DB_PATH, read_only=True)
                self._init_db()
                
                # Get active items
                item_ids = [r[0] for r in self.con_raw.execute("SELECT DISTINCT item_id FROM prices_5m").fetchall()]
                logging.info(f"Found {len(item_ids)} items to process.")
                
                processed_count = 0
                
                for item_id in item_ids:
                    # Fetch last 200 hours (buffer for window)
                    query = f"""
                        SELECT timestamp, avg_high_price, avg_low_price 
                        FROM prices_5m 
                        WHERE item_id = {item_id} 
                        ORDER BY timestamp DESC 
                        LIMIT {WINDOW_SIZE + 50}
                    """
                    df = self.con_raw.execute(query).fetchdf().sort_values('timestamp')
                    
                    result = self.process_item(item_id, df)
                    
                    if result:
                        self.con_proc.execute("""
                            INSERT OR REPLACE INTO tensors_1h (item_id, window_end, encoder_cont, targets)
                            VALUES (?, ?, ?, ?)
                        """, (item_id, result['window_end'], result['encoder_cont'], result['targets']))
                        processed_count += 1
                        
                logging.info(f"Batch complete. Processed {processed_count} tensors.")
                self.con_raw.close()
                self.con_proc.close()
                break # Success
                
            except Exception as e:
                logging.warning(f"Batch attempt {attempt+1} failed: {e}")
                if self.con_raw:
                    try:
                        self.con_raw.close()
                    except Exception as close_err:
                        logging.debug(f"Error closing raw connection: {close_err}")
                if self.con_proc:
                    try:
                        self.con_proc.close()
                    except Exception as close_err:
                        logging.debug(f"Error closing processed connection: {close_err}")

                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1)) # Backoff: 5, 10, 15, 20s
                else:
                    logging.error("All batch retries failed.")

    def run(self):
        logging.info("Service started.")
        while self.running:
            try:
                self.run_batch()
            except Exception as e:
                logging.error(f"Batch failed: {e}")
            
            # Sleep 1 hour
            time.sleep(3600)

if __name__ == "__main__":
    transformer = DataTransformer()
    transformer.run()
