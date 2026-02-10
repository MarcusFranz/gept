import duckdb
import time
import logging
import sys
import os
from datetime import datetime

# Configuration
DATA_DIR = "/home/ubuntu/osrs_collector/data"
HOURLY_DB = os.path.join(DATA_DIR, "osrs_hourly.duckdb")
VOLUMES_DB = os.path.join(DATA_DIR, "daily_volumes.duckdb")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/ubuntu/osrs_collector/daily_volume.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def aggregate_volumes():
    logging.info("Starting volume aggregation from Hourly Prices...")
    
    # We need to connect to both DBs
    # Strategy: Attach VOLUMES_DB to the HOURLY_DB connection
    try:
        con = duckdb.connect(HOURLY_DB, read_only=True)
        con.execute(f"ATTACH '{VOLUMES_DB}' AS db_vol")
        
        # Aggregation Logic:
        # Sum (avg_high_price_vol + avg_low_price_vol) grouped by day
        # We use time_bucket or just cast to DATE
        
        con.execute("""
            INSERT OR REPLACE INTO db_vol.daily_volumes (item_id, timestamp, volume)
            SELECT 
                item_id,
                timestamp::DATE as ts,
                SUM(COALESCE(avg_high_price_vol, 0) + COALESCE(avg_low_price_vol, 0)) as vol
            FROM prices_1h
            GROUP BY 1, 2
        """)
        
        row_count = con.execute("SELECT COUNT(*) FROM db_vol.daily_volumes").fetchone()[0]
        logging.info(f"Aggregation complete. Daily Volumes now contains {row_count:, } rows.")
        
        con.close()
    except Exception as e:
        logging.error(f"Aggregation failed: {e}")

if __name__ == "__main__":
    while True:
        aggregate_volumes()
        # Sleep for 1 hour to stay synced with hourly backfill
        logging.info("Sleeping for 1 hour...")
        time.sleep(3600)