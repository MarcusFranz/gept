import requests
import duckdb
import time
import logging
import sys
import pandas as pd
from datetime import datetime

# Configuration
DB_PATH = "data/osrs_high_frequency.duckdb"
USER_AGENT = "GePT-Data-Collector - OS: Darwin - Contact: @marcusfranz (Discord)"
API_BASE = "https://api.weirdgloop.org/exchange/history/osrs/all"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("backfill.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class HistoryBackfiller:
    def __init__(self):
        self.con = duckdb.connect(DB_PATH)

    def get_tracked_items(self):
        """Get list of item IDs we are currently tracking."""
        # We'll use the items found in the prices_latest table as our 'active' list
        # or fall back to a known top list if empty.
        try:
            items = [r[0] for r in self.con.execute("SELECT DISTINCT item_id FROM prices_latest").fetchall()]
            if not items:
                logging.warning("No items found in prices_latest. Defaulting to top 100 placeholder.")
                # Fallback: Just Abyssal Whip (4151) and a few others to test
                items = [4151, 5698, 2, 444, 560] 
            return items
        except:
            return [4151] # Fallback

    def backfill_item(self, item_id):
        """Fetch full 5m history for an item."""
        try:
            # The Wiki API supports requesting specific IDs.
            # Using the 'all' endpoint with 'id' parameter is best.
            url = f"{API_BASE}?id={item_id}" 
            # Note: weirdgloop API usually returns daily by default or full history. 
            # For 5m data, we specifically need the /5m endpoint which is different.
            # Real-time API has /5m (last 1h), but historical 5m is via 'timeseries' endpoint on weirdgloop?
            # Actually, standard weirdgloop 'history' is usually daily. 
            # The 5m history is accessible via: https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=5m&id=4151
            
            url = f"https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=5m&id={item_id}"
            headers = {"User-Agent": USER_AGENT}
            
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code != 200:
                logging.error(f"Failed to fetch history for {item_id}: {response.status_code}")
                return

            data = response.json().get("data", [])
            if not data:
                return

            # Data format: [{"timestamp": 12345, "avgHighPrice": 100, ...}, ...]
            rows = []
            for entry in data:
                ts = datetime.fromtimestamp(entry.get("timestamp"))
                rows.append((
                    item_id,
                    ts,
                    entry.get("avgHighPrice"),
                    entry.get("highPriceVolume", 0),
                    entry.get("avgLowPrice"),
                    entry.get("lowPriceVolume", 0)
                ))

            if rows:
                self.con.executemany("""
                    INSERT OR IGNORE INTO prices_5m 
                    (item_id, timestamp, avg_high_price, high_price_volume, avg_low_price, low_price_volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, rows)
                logging.info(f"Backfilled {len(rows)} rows for item {item_id}")

        except Exception as e:
            logging.error(f"Error backfilling {item_id}: {e}")

    def run(self):
        logging.info("Starting Historical Backfill...")
        items = self.get_tracked_items()
        logging.info(f"Targeting {len(items)} items.")
        
        # Batch processing to avoid rate limits
        for i, item_id in enumerate(items):
            self.backfill_item(item_id)
            time.sleep(0.5) # Gentle rate limit
            
            if i % 100 == 0:
                logging.info(f"Progress: {i}/{len(items)}")

        logging.info("Backfill Complete.")
        self.con.close()

if __name__ == "__main__":
    bf = HistoryBackfiller()
    bf.run()
