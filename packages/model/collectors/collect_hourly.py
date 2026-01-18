import requests
import duckdb
import time
import logging
import sys
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Configuration
DATA_DIR = "/home/ubuntu/osrs_collector/data"
DB_PATH = os.path.join(DATA_DIR, "osrs_hourly.duckdb")
ITEMS_DB = os.path.join(DATA_DIR, "items.duckdb")
STATUS_FILE = os.path.join(DATA_DIR, "hourly_collector_status.json")
HISTORY_API_BASE = "https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep=1h&id={item_id}&begin={begin}"
USER_AGENT = "GePT-Hourly-Collector/1.0 (Contact: @marcusfranz)"
MAX_WORKERS = 15 
START_TIMESTAMP = 1609459200 # Jan 1, 2021 00:00:00

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("/home/ubuntu/osrs_collector/hourly_collector.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def write_status(state, **kwargs):
    """Write status file for dashboard."""
    status = {"state": state, "updated_at": datetime.now().isoformat(), **kwargs}
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)

class HourlyCollector:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        con = duckdb.connect(DB_PATH)
        con.execute("""
            CREATE TABLE IF NOT EXISTS prices_1h (
                item_id INTEGER,
                timestamp TIMESTAMP,
                avg_high_price INTEGER,
                high_price_volume INTEGER,
                avg_low_price INTEGER,
                low_price_volume INTEGER,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (item_id, timestamp)
            )
        """)
        con.close()

    def fetch_chunk(self, item_id, begin_ts):
        headers = {"User-Agent": USER_AGENT}
        url = HISTORY_API_BASE.format(item_id=item_id, begin=int(begin_ts))
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                return resp.json().get("data", [])
            return []
        except:
            return []

    def backfill_item(self, item_id):
        con = duckdb.connect(DB_PATH, read_only=True)
        last_ts = con.execute("SELECT MAX(timestamp) FROM prices_1h WHERE item_id = ?", (item_id,)).fetchone()[0]
        con.close()

        if last_ts:
            if (datetime.now() - last_ts).total_seconds() < 7200:
                return item_id, 0, True  # Already up to date
            next_begin = int(last_ts.timestamp()) + 3600
        else:
            next_begin = START_TIMESTAMP

        total_rows = 0
        now_ts = int(time.time())
        
        while next_begin < now_ts - 3600:
            data = self.fetch_chunk(item_id, next_begin)
            if not data:
                if next_begin < now_ts - (30 * 24 * 3600):
                    next_begin += (30 * 24 * 3600)
                    continue
                else:
                    break

            rows = []
            max_batch_ts = 0
            for e in data:
                ts = e.get("timestamp")
                if not ts: continue
                if ts > max_batch_ts: max_batch_ts = ts
                rows.append((
                    item_id, 
                    datetime.fromtimestamp(ts),
                    e.get("avgHighPrice"),
                    e.get("highPriceVolume", 0),
                    e.get("avgLowPrice"),
                    e.get("lowPriceVolume", 0)
                ))
            
            if rows:
                con_write = duckdb.connect(DB_PATH)
                try:
                    con_write.executemany("INSERT OR IGNORE INTO prices_1h VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)", rows)
                    total_rows += len(rows)
                finally:
                    con_write.close()
            
            if max_batch_ts <= next_begin:
                break
            
            next_begin = max_batch_ts + 3600
            time.sleep(0.1)

        return item_id, total_rows, False

    def run_collection(self):
        start_time = time.time()
        write_status("running", started_at=datetime.now().isoformat())
        logging.info("Starting hourly collection...")
        
        con_items = duckdb.connect(ITEMS_DB, read_only=True)
        ids = [r[0] for r in con_items.execute("SELECT id FROM item_mapping").fetchall()]
        con_items.close()

        logging.info(f"Processing {len(ids)} items.")
        
        total_new_rows = 0
        items_updated = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_id = {executor.submit(self.backfill_item, iid): iid for iid in ids}
            
            done_count = 0
            for future in as_completed(future_to_id):
                iid, rows, was_current = future.result()
                done_count += 1
                if rows > 0:
                    items_updated += 1
                    total_new_rows += rows
                    logging.info(f"[{done_count}/{len(ids)}] Item {iid}: +{rows} rows")

        elapsed = time.time() - start_time
        logging.info(f"Collection complete: {items_updated} items updated, {total_new_rows} new rows in {elapsed:.1f}s")
        return elapsed

    def run_loop(self):
        """Run collection then sleep until next hour boundary."""
        while True:
            elapsed = self.run_collection()
            
            # Calculate sleep until next hour
            now = datetime.now()
            next_hour = (now + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
            sleep_seconds = (next_hour - now).total_seconds()
            
            if sleep_seconds > 0:
                wake_time = next_hour.strftime("%H:%M:%S")
                logging.info(f"Sleeping until {wake_time} ({sleep_seconds:.0f}s)")
                write_status("sleeping", wake_at=next_hour.isoformat(), sleep_seconds=sleep_seconds)
                time.sleep(sleep_seconds)

if __name__ == "__main__":
    collector = HourlyCollector()
    collector.run_loop()
