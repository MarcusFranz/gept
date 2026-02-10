import requests
import duckdb
import time
import logging
from datetime import datetime
import os
import signal
import sys

# Configuration
DB_PATH = "data/osrs_high_frequency.duckdb"
API_BASE = "https://prices.runescape.wiki/api/v1/osrs"
USER_AGENT = "GePT-Data-Collector - OS: Darwin - Contact: @marcusfranz (Discord)"
FETCH_INTERVAL_SECONDS = 300  # 5 minutes
LATEST_INTERVAL_SECONDS = 60  # 1 minute

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("collector.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class DataCollector:
    def __init__(self):
        self.running = True
        self._init_db()

    def _get_connection(self):
        return duckdb.connect(DB_PATH)

    def _init_db(self):
        """Initialize the database schema."""
        con = self._get_connection()
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS prices_5m (
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
            
            con.execute("""
                CREATE TABLE IF NOT EXISTS prices_latest (
                    item_id INTEGER,
                    high INTEGER,
                    high_time TIMESTAMP,
                    low INTEGER,
                    low_time TIMESTAMP,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logging.info("Database schema initialized.")
        finally:
            con.close()

    def fetch_5m_prices(self):
        """Fetch 5-minute average prices."""
        try:
            url = f"{API_BASE}/5m"
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logging.error(f"Failed to fetch 5m data: {response.status_code}")
                return

            data = response.json().get("data", {})
            timestamp = response.json().get("timestamp") # Unix timestamp of the snapshot
            
            # Prepare batch insert
            rows = []
            
            # Convert Unix timestamp to datetime
            snapshot_dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()

            for item_id, stats in data.items():
                rows.append((
                    int(item_id),
                    snapshot_dt,
                    stats.get("avgHighPrice"),
                    stats.get("highPriceVolume", 0),
                    stats.get("avgLowPrice"),
                    stats.get("lowPriceVolume", 0)
                ))

            if rows:
                con = self._get_connection()
                try:
                    con.executemany("""
                        INSERT OR IGNORE INTO prices_5m 
                        (item_id, timestamp, avg_high_price, high_price_volume, avg_low_price, low_price_volume)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, rows)
                    logging.info(f"Inserted {len(rows)} records into prices_5m.")
                finally:
                    con.close()
            
        except Exception as e:
            logging.error(f"Error in fetch_5m_prices: {e}")

    def fetch_latest_prices(self):
        """Fetch latest real-time prices."""
        try:
            url = f"{API_BASE}/latest"
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logging.error(f"Failed to fetch latest data: {response.status_code}")
                return

            data = response.json().get("data", {})
            
            rows = []
            for item_id, stats in data.items():
                rows.append((
                    int(item_id),
                    stats.get("high"),
                    datetime.fromtimestamp(stats.get("highTime")) if stats.get("highTime") else None,
                    stats.get("low"),
                    datetime.fromtimestamp(stats.get("lowTime")) if stats.get("lowTime") else None
                ))

            if rows:
                con = self._get_connection()
                try:
                    con.executemany("""
                        INSERT INTO prices_latest 
                        (item_id, high, high_time, low, low_time)
                        VALUES (?, ?, ?, ?, ?)
                    """, rows)
                    logging.info(f"Inserted {len(rows)} records into prices_latest.")
                finally:
                    con.close()

        except Exception as e:
            logging.error(f"Error in fetch_latest_prices: {e}")

    def run(self):
        logging.info("Starting Data Collector...")
        
        last_5m_fetch = 0
        last_latest_fetch = 0
        
        while self.running:
            now = time.time()
            
            # Fetch 5m data
            if now - last_5m_fetch >= FETCH_INTERVAL_SECONDS:
                logging.info("Fetching 5m data...")
                self.fetch_5m_prices()
                last_5m_fetch = time.time()
            
            # Fetch latest data
            if now - last_latest_fetch >= LATEST_INTERVAL_SECONDS:
                logging.info("Fetching latest data...")
                self.fetch_latest_prices()
                last_latest_fetch = time.time()
            
            time.sleep(1) # Prevent CPU spin

    def shutdown(self, signum, frame):
        logging.info("Shutdown signal received.")
        self.running = False
        sys.exit(0)

if __name__ == "__main__":
    collector = DataCollector()
    signal.signal(signal.SIGINT, collector.shutdown)
    signal.signal(signal.SIGTERM, collector.shutdown)
    collector.run()
