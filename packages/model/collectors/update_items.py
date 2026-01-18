import requests
import duckdb
import time
import logging
import sys
import signal
from datetime import datetime

# Configuration
DB_PATH = "data/items.duckdb"
MAPPING_URL = "https://prices.runescape.wiki/api/v1/osrs/mapping"
USER_AGENT = "GePT-Data-Collector - OS: Darwin - Contact: @marcusfranz (Discord)"
FETCH_INTERVAL_SECONDS = 86400  # 24 hours

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("item_updater.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class ItemUpdater:
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
                CREATE TABLE IF NOT EXISTS item_mapping (
                    id INTEGER PRIMARY KEY,
                    name STRING,
                    members BOOLEAN,
                    limit_ge INTEGER,
                    value INTEGER,
                    highalch INTEGER,
                    lowalch INTEGER,
                    examine STRING,
                    icon STRING,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logging.info("Database schema initialized.")
        finally:
            con.close()

    def fetch_mapping(self):
        """Fetch OSRS item mapping."""
        try:
            logging.info("Fetching item mapping...")
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(MAPPING_URL, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logging.error(f"Failed to fetch mapping: {response.status_code}")
                return

            data = response.json()
            
            # Prepare batch upsert
            con = self._get_connection()
            try:
                count = 0
                for item in data:
                    # Upsert logic: INSERT OR REPLACE
                    # DuckDB supports INSERT OR REPLACE INTO or ON CONFLICT
                    try:
                        con.execute("""
                            INSERT OR REPLACE INTO item_mapping 
                            (id, name, members, limit_ge, value, highalch, lowalch, examine, icon, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, (
                            item.get("id"),
                            item.get("name"),
                            item.get("members"),
                            item.get("limit"),
                            item.get("value"),
                            item.get("highalch"),
                            item.get("lowalch"),
                            item.get("examine"),
                            item.get("icon")
                        ))
                        count += 1
                    except Exception as e:
                        logging.error(f"Error inserting item {item.get('id')}: {e}")

                logging.info(f"Updated mapping for {count} items.")
            finally:
                con.close()

        except Exception as e:
            logging.error(f"Error in fetch_mapping: {e}")

    def run(self):
        logging.info("Starting Item Updater...")
        
        while self.running:
            self.fetch_mapping()
            # Sleep for interval
            time.sleep(FETCH_INTERVAL_SECONDS)

    def shutdown(self, signum, frame):
        logging.info("Shutdown signal received.")
        self.running = False
        sys.exit(0)

if __name__ == "__main__":
    updater = ItemUpdater()
    signal.signal(signal.SIGINT, updater.shutdown)
    signal.signal(signal.SIGTERM, updater.shutdown)
    updater.run()
