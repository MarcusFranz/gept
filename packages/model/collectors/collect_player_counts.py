import requests
import duckdb
import time
import logging
import re
import signal
import sys
from datetime import datetime

# Configuration
DB_PATH = "data/player_counts.duckdb"
URL = "https://oldschool.runescape.com/"
USER_AGENT = "GePT-Data-Collector - OS: Darwin - Contact: @marcusfranz (Discord)"
FETCH_INTERVAL_SECONDS = 60

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("player_count_collector.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class PlayerCountCollector:
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
                CREATE TABLE IF NOT EXISTS player_counts (
                    timestamp TIMESTAMP,
                    count INTEGER,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logging.info("Database schema initialized.")
        finally:
            con.close()

    def fetch_player_count(self):
        """Fetch OSRS player count from the homepage."""
        try:
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(URL, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logging.error(f"Failed to fetch homepage: {response.status_code}")
                return

            # Regex to find "There are currently X people playing!"
            # The site usually has <p class='player-count'>There are currently 123,456 people playing!</p>
            # We look for the number pattern specifically.
            content = response.text
            match = re.search(r"There are currently ([\d,]+) people playing!", content)
            
            if match:
                count_str = match.group(1).replace(",", "")
                count = int(count_str)
                
                con = self._get_connection()
                try:
                    con.execute("""
                        INSERT INTO player_counts (timestamp, count)
                        VALUES (CURRENT_TIMESTAMP, ?)
                    """, [count])
                    logging.info(f"Logged player count: {count}")
                finally:
                    con.close()
            else:
                logging.warning("Could not find player count pattern in HTML.")

        except Exception as e:
            logging.error(f"Error in fetch_player_count: {e}")

    def run(self):
        logging.info("Starting Player Count Collector...")
        
        while self.running:
            self.fetch_player_count()
            
            # Sleep for the remainder of the interval
            time.sleep(FETCH_INTERVAL_SECONDS)

    def shutdown(self, signum, frame):
        logging.info("Shutdown signal received.")
        self.running = False
        sys.exit(0)

if __name__ == "__main__":
    collector = PlayerCountCollector()
    signal.signal(signal.SIGINT, collector.shutdown)
    signal.signal(signal.SIGTERM, collector.shutdown)
    collector.run()
