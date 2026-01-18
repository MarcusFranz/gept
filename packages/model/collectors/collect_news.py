import requests
import duckdb
import time
import logging
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
import signal
import email.utils
import argparse
from sentence_transformers import SentenceTransformer
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "news.duckdb")

RSS_URL = "https://secure.runescape.com/m=news/latest_news.rss?oldschool=true"
WIKI_API = "https://oldschool.runescape.wiki/api.php"
USER_AGENT = "GePT-News-Backfiller/1.0 (Contact: @marcusfranz)"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FETCH_INTERVAL_SECONDS = 3600

# Ensure data dir exists
os.makedirs(DATA_DIR, exist_ok=True)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(os.path.join(BASE_DIR, "news_collector.log")), logging.StreamHandler(sys.stdout)]
)

class NewsCollector:
    def __init__(self):
        self.running = True
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu')
        self._init_db()

    def _init_db(self):
        con = duckdb.connect(DB_PATH)
        con.execute("""
            CREATE TABLE IF NOT EXISTS news (
                guid STRING PRIMARY KEY,
                title STRING,
                link STRING,
                description STRING,
                category STRING,
                pub_date TIMESTAMP,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                category_ai STRING, 
                confidence FLOAT,   
                embedding FLOAT[]
            )
        """)
        # Migration
        try:
            con.execute("ALTER TABLE news ADD COLUMN embedding FLOAT[]")
        except:
            pass
        con.close()

    def compute_embedding(self, text):
        return self.model.encode(text).tolist()

    def upsert_news(self, guid, title, link, description, category, pub_date):
        con = duckdb.connect(DB_PATH)
        try:
            # Check exist
            existing = con.execute("SELECT guid FROM news WHERE guid = ?", (guid,)).fetchone()
            if existing: return False
            
            emb = self.compute_embedding(f"{title}. {description}")
            con.execute("""
                INSERT INTO news (guid, title, link, description, category, pub_date, embedding) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (guid, title, link, description, category, pub_date, emb))
            return True
        except Exception as e:
            logging.error(f"DB Error: {e}")
            return False
        finally:
            con.close()

    def backfill_wiki(self, start_year=2015):
        logging.info(f"Starting Wiki Backfill from {start_year}...")
        current_year = datetime.now().year
        
        for year in range(current_year, start_year - 1, -1):
            logging.info(f"Checking Year: {year}")
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{year}_updates",
                "cmlimit": 500,
                "format": "json"
            }
            try:
                resp = requests.get(WIKI_API, params=params, headers={"User-Agent": USER_AGENT}).json()
                members = resp.get("query", {}).get("categorymembers", [])
                
                if not members:
                    logging.warning(f"No updates found for {year} (Check Category Name?)")
                    continue

                for member in members:
                    title = member['title']
                    # Fetch page content
                    p_params = {
                        "action": "query",
                        "prop": "extracts|revisions",
                        "exintro": True,
                        "explaintext": True,
                        "titles": title,
                        "rvprop": "timestamp",
                        "format": "json"
                    }
                    p_resp = requests.get(WIKI_API, params=p_params, headers={"User-Agent": USER_AGENT}).json()
                    pages = p_resp.get('query', {}).get('pages', {})
                    if not pages: continue
                    
                    page = next(iter(pages.values()))
                    
                    desc = page.get("extract", "")
                    date_str = page.get("revisions", [{}])[0].get("timestamp", "")
                    
                    # Parse Date
                    if date_str:
                        pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        pub_date = datetime.now()
                    
                    link = f"https://oldschool.runescape.wiki/w/{title.replace(' ', '_')}"
                    
                    if self.upsert_news(title, title, link, desc, "Wiki Update", pub_date):
                        logging.info(f"Backfilled Wiki: {title}")
                    
                    time.sleep(0.2)
            except Exception as e:
                logging.error(f"Error in backfill year {year}: {e}")

    def fetch_rss(self):
        """Live RSS Fetcher"""
        try:
            resp = requests.get(RSS_URL, headers={"User-Agent": USER_AGENT}, timeout=15)
            if resp.status_code != 200: return
            root = ET.fromstring(resp.content)
            channel = root.find("channel")
            if not channel: return
            
            for item in channel.findall("item"):
                title = item.find("title").text
                link = item.find("link").text
                desc = item.find("description").text
                guid = item.find("guid").text
                
                pub_date = datetime.now() # Simplify for brevity/resilience
                
                if self.upsert_news(guid, title, link, desc, "RSS", pub_date):
                    logging.info(f"RSS New Item: {title}")
                    
        except Exception as e:
            logging.error(f"RSS Error: {e}")

    def run(self, backfill=False):
        if backfill: 
            self.backfill_wiki()
            return

        logging.info("Starting News Service...")
        while self.running:
            self.fetch_rss()
            time.sleep(FETCH_INTERVAL_SECONDS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()
    NewsCollector().run(backfill=args.backfill)
