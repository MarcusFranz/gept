import duckdb
import requests
import time
import logging
import sys
import re
from itertools import combinations
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "meta_clusters.duckdb")
ITEMS_DB = os.path.join(DATA_DIR, "items.duckdb")

USER_AGENT = "GePT-Meta-Collector/1.0 (Contact: @marcusfranz)"
WIKI_API = "https://oldschool.runescape.wiki/api.php"

# The High-Value PvM Content List (Using /Strategies suffix)
STRATEGY_PAGES = [
    "Chambers of Xeric/Strategies",
    "Theatre of Blood/Strategies",
    "Tombs of Amascut/Strategies",
    "Nex/Strategies",
    "Zulrah/Strategies",
    "Vorkath/Strategies",
    "Phantom Muspah/Strategies",
    "The Leviathan/Strategies",
    "The Whisperer/Strategies",
    "Vardorvis/Strategies",
    "Duke Sucellus/Strategies",
    "Alchemical Hydra/Strategies",
    "Corporeal Beast/Strategies",
    "Kree'arra/Strategies",
    "Commander Zilyana/Strategies",
    "General Graardor/Strategies",
    "K'ril Tsutsaroth/Strategies"
]

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "meta_cluster.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

class MetaClusterCollector:
    def __init__(self):
        self.item_map = {} # Name -> ID
        self._load_items()
        self._init_db()

    def _load_items(self):
        """Load items for name matching."""
        logging.info("Loading item map...")
        con = duckdb.connect(ITEMS_DB, read_only=True)
        rows = con.execute("SELECT id, name FROM item_mapping").fetchall()
        for r in rows:
            # Lowercase + remove special chars for loose matching
            clean_name = r[1].lower().strip()
            self.item_map[clean_name] = r[0]
        con.close()

    def _init_db(self):
        con = duckdb.connect(DB_PATH)
        con.execute("""
            CREATE TABLE IF NOT EXISTS cluster_edges (
                source_item INTEGER,
                target_item INTEGER,
                weight INTEGER,
                source_page STRING,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_item, target_item, source_page)
            )
        """)
        con.close()

    def fetch_wikitext(self, page_title):
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "wikitext",
            "format": "json"
        }
        try:
            resp = requests.get(WIKI_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=15)
            data = resp.json()
            return data.get('parse', {}).get('wikitext', {}).get('*', "")
        except Exception as e:
            logging.error(f"Error fetching {page_title}: {e}")
            return ""

    def parse_loadouts(self, wikitext, page_name):
        # Find {{Equipment... }} blocks
        loadouts_found = 0
        con = duckdb.connect(DB_PATH)
        
        # Split text into potential table blocks
        # Wikitext uses {{Equipment|...}}
        blocks = wikitext.split('{{Equipment')
        
        for block in blocks[1:]: # Skip pre-match
            # End the block at the next '}}' (approximate)
            # Beware of nested braces, but Equipment usually doesn't have them except for links [[ ]]
            # A safer split is splitting by `\n}}` or just `}}` if we assume standard formatting.
            block_content = block.split('}}')[0]
            
            items_found = []
            
            lines = block_content.split('\n')
            for line in lines:
                # Regex for |key = value or |key=value
                if '=' in line and line.strip().startswith('|'):
                    parts = line.split('=', 1)
                    if len(parts) < 2: continue
                    
                    value = parts[1].strip()
                    # Remove brackets [[ ]] if present
                    value = value.replace('[[', '').replace(']]', '')
                    # Remove pipe aliases [[Name|Alias]] -> Name
                    if '|' in value:
                        value = value.split('|')[0]
                    
                    clean_val = value.lower().strip()
                    
                    if clean_val in self.item_map:
                        items_found.append(self.item_map[clean_val])

            # Process Clique
            unique_items = list(set(items_found))
            if len(unique_items) < 2: continue
            
            # Create edges between all items in this loadout
            for source, target in combinations(unique_items, 2):
                if source > target: source, target = target, source
                con.execute("""
                    INSERT INTO cluster_edges (source_item, target_item, weight, source_page)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT (source_item, target_item, source_page) 
                    DO UPDATE SET weight = weight + 1
                """, (source, target, page_name))
            
            loadouts_found += 1
                
        con.close()
        return loadouts_found

    def run(self):
        logging.info(f"Starting Meta Cluster Collection (Wikitext Mode)...")
        
        total_loadouts = 0
        for page in STRATEGY_PAGES:
            logging.info(f"Scraping {page}...")
            text = self.fetch_wikitext(page)
            
            if text:
                count = self.parse_loadouts(text, page)
                logging.info(f"  -> Found {count} loadouts.")
                total_loadouts += count
            else:
                logging.warning(f"No text found for {page} (Check title?)")
            
            time.sleep(1) 
            
        logging.info(f"Meta Cluster Collection Complete. Total Loadouts: {total_loadouts}")

if __name__ == "__main__":
    col = MetaClusterCollector()
    col.run()
