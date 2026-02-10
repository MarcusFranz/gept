import duckdb
import requests
from bs4 import BeautifulSoup
import time
import logging
import sys
import re
from urllib.parse import quote
import os

# Configuration
BASE_DIR = "/home/ubuntu/osrs_collector"
DATA_DIR = os.path.join(BASE_DIR, "data")
ITEMS_DB = os.path.join(DATA_DIR, "items.duckdb")
RECIPES_DB = os.path.join(DATA_DIR, "recipes.duckdb")
USER_AGENT = "GePT-Recipe-Collector/1.0 (Contact: @marcusfranz)"
WIKI_BASE = "https://oldschool.runescape.wiki/w/"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "recipe_collector.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

class RecipeCollector:
    def __init__(self):
        self.item_map = {} # Name -> ID
        self.id_map = {}   # ID -> Name
        self._load_items()
        self._init_db()

    def _load_items(self):
        logging.info("Loading item mapping...")
        con = duckdb.connect(ITEMS_DB, read_only=True)
        # FIX: use 'limit_ge' instead of 'ge_limit'
        rows = con.execute("""
            SELECT id, name FROM item_mapping 
            ORDER BY limit_ge DESC, members DESC
        """).fetchall()
        for r in rows:
            self.id_map[r[0]] = r[1]
            self.item_map[r[1].lower()] = r[0]
        con.close()

    def _init_db(self):
        con = duckdb.connect(RECIPES_DB)
        con.execute("""
            CREATE TABLE IF NOT EXISTS recipes (
                output_id INTEGER,
                input_id INTEGER,
                quantity INTEGER,
                method STRING,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (output_id, input_id)
            )
        """)
        con.close()
    
    def fetch_wiki_page(self, item_name):
        url = f"{WIKI_BASE}{quote(item_name.replace(' ', '_'))}"
        try:
            headers = {"User-Agent": USER_AGENT}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                return resp.content
            return None
        except Exception as e:
            return None

    def parse_ingredients(self, html, output_id):
        soup = BeautifulSoup(html, 'html.parser')
        found_ingredients = []
        
        # Look for tables with 'Materials' or 'Ingredients' in caption
        tables = soup.find_all('table')
        
        for table in tables:
            caption = table.find('caption')
            cap_text = caption.get_text().lower() if caption else ""
            
            if "material" in cap_text or "ingredient" in cap_text or "input" in cap_text:
                rows = table.find_all('tr')
                for row in rows:
                    links = row.find_all('a')
                    for link in links:
                        name = link.get('title')
                        if name and name.lower() in self.item_map:
                            input_id = self.item_map[name.lower()]
                            if input_id != output_id:
                                found_ingredients.append(input_id)
        
        return list(set(found_ingredients))

    def run(self):
        targets = [id for id in self.id_map.keys()]
        logging.info(f"Scanning {len(targets)} items for recipes (Priority: High Volume first)...")
        
        found_count = 0
        con = duckdb.connect(RECIPES_DB)
        
        for output_id in targets:
            name = self.id_map[output_id]
            html = self.fetch_wiki_page(name)
            if html:
                ingredients = self.parse_ingredients(html, output_id)
                if ingredients:
                    found_count += 1
                    logging.info(f"SUCCESS: Found Recipe for {name}: {ingredients}")
                    for input_id in ingredients:
                        con.execute("""
                            INSERT OR IGNORE INTO recipes (output_id, input_id, quantity, method)
                            VALUES (?, ?, 1, 'wiki_scrape')
                        """, (output_id, input_id))
            
            time.sleep(0.3) 
            if found_count > 0 and found_count % 5 == 0:
                logging.info(f"Total Recipes Found: {found_count}")

        con.close()

if __name__ == "__main__":
    RecipeCollector().run()
