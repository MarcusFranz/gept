import requests
import duckdb
import time
import logging
import sys
import re
import os
import json

# Configuration
DATA_DIR = "/home/ubuntu/osrs_collector/data"
ITEMS_DB = os.path.join(DATA_DIR, "items.duckdb")
WIKI_API = "https://oldschool.runescape.wiki/api.php"
USER_AGENT = "GePT-Item-Enricher/1.0 (Contact: @marcusfranz)"
BATCH_SIZE = 50

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/ubuntu/osrs_collector/item_enricher.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

class ItemEnricher:
    def __init__(self):
        self._init_db()
        self.item_map = self._load_items()

    def _init_db(self):
        con = duckdb.connect(ITEMS_DB)
        con.execute("""
            CREATE TABLE IF NOT EXISTS item_enriched (
                id INTEGER PRIMARY KEY,
                name VARCHAR,
                equipment_slot VARCHAR,
                combat_style VARCHAR,
                content_type VARCHAR,
                content_source VARCHAR,
                is_stackable BOOLEAN,
                is_equipable BOOLEAN,
                is_consumable BOOLEAN,
                is_raid_drop BOOLEAN,
                is_boss_drop BOOLEAN,
                is_skilling_supply BOOLEAN,
                last_enriched TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.close()

    def _load_items(self):
        con = duckdb.connect(ITEMS_DB, read_only=True)
        rows = con.execute("SELECT id, name FROM item_mapping").fetchall()
        con.close()
        return {r[1]: r[0] for r in rows}

    def fetch_wiki_data(self, titles):
        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "titles": "|".join(titles),
            "format": "json",
            "redirects": 1
        }
        headers = {"User-Agent": USER_AGENT}
        try:
            resp = requests.get(WIKI_API, params=params, headers=headers, timeout=20)
            return resp.json()
        except Exception as e:
            logging.error(f"Wiki API error: {e}")
            return {}

    def parse_item_data(self, title, wikitext):
        data = {
            "name": title,
            "equipment_slot": "none",
            "combat_style": "none",
            "content_type": "general",
            "content_source": "general",
            "is_stackable": False,
            "is_equipable": False,
            "is_consumable": False,
            "is_raid_drop": False,
            "is_boss_drop": False,
            "is_skilling_supply": False
        }

        if not wikitext:
            return data

        wt = wikitext.lower()

        # 1. Equipment Slot & Style
        # Match |slot = Head or |slot=Weapon
        slot_match = re.search(r"\|slot\s*=\s*([^|\n]+)", wt)
        if slot_match:
            slot_val = slot_match.group(1).strip().lower()
            if slot_val:
                data["equipment_slot"] = slot_val
                data["is_equipable"] = True

        if "weapon" in data["equipment_slot"] or "2h" in data["equipment_slot"]:
            if any(x in wt for x in ["magic", "staff", "wand"]): data["combat_style"] = "magic"
            elif any(x in wt for x in ["ranged", "bow", "crossbow", "dart"]): data["combat_style"] = "ranged"
            else: data["combat_style"] = "melee"

        # 2. Content Type & Skilling
        if any(x in wt for x in ["{{potion", "food=yes", "drink=yes"]):
            data["is_consumable"] = True
            data["content_type"] = "consumable"
        
        skilling_keywords = ["herb", "ore", "log", "seed", "raw", "unf", "grimy", "bar"]
        if any(x in wt for x in skilling_keywords):
            data["is_skilling_supply"] = True
            data["content_type"] = "skilling"

        # 3. Drops & Sources
        raid_keywords = ["chambers of xeric", "theatre of blood", "tombs of amascut"]
        if any(x in wt for x in raid_keywords):
            data["is_raid_drop"] = True
            data["content_type"] = "pvm"
            if "chambers" in wt: data["content_source"] = "cox"
            elif "theatre" in wt: data["content_source"] = "tob"
            elif "tombs" in wt: data["content_source"] = "toa"

        if "dropped by" in wt or "reward from" in wt:
            data["is_boss_drop"] = True
            data["content_type"] = "pvm"
            if any(x in wt for x in ["god wars", "nex"]): data["content_source"] = "gwd"
            elif "wilderness" in wt: data["content_source"] = "wildy"
            elif "slayer" in wt: data["content_source"] = "slayer"
            elif any(x in wt for x in ["desert treasure ii", "vardorvis", "duke sucellus"]): data["content_source"] = "dt2"

        # 4. Stackable
        if "|stackable=yes" in wt or "|stackable = yes" in wt:
            data["is_stackable"] = True

        return data

    def enrich_all(self):
        titles = list(self.item_map.keys())
        total = len(titles)
        logging.info(f"Starting enrichment for {total} items...")

        con = duckdb.connect(ITEMS_DB)
        
        for i in range(0, total, BATCH_SIZE):
            batch_titles = titles[i:i + BATCH_SIZE]
            wiki_resp = self.fetch_wiki_data(batch_titles)
            
            pages = wiki_resp.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                title = page_data.get("title")
                if "revisions" not in page_data: continue
                
                wikitext = page_data["revisions"][0]["*"]
                
                if title in self.item_map:
                    item_id = self.item_map[title]
                    enriched = self.parse_item_data(title, wikitext)
                    
                    con.execute("""
                        INSERT OR REPLACE INTO item_enriched 
                        (id, name, equipment_slot, combat_style, content_type, content_source, 
                         is_stackable, is_equipable, is_consumable, is_raid_drop, is_boss_drop, is_skilling_supply)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (item_id, title, enriched["equipment_slot"], enriched["combat_style"], 
                          enriched["content_type"], enriched["content_source"], enriched["is_stackable"], 
                          enriched["is_equipable"], enriched["is_consumable"], enriched["is_raid_drop"], 
                          enriched["is_boss_drop"], enriched["is_skilling_supply"]))

            logging.info(f"Progress: {min(i + BATCH_SIZE, total)}/{total}")
            time.sleep(1)

        con.close()
        logging.info("Enrichment complete.")

if __name__ == "__main__":
    enricher = ItemEnricher()
    enricher.enrich_all()