#!/usr/bin/env python3
"""OSRS Price Collector - PostgreSQL/TimescaleDB Version"""
import os
import requests
import psycopg2
from psycopg2.extras import execute_values
import time
import logging
import sys
from datetime import datetime, timezone

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'osrs_data'),
    'user': os.getenv('DB_USER', 'osrs_user'),
    'password': os.environ['DB_PASS']
}
API_BASE = 'https://prices.runescape.wiki/api/v1/osrs'
USER_AGENT = 'GePT-Collector/2.0 (PostgreSQL)'

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

class PGCollector:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self._ensure_tables()
        self.last_5m_ts = None
        log.info('Connected to PostgreSQL')
    
    def _ensure_tables(self):
        cur = self.conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS prices_latest (
            item_id INTEGER NOT NULL, timestamp TIMESTAMPTZ NOT NULL,
            high_price INTEGER, high_time TIMESTAMPTZ,
            low_price INTEGER, low_time TIMESTAMPTZ)''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_latest_ts ON prices_latest(timestamp DESC)')
        self.conn.commit()
        cur.close()
    
    def fetch_5m(self):
        try:
            r = requests.get(f'{API_BASE}/5m', headers={'User-Agent': USER_AGENT}, timeout=30)
            r.raise_for_status()
            data = r.json().get('data', {})
            ts = datetime.fromtimestamp(r.json().get('timestamp', time.time()), tz=timezone.utc)
            if ts == self.last_5m_ts:
                log.info(f'5m: skipping duplicate timestamp {ts}')
                return 0
            self.last_5m_ts = ts
            rows = [(int(item_id), ts, p.get('avgHighPrice'), p.get('highPriceVolume'),
                     p.get('avgLowPrice'), p.get('lowPriceVolume')) for item_id, p in data.items()]
            if rows:
                cur = self.conn.cursor()
                execute_values(cur, '''INSERT INTO price_data_5min 
                    (item_id, timestamp, avg_high_price, high_price_volume, avg_low_price, low_price_volume)
                    VALUES %s''', rows)
                self.conn.commit()
                cur.close()
                log.info(f'5m: inserted {len(rows)} rows for {ts}')
                return len(rows)
        except Exception as e:
            log.error(f'5m error: {e}')
            self.conn.rollback()
        return 0
    
    def fetch_latest(self):
        try:
            r = requests.get(f'{API_BASE}/latest', headers={'User-Agent': USER_AGENT}, timeout=30)
            r.raise_for_status()
            data = r.json().get('data', {})
            ts = datetime.now(timezone.utc)
            rows = []
            for item_id, p in data.items():
                ht = datetime.fromtimestamp(p['highTime'], tz=timezone.utc) if p.get('highTime') else None
                lt = datetime.fromtimestamp(p['lowTime'], tz=timezone.utc) if p.get('lowTime') else None
                rows.append((int(item_id), ts, p.get('high'), ht, p.get('low'), lt))
            if rows:
                cur = self.conn.cursor()
                execute_values(cur, '''INSERT INTO prices_latest 
                    (item_id, timestamp, high_price, high_time, low_price, low_time) VALUES %s''', rows)
                self.conn.commit()
                cur.close()
                log.info(f'Latest: inserted {len(rows)} rows')
                return len(rows)
        except Exception as e:
            log.error(f'Latest error: {e}')
            self.conn.rollback()
        return 0
    
    def run(self):
        log.info('Starting collector - 5m every 300s, latest every 60s')
        last_5m = 0
        last_latest = 0
        while True:
            now = time.time()
            if now - last_5m >= 300:
                self.fetch_5m()
                last_5m = now
            if now - last_latest >= 60:
                self.fetch_latest()
                last_latest = now
            time.sleep(10)

if __name__ == '__main__':
    PGCollector().run()
