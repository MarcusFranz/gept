import requests, duckdb, time, os
from datetime import datetime
DB = 'data/daily_prices.duckdb'
ITEMS = 'data/items.duckdb'
con = duckdb.connect(DB)
con.execute('CREATE TABLE IF NOT EXISTS daily_prices (item_id INTEGER, timestamp TIMESTAMP, price INTEGER, volume BIGINT, PRIMARY KEY (item_id, timestamp))')
con_i = duckdb.connect(ITEMS, read_only=True)
ids = [r[0] for r in con_i.execute('SELECT id FROM item_mapping').fetchall()]
con_i.close()
print(f'Backfilling {len(ids)} items...')
for i, iid in enumerate(ids):
    try:
        r = requests.get(f'https://api.weirdgloop.org/exchange/history/osrs/all?id={iid}', timeout=20)
        data = r.json().get(str(iid), [])
        rows = []
        for e in data:
            ts = e['timestamp']
            dt = datetime.fromtimestamp(ts/1000 if ts > 1e11 else ts) if isinstance(ts, int) else datetime.fromisoformat(ts.replace('Z', '+00:00'))
            rows.append((iid, dt, e.get('price',0), e.get('volume',0)))
        con.executemany('INSERT OR IGNORE INTO daily_prices VALUES (?,?,?,?)', rows)
        if i % 100 == 0: print(f'Progress: {i}/{len(ids)}')
        time.sleep(0.2)
    except: continue
con.close()
