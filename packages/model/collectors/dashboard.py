#!/usr/bin/env python3
"""
GePT Data Factory Dashboard v3.0 - PostgreSQL Edition
======================================================
Live monitoring dashboard for OSRS data collection services.
Now using PostgreSQL/TimescaleDB as the primary data store.
"""

import os
import subprocess
import sys
import time
import argparse
from datetime import datetime, timezone
import psycopg2
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading

# Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "osrs_data")
DB_USER = os.getenv("DB_USER", "osrs_user")
DB_PASS = os.environ["DB_PASS"]
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))

# Tables to monitor
TABLES = [
    {"name": "price_data_5min", "display": "5-Minute Prices", "time_col": "timestamp"},
    {"name": "prices_1h", "display": "Hourly Prices", "time_col": "timestamp"},
    {"name": "prices_latest_1m", "display": "1-Minute Ticks", "time_col": "timestamp"},
    {"name": "predictions", "display": "ML Predictions", "time_col": "time"},
]

# Docker containers to monitor
CONTAINERS = [
    "osrs-ge-collector",
    "osrs-hourly-collector",
    "osrs-news-collector",
    "osrs-latest-1m",
    "osrs-dashboard",
]


def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS
    )


def get_table_stats(conn):
    """Get row count, size, and freshness for each table."""
    stats = []
    with conn.cursor() as cur:
        for table in TABLES:
            try:
                # Row count
                cur.execute(f"SELECT COUNT(*) FROM {table['name']}")
                row_count = cur.fetchone()[0]
                
                # Table size
                cur.execute(f"""
                    SELECT pg_size_pretty(pg_total_relation_size('{table['name']}'))
                """)
                size = cur.fetchone()[0]
                
                # Latest timestamp
                if table['time_col']:
                    cur.execute(f"""
                        SELECT MAX({table['time_col']}), 
                               EXTRACT(EPOCH FROM (NOW() - MAX({table['time_col']})))
                        FROM {table['name']}
                    """)
                    result = cur.fetchone()
                    latest = result[0]
                    age_seconds = result[1] if result[1] else 0
                else:
                    latest = None
                    age_seconds = 0
                
                stats.append({
                    "name": table['name'],
                    "display": table['display'],
                    "rows": row_count,
                    "size": size,
                    "latest": latest.isoformat() if latest else None,
                    "age_seconds": age_seconds,
                    "status": "healthy" if (age_seconds < 600 if table["name"] == "price_data_5min" else age_seconds < 300) else "stale" if age_seconds < 3600 else "critical"
                })
            except Exception as e:
                stats.append({
                    "name": table['name'],
                    "display": table['display'],
                    "rows": 0,
                    "size": "0 bytes",
                    "latest": None,
                    "age_seconds": 0,
                    "status": "error",
                    "error": str(e)
                })
    return stats


def get_container_status():
    """Get Docker container status."""
    statuses = []
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}|{{.Status}}|{{.Ports}}"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("|")
                name = parts[0]
                if name in CONTAINERS:
                    statuses.append({
                        "name": name,
                        "status": parts[1] if len(parts) > 1 else "unknown",
                        "ports": parts[2] if len(parts) > 2 else ""
                    })
    except Exception as e:
        pass
    return statuses


def get_dashboard_data():
    """Collect all dashboard data."""
    try:
        conn = get_db_connection()
        table_stats = get_table_stats(conn)
        conn.close()
    except Exception as e:
        table_stats = [{"error": str(e)}]
    
    container_status = get_container_status()
    
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tables": table_stats,
        "containers": container_status
    }


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logging
    
    def do_GET(self):
        if self.path == "/api/status":
            data = get_dashboard_data()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data, default=str).encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode())


HTML_DASHBOARD = """<!DOCTYPE html>
<html>
<head>
    <title>GePT Data Factory</title>
    <meta charset="utf-8">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
        .card h2 { color: #8b949e; font-size: 12px; text-transform: uppercase; margin-bottom: 12px; }
        .table-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #21262d; }
        .table-row:last-child { border-bottom: none; }
        .table-name { color: #58a6ff; font-weight: 500; }
        .stat { color: #8b949e; font-size: 13px; }
        .stat-value { color: #c9d1d9; font-weight: 600; }
        .status { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
        .status.healthy { background: #3fb950; }
        .status.stale { background: #d29922; }
        .status.critical { background: #f85149; }
        .status.error { background: #f85149; }
        .container-row { display: flex; align-items: center; padding: 6px 0; }
        .container-name { flex: 1; }
        .container-status { font-size: 12px; color: #8b949e; }
        .container-status.up { color: #3fb950; }
        .refresh { color: #8b949e; font-size: 12px; margin-top: 20px; }
        .big-number { font-size: 32px; font-weight: 700; color: #58a6ff; }
    </style>
</head>
<body>
    <h1>GePT Data Factory</h1>
    <div class="grid" id="dashboard">Loading...</div>
    <p class="refresh">Last updated: <span id="updated">-</span> | Auto-refresh: 5s</p>
    <script>
        function formatNumber(n) { return n ? n.toLocaleString() : '0'; }
        function formatAge(seconds) {
            if (!seconds) return 'N/A';
            if (seconds < 60) return Math.round(seconds) + 's ago';
            if (seconds < 3600) return Math.round(seconds/60) + 'm ago';
            if (seconds < 86400) return Math.round(seconds/3600) + 'h ago';
            return Math.round(seconds/86400) + 'd ago';
        }
        async function refresh() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                let html = '<div class="card"><h2>Database Tables</h2>';
                for (const t of data.tables) {
                    html += '<div class="table-row"><div><span class="status ' + t.status + '"></span><span class="table-name">' + t.display + '</span></div>';
                    html += '<div class="stat"><span class="stat-value">' + formatNumber(t.rows) + '</span> rows | ' + t.size + ' | ' + formatAge(t.age_seconds) + '</div></div>';
                }
                html += '</div>';
                html += '<div class="card"><h2>Collectors</h2>';
                for (const c of data.containers) {
                    const isUp = c.status.includes('Up');
                    html += '<div class="container-row"><span class="status ' + (isUp ? 'healthy' : 'critical') + '"></span>';
                    html += '<span class="container-name">' + c.name + '</span>';
                    html += '<span class="container-status ' + (isUp ? 'up' : '') + '">' + c.status + '</span></div>';
                }
                html += '</div>';
                document.getElementById('dashboard').innerHTML = html;
                document.getElementById('updated').textContent = new Date().toLocaleTimeString();
            } catch(e) { console.error(e); }
        }
        refresh();
        setInterval(refresh, 5000);
    </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT)
    args = parser.parse_args()
    
    print(f"Starting GePT Dashboard on port {args.port}...")
    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
