#!/usr/bin/env python3
"""
Fetch an item's predictions from the production engine host (via SSH) and render
an SVG graph for the next 48 hour offsets.

Notes on what this graphs:
- The engine "predictions" are not a time-series market price forecast.
- They are precomputed flip configurations (buy/sell price, fill probability,
  expected value) conditioned on a target fill window (hour_offset) and an
  offset_pct setting.

This tool selects (for each hour_offset) the "best" row by the same score used
by the engine's /api/v1/predictions/{item_id} endpoint:
  expected_value * fill_probability
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import urllib.parse


PROD_SSH_HOST = os.environ.get("GEPT_PROD_SSH", "ubuntu@150.136.170.128")
SSH_KEY_PATH = os.environ.get(
    "GEPT_SSH_KEY",
    str(pathlib.Path(__file__).resolve().parents[1] / ".secrets" / "oracle_key.pem"),
)
KNOWN_HOSTS_PATH = os.environ.get(
    "GEPT_KNOWN_HOSTS",
    str(pathlib.Path(__file__).resolve().parents[1] / "infra" / "ssh" / "known_hosts"),
)


def _ssh_curl_json(url: str) -> object:
    # Fetch via SSH so we never need to handle INTERNAL_API_KEY locally.
    # The remote host sources /home/ubuntu/gept/.env which contains INTERNAL_API_KEY.
    remote = (
        "set -a; source /home/ubuntu/gept/.env; set +a; "
        f'curl -s -H "X-API-Key: $INTERNAL_API_KEY" "{url}"'
    )
    cmd = [
        "ssh",
        "-i",
        SSH_KEY_PATH,
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS_PATH}",
        "-o",
        "StrictHostKeyChecking=yes",
        PROD_SSH_HOST,
        remote,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    # curl may return empty (shouldn't) - guard for better error message
    raw = out.decode("utf-8", errors="replace").strip()
    if not raw:
        raise RuntimeError("Empty response from engine (curl returned no body).")
    return json.loads(raw)


def _ssh_docker_python_json(py_src: str) -> object:
    # Run a python snippet inside the prod container and parse JSON from stdout.
    remote = f"docker exec -i gept-engine python - <<'PY'\n{py_src}\nPY"
    cmd = [
        "ssh",
        "-i",
        SSH_KEY_PATH,
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS_PATH}",
        "-o",
        "StrictHostKeyChecking=yes",
        PROD_SSH_HOST,
        remote,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    raw = out.decode("utf-8", errors="replace").strip()
    if not raw:
        raise RuntimeError("Empty response from docker python snippet.")
    return json.loads(raw)


def get_predictions_from_prod_db(item_id: int, model_id: str) -> dict:
    # Only returns non-sensitive prediction rows.
    py = f"""
import json, os
from sqlalchemy import create_engine, text

item_id = {item_id}
model_id = {json.dumps(model_id)}

engine = create_engine(os.environ["DB_CONNECTION_STRING"])
with engine.connect() as conn:
    item_time = conn.execute(
        text("select max(time) from predictions where item_id=:id and model_id=:mid"),
        {{"id": item_id, "mid": model_id}},
    ).scalar()

    if not item_time:
        print(json.dumps({{"ok": False, "error": "no_predictions_for_item_model"}}))
    else:
        rows = conn.execute(
            text(\"\"\"
                select
                       time::text as time,
                       item_id,
                       item_name,
                       hour_offset,
                       offset_pct::float as offset_pct,
                       fill_probability::float as fill_probability,
                       expected_value::float as expected_value,
                       buy_price::float as buy_price,
                       sell_price::float as sell_price,
                       confidence
                from predictions
                where item_id=:id and model_id=:mid and time=:t and hour_offset between 1 and 48
                order by hour_offset asc, offset_pct asc
            \"\"\"),
            {{"id": item_id, "mid": model_id, "t": item_time}},
        ).mappings().all()
        out = {{"ok": True, "time": str(item_time), "item_id": item_id, "model_id": model_id, "rows": [dict(r) for r in rows]}}
        print(json.dumps(out))
"""
    return _ssh_docker_python_json(py)


def search_item(query: str, limit: int = 10) -> list[dict]:
    q = urllib.parse.quote(query)
    url = f"http://127.0.0.1:8001/api/v1/items/search?q={q}&limit={limit}"
    data = _ssh_curl_json(url)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected search response type: {type(data)}")
    return data


def get_predictions(item_id: int) -> dict:
    url = f"http://127.0.0.1:8001/api/v1/predictions/{item_id}"
    data = _ssh_curl_json(url)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected predictions response type: {type(data)}")
    return data


def choose_best_per_hour(all_rows: list[dict], max_hour: int = 48) -> list[dict]:
    by_hour: dict[int, dict] = {}
    for row in all_rows:
        try:
            h = int(row.get("hour_offset"))
        except Exception:
            continue
        if h < 1 or h > max_hour:
            continue
        ev = float(row.get("expected_value") or 0.0)
        fp = float(row.get("fill_probability") or 0.0)
        score = ev * fp
        cur = by_hour.get(h)
        if cur is None or score > float(cur["_score"]):
            row2 = dict(row)
            row2["_score"] = score
            by_hour[h] = row2

    out = [by_hour[h] for h in sorted(by_hour.keys())]
    return out


def _nice_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        return [vmin]
    span = vmax - vmin
    raw_step = span / max(1, (n - 1))
    mag = 10 ** math.floor(math.log10(raw_step))
    step = mag
    for s in [1, 2, 2.5, 5, 10]:
        if raw_step <= s * mag:
            step = s * mag
            break
    start = math.floor(vmin / step) * step
    ticks = []
    x = start
    # guard: at most 50 ticks
    while x <= vmax + 1e-9 and len(ticks) < 50:
        if x >= vmin - 1e-9:
            ticks.append(x)
        x += step
    if not ticks:
        ticks = [vmin, vmax]
    return ticks


def render_svg(
    *,
    title: str,
    subtitle: str,
    hours: list[int],
    buy: list[float],
    sell: list[float],
    fill_prob: list[float],
    output_path: pathlib.Path,
) -> None:
    W, H = 1200, 720
    margin_l, margin_r, margin_t, margin_b = 80, 40, 70, 50
    gap = 35
    top_h = 420
    bottom_h = H - margin_t - margin_b - top_h - gap

    def x_map(hh: float) -> float:
        xmin, xmax = 1.0, 48.0
        return margin_l + (hh - xmin) / (xmax - xmin) * (W - margin_l - margin_r)

    # Top chart (prices)
    y_min = min(min(buy), min(sell))
    y_max = max(max(buy), max(sell))
    pad = max(1.0, (y_max - y_min) * 0.08)
    y_min -= pad
    y_max += pad

    def y_price_map(v: float) -> float:
        return margin_t + top_h - (v - y_min) / (y_max - y_min) * top_h

    # Bottom chart (fill probability)
    fp_min = 0.0
    fp_max = 1.0

    def y_fp_map(v: float) -> float:
        y0 = margin_t + top_h + gap
        return y0 + bottom_h - (v - fp_min) / (fp_max - fp_min) * bottom_h

    def polyline(xs: list[float], ys: list[float]) -> str:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
        return pts

    x_vals = [x_map(h) for h in hours]
    buy_y = [y_price_map(v) for v in buy]
    sell_y = [y_price_map(v) for v in sell]
    fp_y = [y_fp_map(v) for v in fill_prob]

    price_ticks = _nice_ticks(y_min, y_max, n=6)
    hour_ticks = list(range(1, 49, 6))
    fp_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # SVG template (minimal but readable)
    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    lines.append("<style>")
    lines.append("  .bg{fill:#0b1220}")
    lines.append("  .grid{stroke:#22314f;stroke-width:1;opacity:0.55}")
    lines.append("  .axis{stroke:#5770a8;stroke-width:1.2;opacity:0.9}")
    lines.append("  .txt{fill:#d7e0ff;font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;}")
    lines.append("  .muted{fill:#97a6d7}")
    lines.append("  .buy{stroke:#3aa0ff;stroke-width:2.2;fill:none}")
    lines.append("  .sell{stroke:#25d3a5;stroke-width:2.2;fill:none}")
    lines.append("  .fp{stroke:#ff5c7a;stroke-width:2.0;fill:none}")
    lines.append("  .legend{font-size:14px}")
    lines.append("  .title{font-size:22px;font-weight:700}")
    lines.append("  .sub{font-size:14px}")
    lines.append("</style>")
    lines.append(f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>')

    # Titles
    lines.append(f'<text class="txt title" x="{margin_l}" y="34">{_xml(title)}</text>')
    lines.append(f'<text class="txt muted sub" x="{margin_l}" y="56">{_xml(subtitle)}</text>')

    # Price grid + axes
    x0 = margin_l
    x1 = W - margin_r
    y_top0 = margin_t
    y_top1 = margin_t + top_h
    lines.append(f'<line class="axis" x1="{x0}" y1="{y_top0}" x2="{x0}" y2="{y_top1}"/>')
    lines.append(f'<line class="axis" x1="{x0}" y1="{y_top1}" x2="{x1}" y2="{y_top1}"/>')

    for t in price_ticks:
        yy = y_price_map(t)
        lines.append(f'<line class="grid" x1="{x0}" y1="{yy:.2f}" x2="{x1}" y2="{yy:.2f}"/>')
        lines.append(
            f'<text class="txt muted" x="{x0-10}" y="{yy+4:.2f}" text-anchor="end">{_fmt_gp(t)}</text>'
        )

    for h in hour_ticks:
        xx = x_map(h)
        lines.append(f'<line class="grid" x1="{xx:.2f}" y1="{y_top0}" x2="{xx:.2f}" y2="{y_top1}"/>')
        lines.append(f'<text class="txt muted" x="{xx:.2f}" y="{y_top1+22}" text-anchor="middle">{h}h</text>')

    lines.append(f'<text class="txt muted" x="{x0}" y="{y_top0-12}">Price (gp)</text>')
    lines.append(f'<text class="txt muted" x="{(x0+x1)/2:.2f}" y="{y_top1+42}" text-anchor="middle">Hour offset (target fill window)</text>')

    # Price lines
    lines.append(f'<polyline class="buy" points="{polyline(x_vals, buy_y)}"/>')
    lines.append(f'<polyline class="sell" points="{polyline(x_vals, sell_y)}"/>')

    # Bottom chart axes/grid
    y_bot0 = margin_t + top_h + gap
    y_bot1 = y_bot0 + bottom_h
    lines.append(f'<line class="axis" x1="{x0}" y1="{y_bot0}" x2="{x0}" y2="{y_bot1}"/>')
    lines.append(f'<line class="axis" x1="{x0}" y1="{y_bot1}" x2="{x1}" y2="{y_bot1}"/>')
    for t in fp_ticks:
        yy = y_fp_map(t)
        lines.append(f'<line class="grid" x1="{x0}" y1="{yy:.2f}" x2="{x1}" y2="{yy:.2f}"/>')
        lines.append(f'<text class="txt muted" x="{x0-10}" y="{yy+4:.2f}" text-anchor="end">{t:.1f}</text>')
    for h in hour_ticks:
        xx = x_map(h)
        lines.append(f'<line class="grid" x1="{xx:.2f}" y1="{y_bot0}" x2="{xx:.2f}" y2="{y_bot1}"/>')
    lines.append(f'<text class="txt muted" x="{x0}" y="{y_bot0-12}">Fill probability</text>')
    lines.append(f'<polyline class="fp" points="{polyline(x_vals, fp_y)}"/>')

    # Legend
    lx, ly = x1 - 260, margin_t - 36
    lines.append(f'<g class="legend">')
    lines.append(f'<line class="buy" x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}"/><text class="txt" x="{lx+36}" y="{ly+5}">Buy price</text>')
    lines.append(f'<line class="sell" x1="{lx}" y1="{ly+18}" x2="{lx+28}" y2="{ly+18}"/><text class="txt" x="{lx+36}" y="{ly+23}">Sell price</text>')
    lines.append(f'<line class="fp" x1="{lx}" y1="{ly+36}" x2="{lx+28}" y2="{ly+36}"/><text class="txt" x="{lx+36}" y="{ly+41}">Fill probability</text>')
    lines.append(f'</g>')

    lines.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _xml(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _fmt_gp(v: float) -> str:
    # Human-friendly OSRS gp formatting for axis ticks.
    try:
        x = float(v)
    except Exception:
        return str(v)
    ax = abs(x)
    if ax >= 1_000_000:
        s = f"{x/1_000_000:.1f}M"
        return s.replace(".0M", "M")
    if ax >= 1_000:
        s = f"{x/1_000:.1f}K"
        return s.replace(".0K", "K")
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.0f}"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="", help="Item name (looked up via /api/v1/items/search unless --item-id is provided)")
    ap.add_argument("--limit", type=int, default=10, help="Search result limit")
    ap.add_argument("--out", default="out", help="Output directory (repo-relative)")
    ap.add_argument("--pick", type=int, default=0, help="Which search result to pick (0-based)")
    ap.add_argument("--item-id", type=int, default=0, help="Skip search and use this item_id")
    ap.add_argument("--model-id", default="", help="When using --source=prod-db, fetch predictions for this model_id")
    ap.add_argument("--source", choices=["api", "prod-db"], default="api", help="Where to fetch predictions from")
    args = ap.parse_args(argv)

    item_id = args.item_id
    item_name = ""
    matches: list[dict] = []

    if item_id <= 0:
        if not args.query:
            print("Provide a query string or --item-id.", file=sys.stderr)
            return 2
        matches = search_item(args.query, limit=args.limit)
        if not matches:
            print(f'No engine items found for query: "{args.query}".', file=sys.stderr)
            print('Tip: try a broader query like "rune" to confirm search is working.', file=sys.stderr)
            return 2

        pick = args.pick
        if pick < 0 or pick >= len(matches):
            print(f"--pick out of range (0..{len(matches)-1})", file=sys.stderr)
            return 2

        chosen = matches[pick]
        item_id = int(chosen["itemId"])
        item_name = str(chosen["name"])

    pred_time = None
    best_by_hour: list[dict] = []
    best_cfg = {}
    source_note = ""

    if args.source == "api":
        pred = get_predictions(item_id)
        rows = pred.get("all_predictions") or []
        best_by_hour = choose_best_per_hour(rows, max_hour=48)
        item_name = item_name or str(pred.get("item_name") or f"Item {item_id}")
        best_cfg = pred.get("best_config") or {}
        source_note = "engine /api/v1/predictions"
    else:
        if not args.model_id:
            print("--model-id is required when --source=prod-db", file=sys.stderr)
            return 2
        data = get_predictions_from_prod_db(item_id, args.model_id)
        if not data.get("ok"):
            print(f"No prod-db predictions for item_id={item_id} model_id={args.model_id}", file=sys.stderr)
            return 3
        pred_time = data.get("time")
        rows = data.get("rows") or []
        item_name = item_name or str(rows[0].get("item_name") if rows else f"Item {item_id}")
        best_by_hour = choose_best_per_hour(rows, max_hour=48)
        best_cfg = {}
        source_note = f"prod-db model_id={args.model_id} time={pred_time}"

    if not best_by_hour:
        print(f"No usable prediction rows for item {item_id} ({item_name})", file=sys.stderr)
        return 3

    hours = [int(r["hour_offset"]) for r in best_by_hour]
    buy = [float(r.get("buy_price") or 0) for r in best_by_hour]
    sell = [float(r.get("sell_price") or 0) for r in best_by_hour]
    fp = [float(r.get("fill_probability") or 0.0) for r in best_by_hour]

    subtitle = f"{source_note} | selected per-hour best by expected_value*fill_probability"
    if best_cfg:
        subtitle = (
            f'Engine best_config: hour_offset={best_cfg.get("hour_offset")} '
            f'offset_pct={best_cfg.get("offset_pct")} | {subtitle}'
        )

    out_dir = pathlib.Path(args.out)
    fname = f"predictions_{item_id}_{item_name.lower().replace(' ', '_')}_48h.svg"
    out_path = out_dir / fname
    render_svg(
        title=f"{item_name} (item_id={item_id})",
        subtitle=subtitle,
        hours=hours,
        buy=buy,
        sell=sell,
        fill_prob=fp,
        output_path=out_path,
    )

    # Print chosen results and output path (no secrets).
    print(json.dumps({"picked": {"itemId": item_id, "name": item_name}, "svg": str(out_path)}, indent=2))
    if len(matches) > 1:
        print("Other matches:")
        for i, m in enumerate(matches[: min(10, len(matches))]):
            if i == pick:
                continue
            print(f"  [{i}] {m.get('itemId')} {m.get('name')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
