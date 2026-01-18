#!/usr/bin/env python3
"""
Generate profitability and calibration reports from trained models.
Analyzes all model metadata to create:
- Profitability report (which items/configs are profitable)
- Calibration summary
- Item viability tier list
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

MODELS_DIR = Path("models")
DATA_DIR = Path("data")

# GE tax is 2%, so net margin for offset X = 2*X - 2%
# offset 1.5% -> net margin = 1% (3% spread - 2% tax)
# offset 2.0% -> net margin = 2% (4% spread - 2% tax)
# offset 2.5% -> net margin = 3% (5% spread - 2% tax)
OFFSET_NET_MARGINS = {
    0.015: 0.01,
    0.02: 0.02,
    0.025: 0.03,
}


def load_item_names():
    """Load item names from expanded_items.json."""
    items_file = DATA_DIR / "expanded_items.json"
    if items_file.exists():
        with open(items_file) as f:
            items = json.load(f)
            return {item["item_id"]: item["name"] for item in items}
    return {}


def collect_all_metadata():
    """Collect all model metadata from models directory."""
    all_meta = []

    for item_dir in MODELS_DIR.iterdir():
        if not item_dir.is_dir():
            continue

        item_id = item_dir.name
        if not item_id.isdigit():
            continue

        # Only load CatBoost meta files (format: Xh_Y.Zpct_meta.json)
        # Skip old roundtrip_* files from previous training
        for meta_file in item_dir.glob("*h_*pct_meta.json"):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                    # Verify this is a CatBoost model with 'auc' field
                    if "auc" in meta:
                        all_meta.append(meta)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read {meta_file}: {e}")

    return all_meta


def analyze_profitability(all_meta, item_names):
    """Analyze profitability across all models."""
    # Group by item
    by_item = defaultdict(list)
    for meta in all_meta:
        by_item[meta["item_id"]].append(meta)

    item_stats = []

    for item_id, models in by_item.items():
        valid_models = [m for m in models if m.get("is_valid", True)]

        if not valid_models:
            continue

        # Calculate average AUC across all models for this item
        avg_auc = sum(m["auc"] for m in valid_models) / len(valid_models)
        max_auc = max(m["auc"] for m in valid_models)
        min_auc = min(m["auc"] for m in valid_models)

        # Best config (highest AUC)
        best_model = max(valid_models, key=lambda m: m["auc"])

        # Count by offset
        offset_counts = defaultdict(list)
        for m in valid_models:
            offset_counts[m["offset"]].append(m["auc"])

        item_name = item_names.get(item_id, models[0].get("item_name", f"Item {item_id}"))

        item_stats.append({
            "item_id": item_id,
            "item_name": item_name,
            "num_models": len(valid_models),
            "avg_auc": avg_auc,
            "max_auc": max_auc,
            "min_auc": min_auc,
            "best_hour": best_model["hour"],
            "best_offset": best_model["offset"],
            "best_auc": best_model["auc"],
            "offset_stats": {
                offset: {
                    "avg_auc": sum(aucs) / len(aucs),
                    "net_margin": OFFSET_NET_MARGINS.get(offset, 0),
                }
                for offset, aucs in offset_counts.items()
            }
        })

    # Sort by average AUC descending
    item_stats.sort(key=lambda x: x["avg_auc"], reverse=True)

    return item_stats


def create_tier_list(item_stats):
    """Create viability tier list based on model quality."""
    tiers = {
        "S": [],  # Excellent (avg AUC >= 0.75)
        "A": [],  # Good (avg AUC >= 0.70)
        "B": [],  # Decent (avg AUC >= 0.65)
        "C": [],  # Marginal (avg AUC >= 0.60)
        "D": [],  # Poor (avg AUC < 0.60)
    }

    for item in item_stats:
        auc = item["avg_auc"]
        if auc >= 0.75:
            tier = "S"
        elif auc >= 0.70:
            tier = "A"
        elif auc >= 0.65:
            tier = "B"
        elif auc >= 0.60:
            tier = "C"
        else:
            tier = "D"

        tiers[tier].append({
            "item_id": item["item_id"],
            "item_name": item["item_name"],
            "avg_auc": round(item["avg_auc"], 4),
            "best_config": f"{item['best_hour']}h @ {item['best_offset']*100:.1f}%",
            "best_auc": round(item["best_auc"], 4),
        })

    return tiers


def generate_markdown_report(item_stats, tiers, all_meta):
    """Generate markdown profitability report."""
    total_models = len(all_meta)
    valid_models = len([m for m in all_meta if m.get("is_valid", True)])
    total_items = len(item_stats)

    # Overall stats
    all_aucs = [m["auc"] for m in all_meta if m.get("is_valid", True)]
    avg_auc_overall = sum(all_aucs) / len(all_aucs) if all_aucs else 0

    # Stats by hour
    hour_stats = defaultdict(list)
    for m in all_meta:
        if m.get("is_valid", True):
            hour_stats[m["hour"]].append(m["auc"])

    # Stats by offset
    offset_stats = defaultdict(list)
    for m in all_meta:
        if m.get("is_valid", True):
            offset_stats[m["offset"]].append(m["auc"])

    report = f"""# GePT Model Profitability Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total Items Trained | {total_items} |
| Total Models | {total_models} |
| Valid Models | {valid_models} |
| Overall Avg AUC | {avg_auc_overall:.4f} |

## Performance by Time Horizon

| Hours | Models | Avg AUC | Interpretation |
|-------|--------|---------|----------------|
"""

    for hour in sorted(hour_stats.keys()):
        aucs = hour_stats[hour]
        avg = sum(aucs) / len(aucs)
        interp = "Excellent" if avg >= 0.75 else "Good" if avg >= 0.70 else "Decent" if avg >= 0.65 else "Fair"
        report += f"| {hour}h | {len(aucs)} | {avg:.4f} | {interp} |\n"

    report += """
## Performance by Offset

| Offset | Net Margin | Models | Avg AUC |
|--------|------------|--------|---------|
"""

    for offset in sorted(offset_stats.keys()):
        aucs = offset_stats[offset]
        avg = sum(aucs) / len(aucs)
        net_margin = OFFSET_NET_MARGINS.get(offset, 0) * 100
        report += f"| {offset*100:.1f}% | {net_margin:.1f}% | {len(aucs)} | {avg:.4f} |\n"

    report += """
## Item Tier List

### Tier S - Excellent (AUC >= 0.75)
"""

    if tiers["S"]:
        report += "| Item | Avg AUC | Best Config | Best AUC |\n|------|---------|-------------|----------|\n"
        for item in tiers["S"][:20]:  # Top 20
            report += f"| {item['item_name']} | {item['avg_auc']:.4f} | {item['best_config']} | {item['best_auc']:.4f} |\n"
        if len(tiers["S"]) > 20:
            report += f"\n*...and {len(tiers['S']) - 20} more S-tier items*\n"
    else:
        report += "*No items in this tier*\n"

    report += """
### Tier A - Good (AUC >= 0.70)
"""

    if tiers["A"]:
        report += "| Item | Avg AUC | Best Config | Best AUC |\n|------|---------|-------------|----------|\n"
        for item in tiers["A"][:20]:
            report += f"| {item['item_name']} | {item['avg_auc']:.4f} | {item['best_config']} | {item['best_auc']:.4f} |\n"
        if len(tiers["A"]) > 20:
            report += f"\n*...and {len(tiers['A']) - 20} more A-tier items*\n"
    else:
        report += "*No items in this tier*\n"

    report += """
### Tier B - Decent (AUC >= 0.65)
"""

    if tiers["B"]:
        report += "| Item | Avg AUC | Best Config | Best AUC |\n|------|---------|-------------|----------|\n"
        for item in tiers["B"][:15]:
            report += f"| {item['item_name']} | {item['avg_auc']:.4f} | {item['best_config']} | {item['best_auc']:.4f} |\n"
        if len(tiers["B"]) > 15:
            report += f"\n*...and {len(tiers['B']) - 15} more B-tier items*\n"
    else:
        report += "*No items in this tier*\n"

    report += """
### Tier C - Marginal (AUC >= 0.60)
"""
    if tiers["C"]:
        report += f"*{len(tiers['C'])} items in this tier*\n"
    else:
        report += "*No items in this tier*\n"

    report += """
### Tier D - Poor (AUC < 0.60)
"""
    if tiers["D"]:
        report += f"*{len(tiers['D'])} items in this tier - consider excluding from production*\n"
    else:
        report += "*No items in this tier*\n"

    report += """
## Top 20 Items by Best Single Model

| Rank | Item | Hour | Offset | AUC |
|------|------|------|--------|-----|
"""

    # Sort all models by AUC
    best_models = sorted(
        [m for m in all_meta if m.get("is_valid", True)],
        key=lambda m: m["auc"],
        reverse=True
    )[:20]

    for i, m in enumerate(best_models, 1):
        name = m.get("item_name", f"Item {m['item_id']}")
        report += f"| {i} | {name} | {m['hour']}h | {m['offset']*100:.1f}% | {m['auc']:.4f} |\n"

    report += """
## Recommendations

Based on this analysis:

1. **Best Time Horizons**: """

    best_hour = max(hour_stats.items(), key=lambda x: sum(x[1])/len(x[1]))[0]
    report += f"{best_hour}h shows the highest average AUC across all items.\n\n"

    report += """2. **Recommended Items for Trading**: Focus on Tier S and A items for highest prediction accuracy.

3. **Offset Strategy**:
   - 2.5% offset: Highest margin (3% net) but requires strong predictions
   - 2.0% offset: Balanced approach (2% net margin)
   - 1.5% offset: Lower margin (1% net) but potentially higher fill rate

4. **Items to Avoid**: Tier D items have AUC < 0.60, meaning predictions are only marginally better than random.
"""

    return report


def main():
    print("=" * 60)
    print("GENERATING PROFITABILITY REPORTS")
    print("=" * 60)

    # Load item names
    print("Loading item names...")
    item_names = load_item_names()
    print(f"Loaded {len(item_names)} item names")

    # Collect all metadata
    print("Collecting model metadata...")
    all_meta = collect_all_metadata()
    print(f"Found {len(all_meta)} models")

    # Analyze profitability
    print("Analyzing profitability...")
    item_stats = analyze_profitability(all_meta, item_names)
    print(f"Analyzed {len(item_stats)} items")

    # Create tier list
    print("Creating tier list...")
    tiers = create_tier_list(item_stats)
    for tier, items in tiers.items():
        print(f"  Tier {tier}: {len(items)} items")

    # Generate markdown report
    print("Generating markdown report...")
    report = generate_markdown_report(item_stats, tiers, all_meta)

    report_path = Path("profitability_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    # Save tier list as JSON
    tier_json_path = Path("item_viability.json")
    with open(tier_json_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_items": len(item_stats),
            "tier_counts": {tier: len(items) for tier, items in tiers.items()},
            "tiers": tiers,
        }, f, indent=2)
    print(f"Saved: {tier_json_path}")

    # Save detailed item stats
    stats_path = Path("item_stats.json")
    with open(stats_path, "w") as f:
        json.dump(item_stats, f, indent=2)
    print(f"Saved: {stats_path}")

    print()
    print("=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total items: {len(item_stats)}")
    print(f"Total models: {len(all_meta)}")
    print(f"Valid models: {len([m for m in all_meta if m.get('is_valid', True)])}")
    print()
    print("Files generated:")
    print(f"  - {report_path}")
    print(f"  - {tier_json_path}")
    print(f"  - {stats_path}")


if __name__ == "__main__":
    main()
