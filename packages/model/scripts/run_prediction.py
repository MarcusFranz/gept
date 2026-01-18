#!/usr/bin/env python3
"""
GE Flipping Prediction System - Main Entry Point

Usage:
    python run_prediction.py              # Get all predictions
    python run_prediction.py --item 565   # Predict for specific item
    python run_prediction.py --best       # Show best opportunities
"""

import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

from predictor import GEPTPredictor


def main():
    parser = argparse.ArgumentParser(description='GE Flipping Predictions')
    parser.add_argument('--item', type=int, help='Specific item ID to predict')
    parser.add_argument('--best', action='store_true', help='Show best opportunities')
    parser.add_argument('--min-ev', type=float, default=0.005, help='Minimum expected value (default: 0.5%%)')
    args = parser.parse_args()

    print("="*60)
    print("GE FLIPPING PREDICTION SYSTEM")
    print("="*60)

    predictor = GEPTPredictor()
    items = predictor.registry.get_available_items()
    print(f"Loaded models for {len(items)} items: {items}\n")

    if args.best:
        print("BEST OPPORTUNITIES (EV >= {:.1%}):".format(args.min_ev))
        print("-"*60)
        opportunities = predictor.get_best_opportunities(min_ev=args.min_ev)

        if not opportunities:
            print("No opportunities found above threshold.")
            return

        for pred in opportunities[:10]:
            print(f"\n{pred.item_name} (ID: {pred.item_id})")
            print(f"  Strategy: {pred.offset_pct:.1f}% offset, {pred.window_hours}h window")
            print(f"  Fill Probability: {pred.fill_probability:.1%}")
            print(f"  Expected Value: {pred.expected_value*100:.2f}%")
            print(f"  Buy at: {pred.buy_target_price:,.0f} gp")
            print(f"  Sell at: {pred.sell_target_price:,.0f} gp")
            print(f"  Confidence: {pred.confidence_tier}")

    elif args.item:
        print(f"PREDICTIONS FOR ITEM {args.item}:")
        print("-"*60)
        predictions = predictor.predict_item(args.item)

        if not predictions:
            print(f"No predictions available for item {args.item}")
            return

        for pred in predictions:
            print(f"\n{pred.target}:")
            print(f"  Fill Probability: {pred.fill_probability:.1%}")
            print(f"  Expected Value: {pred.expected_value*100:.2f}%")
            print(f"  Current Prices: High={pred.current_high:,.0f}, Low={pred.current_low:,.0f}")
            print(f"  Buy Target: {pred.buy_target_price:,.0f} gp")
            print(f"  Sell Target: {pred.sell_target_price:,.0f} gp")
            print(f"  Confidence: {pred.confidence_tier}")

    else:
        print("ALL PREDICTIONS:")
        print("-"*60)

        all_preds = predictor.predict_all()

        for item_id, predictions in all_preds.items():
            print(f"\n{predictions[0].item_name} (ID: {item_id}):")
            for pred in predictions:
                print(f"  {pred.target}: prob={pred.fill_probability:.1%}, "
                      f"EV={pred.expected_value*100:.2f}%, conf={pred.confidence_tier}")


if __name__ == "__main__":
    main()
