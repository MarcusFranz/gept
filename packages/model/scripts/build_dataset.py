#!/usr/bin/env python3
"""
CLI for building datasets from recipes

Usage:
    # List available recipes
    python scripts/build_dataset.py --list

    # Build from a recipe
    python scripts/build_dataset.py --recipe baseline

    # Build from a recipe with custom suffix
    python scripts/build_dataset.py --recipe high_volume --suffix _jan2026

    # Initialize standard recipe library
    python scripts/build_dataset.py --init-recipes
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets.dataset_builder import DatasetBuilder
from datasets.dataset_recipe import RecipeLibrary, initialize_recipe_library
from loguru import logger


def list_recipes():
    """List all available recipes"""
    library = RecipeLibrary()
    recipes = library.list_recipes()

    if not recipes:
        print("No recipes found. Run with --init-recipes to create standard recipes.")
        return

    print("\n📋 Available Dataset Recipes:\n")
    for name in sorted(recipes):
        try:
            info = library.get_recipe_info(name)
            print(f"  {name}")
            print(f"    Description: {info['description']}")
            print(f"    Granularity: {info['granularity']}")
            print(f"    Feature Set: {info['feature_set']}")
            print()
        except Exception as e:
            print(f"  {name} (error loading: {e})")


def build_from_recipe(recipe_name: str, suffix: Optional[str] = None):
    """Build a dataset from a recipe"""
    logger.info(f"Building dataset from recipe: {recipe_name}")

    # Load recipe
    library = RecipeLibrary()
    try:
        recipe = library.load_recipe(recipe_name)
    except FileNotFoundError:
        logger.error(f"Recipe not found: {recipe_name}")
        logger.info("Available recipes:")
        for name in library.list_recipes():
            logger.info(f"  - {name}")
        sys.exit(1)

    # Build dataset
    builder = DatasetBuilder(granularity_minutes=recipe.get_granularity_minutes())
    metadata = builder.build_from_recipe(recipe, version_suffix=suffix)

    # Print summary
    print("\n✅ Dataset Build Complete!\n")
    print(f"Version: {metadata.version}")
    print(f"Description: {metadata.description}")
    print(f"Items: {metadata.item_count}")
    print(f"Date Range: {metadata.date_range_start} to {metadata.date_range_end}")
    print(f"Granularity: {metadata.granularity_minutes} minutes")
    print(f"\nDataset Split:")
    print(f"  Train: {metadata.train_rows:,} rows")
    print(f"  Val:   {metadata.val_rows:,} rows")
    print(f"  Test:  {metadata.test_rows:,} rows")
    print(f"  Total: {metadata.train_rows + metadata.val_rows + metadata.test_rows:,} rows")
    print(f"\nQuality Metrics:")
    print(f"  Avg Completeness: {metadata.avg_completeness:.2%}")
    print(f"  Avg Volume: {metadata.avg_volume:,.0f}")
    print(f"\nFiles saved to:")
    print(f"  {Path(builder.output_dir) / metadata.version}")


def init_recipes():
    """Initialize standard recipe library"""
    print("Initializing standard recipe library...")
    initialize_recipe_library()
    print("\n✅ Recipe library initialized!")


def main():
    parser = argparse.ArgumentParser(
        description="Build datasets from recipes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List recipes
  python scripts/build_dataset.py --list

  # Build baseline dataset
  python scripts/build_dataset.py --recipe baseline

  # Build with custom suffix
  python scripts/build_dataset.py --recipe high_volume --suffix _jan2026

  # Initialize recipe library
  python scripts/build_dataset.py --init-recipes
        """
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available recipes"
    )

    parser.add_argument(
        "--recipe",
        type=str,
        help="Recipe name to build"
    )

    parser.add_argument(
        "--suffix",
        type=str,
        help="Optional suffix for version name"
    )

    parser.add_argument(
        "--init-recipes",
        action="store_true",
        help="Initialize standard recipe library"
    )

    args = parser.parse_args()

    # Handle commands
    if args.init_recipes:
        init_recipes()
    elif args.list:
        list_recipes()
    elif args.recipe:
        build_from_recipe(args.recipe, args.suffix)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
