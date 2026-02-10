import duckdb
import pandas as pd
import numpy as np
import os

# Configuration
DATA_DIR = "/home/ubuntu/osrs_collector/data"
ITEMS_DB = os.path.join(DATA_DIR, "items.duckdb")
RECIPES_DB = os.path.join(DATA_DIR, "recipes.duckdb")
CLUSTERS_DB = os.path.join(DATA_DIR, "meta_clusters.duckdb")
VOLUMES_DB = os.path.join(DATA_DIR, "daily_volumes.duckdb")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed", "item_properties.parquet")

def compile_properties():
    print("Connecting to databases...")
    con = duckdb.connect(ITEMS_DB)
    
    # 1. Attach other databases
    con.execute(f"ATTACH '{RECIPES_DB}' AS db_recipes")
    con.execute(f"ATTACH '{CLUSTERS_DB}' AS db_clusters")
    con.execute(f"ATTACH '{VOLUMES_DB}' AS db_volumes")

    print("Calculating recursive recipe depth...")
    # Calculate depth: raw materials = 0, items made from them = 1, etc.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE recipe_metrics AS
        WITH RECURSIVE depth_calc AS (
            SELECT DISTINCT r.input_id as item_id, 0 as depth
            FROM db_recipes.recipes r
            WHERE r.input_id NOT IN (SELECT output_id FROM db_recipes.recipes)
            
            UNION ALL
            
            SELECT r.output_id, dc.depth + 1
            FROM db_recipes.recipes r
            JOIN depth_calc dc ON r.input_id = dc.item_id
            WHERE dc.depth < 5
        )
        SELECT 
            item_id, 
            MAX(depth) as recipe_depth,
            (SELECT COUNT(*) FROM db_recipes.recipes WHERE input_id = depth_calc.item_id) as num_recipe_outputs
        FROM depth_calc
        GROUP BY 1
    """)

    print("Refining combat styles and tiers...")
    
    # FIX: Qualify m.name to avoid ambiguity
    combat_style_sql = """
        CASE 
            WHEN e.combat_style != 'none' THEN e.combat_style
            WHEN e.equipment_slot IN ('weapon', '2h') THEN
                CASE 
                    WHEN m.name LIKE '%staff%' OR m.name LIKE '%wand%' OR m.name LIKE '%mystic%' THEN 'magic'
                    WHEN m.name LIKE '%bow%' OR m.name LIKE '%bolt%' OR m.name LIKE '%arrow%' OR m.name LIKE '%dart%' THEN 'ranged'
                    ELSE 'melee'
                END
            ELSE 'none'
        END
    """

    price_tier_sql = """
        CASE 
            WHEN value < 1000 THEN 1
            WHEN value < 10000 THEN 2
            WHEN value < 100000 THEN 3
            WHEN value < 1000000 THEN 4
            WHEN value < 10000000 THEN 5
            WHEN value >= 10000000 THEN 6
            ELSE 0
        END
    """

    volume_tier_sql = """
        CASE 
            WHEN vol.avg_vol < 100 THEN 1
            WHEN vol.avg_vol < 1000 THEN 2
            WHEN vol.avg_vol < 10000 THEN 3
            WHEN vol.avg_vol < 100000 THEN 4
            WHEN vol.avg_vol >= 100000 THEN 5
            ELSE 1
        END
    """

    final_query = f"""
        WITH avg_vols AS (
            SELECT item_id, AVG(volume) as avg_vol
            FROM db_volumes.daily_volumes
            GROUP BY 1
        ),
        recipes_feat AS (
            SELECT 
                output_id as item_id,
                COUNT(DISTINCT input_id) as num_recipe_inputs,
                1 as is_producible
            FROM db_recipes.recipes
            GROUP BY 1
        )
        
        SELECT 
            m.id as item_id,
            m.name as item_name,
            e.equipment_slot,
            {combat_style_sql} as combat_style,
            e.content_type,
            e.content_source,
            {price_tier_sql} as price_tier,
            {volume_tier_sql} as volume_tier,
            m.limit_ge as ge_limit,
            COALESCE(c.meta_cluster, 0) as meta_cluster,
            
            -- Continuous
            LOG(GREATEST(m.limit_ge, 1)) as log_ge_limit,
            LOG(GREATEST(m.highalch, 1)) as log_high_alch,
            CAST(m.members AS INTEGER) as is_members,
            CAST(e.is_stackable AS INTEGER) as is_stackable,
            CAST(e.is_equipable AS INTEGER) as is_equipable,
            CAST(e.is_consumable AS INTEGER) as is_consumable,
            CAST(e.is_raid_drop AS INTEGER) as is_raid_drop,
            CAST(e.is_boss_drop AS INTEGER) as is_boss_drop,
            CAST(e.is_skilling_supply AS INTEGER) as is_skilling_supply,
            
            -- Recipe graph features
            COALESCE(rm.recipe_depth, 0) as recipe_depth,
            COALESCE(r.num_recipe_inputs, 0) as num_recipe_inputs,
            COALESCE(rm.num_recipe_outputs, 0) as num_recipe_outputs,
            COALESCE(r.is_producible, 0) as is_producible
            
        FROM item_mapping m
        JOIN item_enriched e ON m.id = e.id
        LEFT JOIN avg_vols vol ON m.id = vol.item_id
        LEFT JOIN recipes_feat r ON m.id = r.item_id
        LEFT JOIN recipe_metrics rm ON m.id = rm.item_id
        LEFT JOIN (
            SELECT source_item as item_id, arg_max(target_item, weight) as meta_cluster
            FROM db_clusters.cluster_edges
            GROUP BY 1
        ) c ON m.id = c.item_id
    """

    df = con.execute(final_query).fetchdf()
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Final item properties compiled to {OUTPUT_PATH}")
    print(f"Total Items: {len(df)}")
    print("\nCombat Style Distribution:")
    print(df['combat_style'].value_counts())

if __name__ == "__main__":
    compile_properties()
