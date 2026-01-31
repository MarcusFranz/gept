"""Prediction loader for fetching pre-computed predictions from PostgreSQL."""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


# OSRS item acronyms for search expansion
# Source: https://oldschool.runescape.wiki/w/Acronyms
OSRS_ACRONYMS = {
    # === GODSWORDS ===
    "ags": "armadyl godsword",
    "bgs": "bandos godsword",
    "sgs": "saradomin godsword",
    "zgs": "zamorak godsword",
    # === MELEE WEAPONS ===
    "dhl": "dragon hunter lance",
    "lance": "dragon hunter lance",
    "rapier": "ghrazi rapier",
    "rap": "ghrazi rapier",
    "blade": "blade of saeldor",
    "scythe": "scythe of vitur",
    "scy": "scythe of vitur",
    "dwh": "dragon warhammer",
    "dwar": "dragon warhammer",
    "dclaws": "dragon claws",
    "gmaul": "granite maul",
    "dds": "dragon dagger",
    "ddp": "dragon dagger",
    "dscim": "dragon scimitar",
    "dscimmy": "dragon scimitar",
    "d2h": "dragon 2h sword",
    "hally": "crystal halberd",
    "chally": "crystal halberd",
    "vls": "vesta's longsword",
    "vw": "voidwaker",
    "sra": "soulreaper axe",
    "abby whip": "abyssal whip",
    "whip": "abyssal whip",
    "tent": "abyssal tentacle",
    "bludgeon": "abyssal bludgeon",
    "saeldor": "blade of saeldor",
    "inq mace": "inquisitor's mace",
    # === RANGED WEAPONS ===
    "tbow": "twisted bow",
    "bp": "toxic blowpipe",
    "bpipe": "toxic blowpipe",
    "blowpipe": "toxic blowpipe",
    "acb": "armadyl crossbow",
    "dcb": "dragon crossbow",
    "zcb": "zaryte crossbow",
    "zbow": "zaryte crossbow",
    "dhcb": "dragon hunter crossbow",
    "dhc": "dragon hunter crossbow",
    "bowfa": "bow of faerdhinen",
    "bofa": "bow of faerdhinen",
    "fbow": "bow of faerdhinen",
    "cbow": "crystal bow",
    "dbow": "dark bow",
    "msb": "magic shortbow",
    "msbi": "magic shortbow (i)",
    "kcb": "karil's crossbow",
    "kbow": "karil's crossbow",
    "bally": "heavy ballista",
    "ballista": "heavy ballista",
    "vbow": "venator bow",
    "rcb": "rune crossbow",
    # === MAGIC WEAPONS ===
    "sang": "sanguinesti staff",
    "kodai": "kodai wand",
    "sotd": "staff of the dead",
    "tsotd": "toxic staff of the dead",
    "swamp": "trident of the swamp",
    "trident": "trident of the seas",
    "sol": "staff of light",
    "harm": "harmonised nightmare staff",
    "harmo": "harmonised nightmare staff",
    "eldritch": "eldritch nightmare staff",
    "eld": "eldritch nightmare staff",
    "voli": "volatile nightmare staff",
    "volly": "volatile nightmare staff",
    "volatile": "volatile nightmare staff",
    "nightmare staff": "nightmare staff",
    "shadow": "tumeken's shadow",
    "tumeken": "tumeken's shadow",
    # === SHIELDS & DEFENDERS ===
    "dfs": "dragonfire shield",
    "dfw": "dragonfire ward",
    "dfward": "dragonfire ward",
    "aws": "ancient wyvern shield",
    "ely": "elysian spirit shield",
    "arcane": "arcane spirit shield",
    "spec": "spectral spirit shield",
    "spectral": "spectral spirit shield",
    "bss": "blessed spirit shield",
    "bulwark": "dinh's bulwark",
    "dinhs": "dinh's bulwark",
    "tbuck": "twisted buckler",
    "buckler": "twisted buckler",
    "avernic": "avernic defender",
    "ddef": "dragon defender",
    "antid": "anti-dragon shield",
    "malediction": "malediction ward",
    "odium": "odium ward",
    # === HELMETS ===
    "serp": "serpentine helm",
    "serpentine": "serpentine helm",
    "nezzy": "helm of neitiznot",
    "nez": "helm of neitiznot",
    "faceguard": "neitiznot faceguard",
    "fg": "neitiznot faceguard",
    "dfh": "dragon full helm",
    "dmed": "dragon med helm",
    "rmed": "rune med helm",
    "rmh": "rune med helm",
    "rfh": "rune full helm",
    "slayer helm": "slayer helmet",
    "slay helm": "slayer helmet",
    # === BODY ARMOR ===
    "bcp": "bandos chestplate",
    "acp": "armadyl chestplate",
    "acs": "armadyl chainskirt",
    "dplate": "dragon platebody",
    "rpb": "rune platebody",
    "ft": "fighter torso",
    "torso": "fighter torso",
    "brassy": "verac's brassard",
    "karils top": "karil's leathertop",
    "ahrims top": "ahrim's robetop",
    # === LEG ARMOR ===
    "tassets": "bandos tassets",
    "tassys": "bandos tassets",
    "dlegs": "dragon platelegs",
    "dskirt": "dragon plateskirt",
    "vskirt": "verac's plateskirt",
    "karils skirt": "karil's leatherskirt",
    "ahrims skirt": "ahrim's robeskirt",
    # === BOOTS ===
    "prims": "primordial boots",
    "primordials": "primordial boots",
    "pegs": "pegasian boots",
    "peggies": "pegasian boots",
    "pegasians": "pegasian boots",
    "eternals": "eternal boots",
    "dboots": "dragon boots",
    "guardian boots": "guardian boots",
    # === GLOVES ===
    "bgloves": "barrows gloves",
    "b gloves": "barrows gloves",
    "barrows gloves": "barrows gloves",
    "ferocious": "ferocious gloves",
    "torm": "tormented bracelet",
    "tormented": "tormented bracelet",
    "vambs": "vambraces",
    "vams": "vambraces",
    "zvambs": "zaryte vambraces",
    "zvams": "zaryte vambraces",
    # === CAPES ===
    "fcape": "fire cape",
    "fc": "fire cape",
    "inf": "infernal cape",
    "infernal": "infernal cape",
    "ava": "ava's assembler",
    "avas": "ava's assembler",
    "assembler": "ava's assembler",
    "accumulator": "ava's accumulator",
    "ma2 cape": "imbued god cape",
    "god cape": "imbued god cape",
    "ardy cloak": "ardougne cloak",
    # === AMULETS & NECKLACES ===
    "fury": "amulet of fury",
    "torture": "amulet of torture",
    "tort": "amulet of torture",
    "anguish": "necklace of anguish",
    "ang": "necklace of anguish",
    "occult": "occult necklace",
    "occ": "occult necklace",
    "glory": "amulet of glory",
    "salve": "salve amulet",
    "salve e": "salve amulet (e)",
    "salve ei": "salve amulet (ei)",
    "pneck": "phoenix necklace",
    "blood fury": "amulet of blood fury",
    "bf": "amulet of blood fury",
    "aotd": "amulet of the damned",
    "torture orn": "amulet of torture (or)",
    "anguish orn": "necklace of anguish (or)",
    # === RINGS ===
    "suffering": "ring of suffering",
    "ros": "ring of suffering",
    "suff": "ring of suffering",
    "b ring": "berserker ring",
    "bring": "berserker ring",
    "bringi": "berserker ring (i)",
    "row": "ring of wealth",
    "rol": "ring of life",
    "rotg": "ring of the gods",
    "tyrannical": "tyrannical ring",
    "treasonous": "treasonous ring",
    "seers ring": "seers ring",
    "archers ring": "archers ring",
    "ultor": "ultor ring",
    "bellator": "bellator ring",
    "magus": "magus ring",
    "venator ring": "venator ring",
    # === ARMOR SETS (prefixes) ===
    "arma": "armadyl",
    "bandos": "bandos",
    "torva": "torva",
    "ancestral": "ancestral",
    "inq": "inquisitor",
    "inquis": "inquisitor",
    "justi": "justiciar",
    "justiciar": "justiciar",
    "dh": "dharok",
    "dharok": "dharok",
    "karils": "karil",
    "ahrims": "ahrim",
    "guthans": "guthan",
    "veracs": "verac",
    "torags": "torag",
    "void": "void knight",
    "evoid": "elite void",
    "prossy": "proselyte",
    "dhide": "dragonhide",
    "d hide": "dragonhide",
    "black dhide": "black dragonhide",
    "crystal armour": "crystal armour",
    "masori": "masori",
    # === PRAYER SCROLLS ===
    "dex": "dexterous prayer scroll",
    "dexterous": "dexterous prayer scroll",
    "augury": "arcane prayer scroll",
    "rigour": "dexterous prayer scroll",
    # === POTIONS ===
    "ppot": "prayer potion",
    "pray pot": "prayer potion",
    "super restore": "super restore",
    "restore": "super restore",
    "scb": "super combat potion",
    "super combat": "super combat potion",
    "ranging pot": "ranging potion",
    "range pot": "ranging potion",
    "divine scb": "divine super combat potion",
    "divine ranging": "divine ranging potion",
    "sara brew": "saradomin brew",
    "brew": "saradomin brew",
    "sanfew": "sanfew serum",
    "stam": "stamina potion",
    "stamina": "stamina potion",
    "antivenom": "anti-venom",
    "antifire": "antifire potion",
    "super antifire": "super antifire potion",
    "extended antifire": "extended antifire",
    "extended super antifire": "extended super antifire",
    # === OTHER VALUABLE ITEMS ===
    "zenny": "zenyte",
    "zenyte": "zenyte",
    "dpick": "dragon pickaxe",
    "d pick": "dragon pickaxe",
    "daxe": "dragon axe",
    "d axe": "dragon axe",
    "dharp": "dragon harpoon",
    "d harp": "dragon harpoon",
    "tome": "tome of fire",
    "tof": "tome of fire",
    "tome of water": "tome of water",
    "lightbearer": "lightbearer",
    "pegcrystal": "pegasian crystal",
    "primcrystal": "primordial crystal",
    "eterncrystal": "eternal crystal",
    "smouldering": "smouldering stone",
    # === RAIDS ITEMS ===
    "tob": "theatre of blood",
    "cox": "chambers of xeric",
    "toa": "tombs of amascut",
    "dex scroll": "dexterous prayer scroll",
    "arcane scroll": "arcane prayer scroll",
    "ancestral hat": "ancestral hat",
    "ancestral top": "ancestral robe top",
    "ancestral bottom": "ancestral robe bottom",
    "masori mask": "masori mask",
    "masori body": "masori body",
    "masori chaps": "masori chaps",
    "torva helm": "torva full helm",
    "torva body": "torva platebody",
    "torva legs": "torva platelegs",
}


def expand_acronym(query: str) -> str:
    """Expand OSRS acronym to full item name if recognized.

    Args:
        query: Search query (may be an acronym)

    Returns:
        Expanded item name if acronym recognized, otherwise original query
    """
    return OSRS_ACRONYMS.get(query.lower().strip(), query)


class PredictionLoader:
    """Fetches pre-computed predictions from the Ampere server's predictions table.

    The inference engine runs every 5 minutes on the Ampere server, storing
    predictions in the `predictions` table. This loader queries those predictions
    and filters them based on user constraints.
    """

    def __init__(self, db_connection_string: str, pool_size: int = 5, preferred_model_id: str = ""):
        """Initialize database connection.

        Args:
            db_connection_string: PostgreSQL connection string
            pool_size: Connection pool size
            preferred_model_id: If set, only serve predictions from this model_id
        """
        self.engine = create_engine(
            db_connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=3,
            pool_pre_ping=True,
        )
        self.preferred_model_id = preferred_model_id
        if preferred_model_id:
            logger.info(f"Model filter active: preferred_model_id={preferred_model_id}")

    def _max_time_subquery(self) -> str:
        """Return the MAX(time) subquery, scoped to preferred model if configured."""
        if self.preferred_model_id:
            return "(SELECT MAX(time) FROM predictions WHERE model_id = :preferred_model_id)"
        return "(SELECT MAX(time) FROM predictions)"

    def _inject_model_params(self, params: dict) -> dict:
        """Add preferred_model_id to params dict if configured."""
        if self.preferred_model_id:
            params["preferred_model_id"] = self.preferred_model_id
        return params

    def get_latest_predictions(
        self,
        min_fill_prob: float = 0.03,
        min_ev: float = 0.005,
        max_hour_offset: Optional[int] = None,
        item_ids: Optional[list[int]] = None,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Fetch latest predictions filtered by constraints.

        Args:
            min_fill_prob: Minimum fill probability (filter out unlikely fills)
            min_ev: Minimum expected value threshold
            max_hour_offset: Maximum hours ahead (for active vs passive trading)
            item_ids: Optional list of specific item IDs to fetch
            limit: Maximum number of predictions to return

        Returns:
            DataFrame with predictions sorted by expected_value descending
        """
        # Build query with filters
        conditions = [
            f"time = {self._max_time_subquery()}",
            "fill_probability >= :min_fill_prob",
            "expected_value >= :min_ev",
        ]

        params = {
            "min_fill_prob": min_fill_prob,
            "min_ev": min_ev,
            "limit": limit,
        }
        self._inject_model_params(params)

        if max_hour_offset:
            conditions.append("hour_offset <= :max_hour_offset")
            params["max_hour_offset"] = max_hour_offset

        if item_ids:
            conditions.append("item_id = ANY(:item_ids)")
            params["item_ids"] = item_ids

        where_clause = " AND ".join(conditions)

        query = text(
            f"""
            SELECT
                item_id,
                item_name,
                hour_offset,
                offset_pct,
                fill_probability,
                expected_value,
                buy_price,
                sell_price,
                current_high,
                current_low,
                confidence,
                time as prediction_time
            FROM predictions
            WHERE {where_clause}
            ORDER BY expected_value DESC
            LIMIT :limit
        """
        )

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params=params)

            if df.empty:
                logger.warning("No predictions found matching criteria")
                return pd.DataFrame()

            return df

        except Exception as e:
            logger.error(f"Error fetching predictions: {e}")
            return pd.DataFrame()

    def get_predictions_for_item(self, item_id: int) -> pd.DataFrame:
        """Get all latest predictions for a specific item.

        Args:
            item_id: OSRS item ID

        Returns:
            DataFrame with all hour_offset/offset_pct combinations for the item
        """
        query = text(
            f"""
            SELECT
                item_id,
                item_name,
                hour_offset,
                offset_pct,
                fill_probability,
                expected_value,
                buy_price,
                sell_price,
                current_high,
                current_low,
                confidence,
                time as prediction_time
            FROM predictions
            WHERE time = {self._max_time_subquery()}
              AND item_id = :item_id
            ORDER BY hour_offset, offset_pct
        """
        )

        try:
            params = self._inject_model_params({"item_id": item_id})
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Error fetching predictions for item {item_id}: {e}")
            return pd.DataFrame()

    def get_best_prediction_per_item(
        self,
        min_fill_prob: float = 0.03,
        min_ev: float = 0.005,
        min_hour_offset: Optional[int] = None,
        max_hour_offset: Optional[int] = None,
        min_offset_pct: Optional[float] = None,
        max_offset_pct: Optional[float] = None,
        limit: int = 100,
        min_volume_24h: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get the single best prediction for each item.

        Selects the hour_offset/offset_pct combination with highest EV per item.

        Args:
            min_fill_prob: Minimum fill probability
            min_ev: Minimum expected value
            min_hour_offset: Minimum hours ahead (for filtering short horizons)
            max_hour_offset: Maximum hours ahead
            min_offset_pct: Minimum offset percentage (e.g., 0.0125 for 1.25%)
            max_offset_pct: Maximum offset percentage (e.g., 0.0250 for 2.5%)
            limit: Maximum number of items
            min_volume_24h: Minimum 24-hour volume (excludes illiquid items)

        Returns:
            DataFrame with one row per item (best configuration)
        """
        # Build hour filter for both min and max (using parameterized queries)
        hour_filter = ""
        if min_hour_offset:
            hour_filter += " AND p.hour_offset >= :min_hour_offset"
        if max_hour_offset:
            hour_filter += " AND p.hour_offset <= :max_hour_offset"

        # Build offset filter
        offset_filter = ""
        if min_offset_pct is not None:
            offset_filter += " AND p.offset_pct >= :min_offset_pct"
        if max_offset_pct is not None:
            offset_filter += " AND p.offset_pct <= :max_offset_pct"

        # Build volume filter with price-tiered thresholds
        # High-value items naturally have lower volume, so we apply carve-outs:
        # - >100M items: min 100 volume (Tbow, Scythe, Torva, etc.)
        # - >10M items: min 500 volume
        # - >1M items: min volume capped at 2000
        # - Other items: use configured min_volume_24h
        volume_filter = ""
        if min_volume_24h is not None and min_volume_24h > 0:
            volume_filter = """
                AND COALESCE(v.total_volume, 0) >= CASE
                    WHEN COALESCE(p.buy_price, 0) > 100000000 THEN 100
                    WHEN COALESCE(p.buy_price, 0) > 10000000 THEN 500
                    WHEN COALESCE(p.buy_price, 0) > 1000000 THEN LEAST(:min_volume_24h, 2000)
                    ELSE :min_volume_24h
                END"""

        # Conditionally build query - skip volume CTE if not needed (slow 24h aggregation)
        needs_volume = min_volume_24h is not None and min_volume_24h > 0

        if needs_volume:
            query = text(
                f"""
                WITH volume_24h AS (
                    SELECT item_id,
                           COALESCE(SUM(high_price_volume), 0)
                           + COALESCE(SUM(low_price_volume), 0) as total_volume
                    FROM price_data_5min
                    WHERE timestamp >= NOW() - INTERVAL '24 hours'
                    GROUP BY item_id
                ),
                ranked AS (
                    SELECT
                        p.item_id,
                        p.item_name,
                        p.hour_offset,
                        p.offset_pct,
                        p.fill_probability,
                        p.expected_value,
                        p.buy_price,
                        p.sell_price,
                        p.current_high,
                        p.current_low,
                        p.confidence,
                        p.time as prediction_time,
                        COALESCE(v.total_volume, 0) as volume_24h,
                        ROW_NUMBER() OVER (
                            PARTITION BY p.item_id ORDER BY p.expected_value DESC
                        ) as rn
                    FROM predictions p
                    LEFT JOIN volume_24h v ON v.item_id = p.item_id
                    WHERE p.time = {self._max_time_subquery()}
                      AND p.fill_probability >= :min_fill_prob
                      AND p.expected_value >= :min_ev
                      {hour_filter}
                      {offset_filter}
                      {volume_filter}
                )
                SELECT *
                FROM ranked
                WHERE rn = 1
                ORDER BY expected_value DESC
                LIMIT :limit
            """
            )
        else:
            # Fast path: skip volume calculation entirely
            query = text(
                f"""
                WITH ranked AS (
                    SELECT
                        p.item_id,
                        p.item_name,
                        p.hour_offset,
                        p.offset_pct,
                        p.fill_probability,
                        p.expected_value,
                        p.buy_price,
                        p.sell_price,
                        p.current_high,
                        p.current_low,
                        p.confidence,
                        p.time as prediction_time,
                        0 as volume_24h,
                        ROW_NUMBER() OVER (
                            PARTITION BY p.item_id ORDER BY p.expected_value DESC
                        ) as rn
                    FROM predictions p
                    WHERE p.time = {self._max_time_subquery()}
                      AND p.fill_probability >= :min_fill_prob
                      AND p.expected_value >= :min_ev
                      {hour_filter}
                      {offset_filter}
                )
                SELECT *
                FROM ranked
                WHERE rn = 1
                ORDER BY expected_value DESC
                LIMIT :limit
            """
            )

        params = {
            "min_fill_prob": min_fill_prob,
            "min_ev": min_ev,
            "limit": limit,
        }
        if min_hour_offset:
            params["min_hour_offset"] = int(min_hour_offset)
        if max_hour_offset:
            params["max_hour_offset"] = int(max_hour_offset)
        if min_offset_pct is not None:
            params["min_offset_pct"] = min_offset_pct
        if max_offset_pct is not None:
            params["max_offset_pct"] = max_offset_pct
        if min_volume_24h is not None and min_volume_24h > 0:
            params["min_volume_24h"] = min_volume_24h
        self._inject_model_params(params)

        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params=params)

            # Drop internal columns (keep volume_24h for stability filter)
            for col in ["rn"]:
                if col in df.columns:
                    df = df.drop(col, axis=1)

            return df

        except Exception as e:
            logger.error(f"Error fetching best predictions per item: {e}")
            return pd.DataFrame()

    def get_latest_timestamp(self) -> Optional[datetime]:
        """Get the timestamp of the most recent predictions.

        Returns:
            Most recent prediction timestamp, or None if no data
        """
        if self.preferred_model_id:
            query = text("SELECT MAX(time) FROM predictions WHERE model_id = :preferred_model_id")
            params = {"preferred_model_id": self.preferred_model_id}
        else:
            query = text("SELECT MAX(time) FROM predictions")
            params = {}

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, params).fetchone()

            if result and result[0]:
                return result[0]
            return None

        except Exception as e:
            logger.error(f"Error fetching latest timestamp: {e}")
            return None

    def get_prediction_age_seconds(self) -> float:
        """Get age of the latest predictions in seconds.

        Returns:
            Age in seconds, or infinity if no predictions
        """
        latest = self.get_latest_timestamp()
        if latest is None:
            return float("inf")

        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)

        return (datetime.now(timezone.utc) - latest).total_seconds()

    def get_item_buy_limit(self, item_id: int) -> Optional[int]:
        """Get GE buy limit for an item.

        Tries OSRS Wiki API first (authoritative source), falls back to database.

        Args:
            item_id: OSRS item ID

        Returns:
            Buy limit or None if not found
        """
        # Try Wiki API first (authoritative source)
        wiki_limit = self._get_wiki_buy_limit(item_id)
        if wiki_limit is not None:
            return wiki_limit

        # Fallback to database
        db_limit = self._get_db_buy_limit(item_id)
        if db_limit is not None:
            logger.debug(f"Using database buy limit for item {item_id}: {db_limit}")
            return db_limit

        return None

    def _get_db_buy_limit(self, item_id: int) -> Optional[int]:
        """Query database for item buy limit.

        Args:
            item_id: OSRS item ID

        Returns:
            Buy limit from database or None if not found
        """
        query = text(
            """
            SELECT buy_limit
            FROM items
            WHERE item_id = :item_id
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"item_id": item_id}).fetchone()

            if result and result[0]:
                return int(result[0])
            return None

        except Exception as e:
            logger.debug(f"Could not fetch buy limit from DB for item {item_id}: {e}")
            return None

    def _get_wiki_buy_limit(self, item_id: int) -> Optional[int]:
        """Get buy limit from OSRS Wiki API.

        Args:
            item_id: OSRS item ID

        Returns:
            Buy limit from Wiki API or None if not available
        """
        from .config import Config
        from .wiki_api import get_wiki_api_client

        config = Config()
        if not config.wiki_api_enabled:
            return None

        try:
            wiki_client = get_wiki_api_client()
            return wiki_client.get_buy_limit(item_id)
        except Exception as e:
            logger.debug(
                f"Could not fetch buy limit from Wiki API for item {item_id}: {e}"
            )
            return None

    def get_batch_buy_limits(self, item_ids: list[int]) -> dict[int, int]:
        """Get GE buy limits for multiple items in batch.

        Fetches buy limits from Wiki API first (batched), then fills in
        remaining items from database in a single query.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            Dict mapping item_id to buy_limit (only includes items with known limits)
        """
        if not item_ids:
            return {}

        result: dict[int, int] = {}
        remaining_ids = set(item_ids)

        # Try Wiki API first (it has a cache, so this is efficient)
        result, remaining_ids = self._get_batch_wiki_buy_limits(list(remaining_ids))

        # Fetch remaining from database in one query
        if remaining_ids:
            db_limits = self._get_batch_db_buy_limits(list(remaining_ids))
            result.update(db_limits)

        return result

    def _get_batch_db_buy_limits(self, item_ids: list[int]) -> dict[int, int]:
        """Query database for buy limits in batch.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            Dict mapping item_id to buy_limit
        """
        if not item_ids:
            return {}

        # Use ANY array for efficient IN clause
        query = text(
            """
            SELECT item_id, buy_limit
            FROM items
            WHERE item_id = ANY(:item_ids)
              AND buy_limit IS NOT NULL
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"item_ids": item_ids}).fetchall()

            return {int(row[0]): int(row[1]) for row in result if row[1]}

        except Exception as e:
            logger.debug(f"Could not fetch batch buy limits from DB: {e}")
            return {}

    def _get_batch_wiki_buy_limits(
        self, item_ids: list[int]
    ) -> tuple[dict[int, int], set[int]]:
        """Get buy limits from Wiki API for multiple items.

        Uses the wiki client's cache for efficiency.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            Tuple of (dict mapping item_id to buy_limit, set of unfound item_ids)
        """
        from .config import Config
        from .wiki_api import get_wiki_api_client

        config = Config()
        if not config.wiki_api_enabled:
            return {}, set(item_ids)

        result: dict[int, int] = {}
        remaining: set[int] = set()

        try:
            wiki_client = get_wiki_api_client()
            for item_id in item_ids:
                limit = wiki_client.get_buy_limit(item_id)
                if limit is not None:
                    result[item_id] = limit
                else:
                    remaining.add(item_id)
        except Exception as e:
            logger.debug(f"Could not fetch batch buy limits from Wiki API: {e}")
            remaining = set(item_ids) - set(result.keys())

        return result, remaining

    def get_item_volume_24h(self, item_id: int) -> Optional[int]:
        """Get 24-hour trade volume for an item.

        Sums buy and sell volumes from the last 24 hours of 5-minute data.

        Args:
            item_id: OSRS item ID

        Returns:
            Total 24h volume (0 if no data), or None on query failure
        """
        query = text(
            """
            SELECT COALESCE(SUM(high_price_volume), 0)
                   + COALESCE(SUM(low_price_volume), 0) as total_volume
            FROM price_data_5min
            WHERE item_id = :item_id
              AND timestamp >= NOW() - INTERVAL '24 hours'
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"item_id": item_id}).fetchone()

            if result and result[0] is not None:
                return int(result[0])
            return 0

        except Exception as e:
            logger.debug(f"Could not fetch 24h volume for item {item_id}: {e}")
            return None

    def get_item_volume_1h(self, item_id: int) -> Optional[int]:
        """Get last hour of trade volume from 5-minute data.

        Args:
            item_id: OSRS item ID

        Returns:
            Total 1h volume (0 if no data), or None on query failure
        """
        query = text(
            """
            SELECT COALESCE(SUM(high_price_volume), 0)
                   + COALESCE(SUM(low_price_volume), 0) as total_volume
            FROM price_data_5min
            WHERE item_id = :item_id
              AND timestamp >= NOW() - INTERVAL '1 hour'
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"item_id": item_id}).fetchone()

            if result and result[0] is not None:
                return int(result[0])
            return 0

        except Exception as e:
            logger.debug(f"Could not fetch 1h volume for item {item_id}: {e}")
            return None

    def get_batch_volumes_24h(self, item_ids: list[int]) -> dict[int, int]:
        """Get 24-hour trade volumes for multiple items in a single query.

        Optimized batch fetch to eliminate N+1 queries.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            Dict mapping item_id to total 24h volume
        """
        if not item_ids:
            return {}

        query = text(
            """
            SELECT item_id,
                   COALESCE(SUM(high_price_volume), 0)
                   + COALESCE(SUM(low_price_volume), 0) as total_volume
            FROM price_data_5min
            WHERE item_id = ANY(:item_ids)
              AND timestamp >= NOW() - INTERVAL '24 hours'
            GROUP BY item_id
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"item_ids": item_ids}).fetchall()

            return {int(row[0]): int(row[1]) for row in result}

        except Exception as e:
            logger.warning(f"Could not fetch batch 24h volumes: {e}")
            return {}

    def get_batch_volumes_1h(self, item_ids: list[int]) -> dict[int, int]:
        """Get last hour trade volumes for multiple items in a single query.

        Optimized batch fetch to eliminate N+1 queries.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            Dict mapping item_id to total 1h volume
        """
        if not item_ids:
            return {}

        query = text(
            """
            SELECT item_id,
                   COALESCE(SUM(high_price_volume), 0)
                   + COALESCE(SUM(low_price_volume), 0) as total_volume
            FROM price_data_5min
            WHERE item_id = ANY(:item_ids)
              AND timestamp >= NOW() - INTERVAL '1 hour'
            GROUP BY item_id
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"item_ids": item_ids}).fetchall()

            return {int(row[0]): int(row[1]) for row in result}

        except Exception as e:
            logger.debug(f"Could not fetch batch 1h volumes: {e}")
            return {}

    def get_price_history(self, item_id: int, hours: int = 24) -> list[dict]:
        """Get price history for an item.

        Returns hourly price data for mini chart display.

        Args:
            item_id: OSRS item ID
            hours: Number of hours of history (default 24)

        Returns:
            List of dicts with 'timestamp' and 'price' (midpoint) keys
        """
        query = text(
            """
            SELECT timestamp, avg_high_price, avg_low_price
            FROM price_data_5min
            WHERE item_id = :item_id
              AND timestamp >= NOW() - make_interval(hours => :hours)
            ORDER BY timestamp ASC
            LIMIT :limit
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    query, {"item_id": item_id, "hours": hours, "limit": hours}
                ).fetchall()

            history = []
            for row in result:
                timestamp, high, low = row
                if high is not None and low is not None:
                    midpoint = (high + low) // 2
                    history.append(
                        {
                            "timestamp": (
                                timestamp.isoformat()
                                if hasattr(timestamp, "isoformat")
                                else str(timestamp)
                            ),
                            "price": midpoint,
                        }
                    )

            return history

        except Exception as e:
            logger.debug(f"Could not fetch price history for item {item_id}: {e}")
            return []

    def get_extended_price_history(self, item_id: int, hours: int = 24) -> list[dict]:
        """Get extended price history with high/low data for sparkline charts.

        Returns hourly price data with all price points for web frontend.

        Args:
            item_id: OSRS item ID
            hours: Number of hours of history (default 24)

        Returns:
            List of dicts with 'timestamp', 'high', 'low', 'avgHigh', 'avgLow' keys
        """
        query = text(
            """
            SELECT timestamp, high_price, low_price, avg_high_price, avg_low_price
            FROM price_data_5min
            WHERE item_id = :item_id
              AND timestamp >= NOW() - make_interval(hours => :hours)
            ORDER BY timestamp ASC
            LIMIT :limit
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    query, {"item_id": item_id, "hours": hours, "limit": hours}
                ).fetchall()

            history = []
            for row in result:
                timestamp, high, low, avg_high, avg_low = row
                # Use available price data, falling back as needed
                actual_high = high if high is not None else avg_high
                actual_low = low if low is not None else avg_low
                actual_avg_high = avg_high if avg_high is not None else high
                actual_avg_low = avg_low if avg_low is not None else low

                if actual_high is not None and actual_low is not None:
                    history.append(
                        {
                            "timestamp": (
                                timestamp.isoformat()
                                if hasattr(timestamp, "isoformat")
                                else str(timestamp)
                            ),
                            "high": int(actual_high),
                            "low": int(actual_low),
                            "avgHigh": (
                                int(actual_avg_high)
                                if actual_avg_high
                                else int(actual_high)
                            ),
                            "avgLow": (
                                int(actual_avg_low)
                                if actual_avg_low
                                else int(actual_low)
                            ),
                        }
                    )

            return history

        except Exception as e:
            logger.debug(
                f"Could not fetch extended price history for item {item_id}: {e}"
            )
            return []

    def get_item_trend(self, item_id: int) -> str:
        """Determine price trend for an item over last 4 hours.

        Args:
            item_id: OSRS item ID

        Returns:
            'Rising', 'Falling', or 'Stable'
        """
        query = text(
            """
            SELECT avg_high_price, avg_low_price
            FROM price_data_5min
            WHERE item_id = :item_id
            ORDER BY timestamp DESC
            LIMIT 4
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"item_id": item_id}).fetchall()

            if len(result) < 2:
                return "Stable"

            # Compare current vs 4h ago using midpoint
            current_mid = (
                (result[0][0] + result[0][1]) / 2
                if result[0][0] and result[0][1]
                else None
            )
            oldest_mid = (
                (result[-1][0] + result[-1][1]) / 2
                if result[-1][0] and result[-1][1]
                else None
            )

            if current_mid is None or oldest_mid is None or oldest_mid == 0:
                return "Stable"

            change_pct = (current_mid - oldest_mid) / oldest_mid

            if change_pct > 0.02:  # > 2% increase
                return "Rising"
            elif change_pct < -0.02:  # > 2% decrease
                return "Falling"
            else:
                return "Stable"

        except Exception as e:
            logger.debug(f"Could not fetch trend for item {item_id}: {e}")
            return "Stable"

    def get_batch_trends(self, item_ids: list[int]) -> dict[int, str]:
        """Determine price trends for multiple items in a single query.

        Fetches the last 4 hours of price data for all items and calculates
        trends based on the price change percentage.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            Dict mapping item_id to trend ('Rising', 'Falling', or 'Stable')
        """
        if not item_ids:
            return {}

        # Fetch last 4 hours of prices for all items at once
        query = text(
            """
            WITH ranked_prices AS (
                SELECT
                    item_id,
                    avg_high_price,
                    avg_low_price,
                    timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY item_id ORDER BY timestamp DESC
                    ) as rn
                FROM price_data_5min
                WHERE item_id = ANY(:item_ids)
                  AND timestamp >= NOW() - INTERVAL '4 hours'
            )
            SELECT item_id, avg_high_price, avg_low_price, rn
            FROM ranked_prices
            WHERE rn IN (1, 4)
            ORDER BY item_id, rn
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"item_ids": item_ids}).fetchall()

            # Group by item_id and calculate trends
            item_prices: dict[int, dict] = {}
            for row in result:
                item_id, high, low, rn = row
                item_id = int(item_id)
                if item_id not in item_prices:
                    item_prices[item_id] = {}

                if rn == 1:  # Current (most recent)
                    item_prices[item_id]["current"] = (high, low)
                elif rn == 4:  # 4 hours ago
                    item_prices[item_id]["oldest"] = (high, low)

            # Calculate trends
            trends: dict[int, str] = {}
            for item_id, prices in item_prices.items():
                current = prices.get("current")
                oldest = prices.get("oldest")

                # Need both prices to calculate trend
                if not current or not oldest:
                    trends[item_id] = "Stable"
                    continue

                current_mid = (
                    (current[0] + current[1]) / 2 if current[0] and current[1] else None
                )
                oldest_mid = (
                    (oldest[0] + oldest[1]) / 2 if oldest[0] and oldest[1] else None
                )

                if current_mid is None or oldest_mid is None or oldest_mid == 0:
                    trends[item_id] = "Stable"
                    continue

                change_pct = (current_mid - oldest_mid) / oldest_mid

                if change_pct > 0.02:  # > 2% increase
                    trends[item_id] = "Rising"
                elif change_pct < -0.02:  # > 2% decrease
                    trends[item_id] = "Falling"
                else:
                    trends[item_id] = "Stable"

            # Default to Stable for items with no data
            for item_id in item_ids:
                if item_id not in trends:
                    trends[item_id] = "Stable"

            return trends

        except Exception as e:
            logger.warning(f"Could not fetch batch trends: {e}")
            # Return Stable for all items on error
            return {item_id: "Stable" for item_id in item_ids}

    def health_check(self) -> dict:
        """Check database connection and prediction freshness.

        Returns:
            Health status dictionary
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            age = self.get_prediction_age_seconds()

            if age == float("inf"):
                status = "error"
                message = "No predictions found"
            elif age > 360:  # > 6 min = missed a 5-min cycle
                status = "warning"
                message = f"Predictions are {age:.0f}s old (stale)"
            else:
                status = "ok"
                message = f"Predictions are {age:.0f}s old"

            return {
                "status": status,
                "component": "prediction_loader",
                "message": message,
                "prediction_age_seconds": age if age != float("inf") else None,
                "connected": True,
            }

        except Exception as e:
            return {
                "status": "error",
                "component": "prediction_loader",
                "error": str(e),
                "connected": False,
            }

    def search_items_by_name(self, query: str, limit: int = 10) -> list[dict]:
        """Search for items by name with fuzzy matching.

        Supports OSRS acronym expansion (e.g., "ags" -> "armadyl godsword").

        Args:
            query: Search query (may be an acronym)
            limit: Maximum number of results

        Returns:
            List of dicts with item_id and item_name
        """
        # Expand acronym if recognized
        expanded_query = expand_acronym(query)

        sql = text(
            f"""
            SELECT item_id, item_name
            FROM (
                SELECT DISTINCT item_id, item_name
                FROM predictions
                WHERE time = {self._max_time_subquery()}
                  AND LOWER(item_name) LIKE LOWER(:query)
            ) AS items
            ORDER BY
                CASE
                    WHEN LOWER(item_name) = LOWER(:exact) THEN 0
                    WHEN LOWER(item_name) LIKE LOWER(:starts) THEN 1
                    ELSE 2
                END,
                item_name
            LIMIT :limit
        """
        )

        params = self._inject_model_params({
            "query": f"%{expanded_query}%",
            "exact": expanded_query,
            "starts": f"{expanded_query}%",
            "limit": limit,
        })
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sql, params)
                return [{"item_id": row[0], "item_name": row[1]} for row in result]
        except Exception as e:
            logger.error(f"Error searching items: {e}")
            return []

    def close(self):
        """Close database connections."""
        self.engine.dispose()

    # === Model Registry Methods ===

    def get_active_model_ids(self) -> set[int]:
        """Get model_ids with ACTIVE status.

        Returns:
            Set of model_ids that are currently active
        """
        query = text(
            """
            SELECT model_id
            FROM model_registry
            WHERE status = 'ACTIVE'
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()
            return {row[0] for row in result}
        except Exception as e:
            logger.error(f"Error fetching active model IDs: {e}")
            return set()

    def get_model_status(self, model_id: int) -> Optional[str]:
        """Get status for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Model status string or None if not found
        """
        query = text(
            """
            SELECT status
            FROM model_registry
            WHERE model_id = :model_id
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"model_id": model_id}).fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error fetching model status for {model_id}: {e}")
            return None

    def get_models_by_item(
        self,
        item_id: int,
        status_filter: Optional[str] = None,
    ) -> list[dict]:
        """Get models for an item, optionally filtered by status.

        Args:
            item_id: OSRS item ID
            status_filter: Optional status to filter by (e.g., 'ACTIVE')

        Returns:
            List of model dicts with model_id, status, mean_auc, trained_at
        """
        if status_filter:
            query = text(
                """
                SELECT model_id, item_id, status, mean_auc, trained_at
                FROM model_registry
                WHERE item_id = :item_id AND status = :status
                ORDER BY trained_at DESC
            """
            )
            params = {"item_id": item_id, "status": status_filter}
        else:
            query = text(
                """
                SELECT model_id, item_id, status, mean_auc, trained_at
                FROM model_registry
                WHERE item_id = :item_id
                ORDER BY trained_at DESC
            """
            )
            params = {"item_id": item_id}

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, params).fetchall()
            return [
                {
                    "model_id": row[0],
                    "item_id": row[1],
                    "status": row[2],
                    "mean_auc": float(row[3]) if row[3] else None,
                    "trained_at": row[4].isoformat() if row[4] else None,
                }
                for row in result
            ]
        except Exception as e:
            logger.error(f"Error fetching models for item {item_id}: {e}")
            return []

    def get_active_model_for_item(self, item_id: int) -> Optional[dict]:
        """Get the active model for an item.

        Args:
            item_id: OSRS item ID

        Returns:
            Model dict or None if no active model exists
        """
        models = self.get_models_by_item(item_id, status_filter="ACTIVE")
        return models[0] if models else None

    def get_model_registry_stats(self) -> dict:
        """Get statistics about model registry status.

        Returns:
            Dict with counts by status
        """
        query = text(
            """
            SELECT status, COUNT(*) as count
            FROM model_registry
            GROUP BY status
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()
            stats = {row[0]: row[1] for row in result}
            return {
                "active": stats.get("ACTIVE", 0),
                "deprecated": stats.get("DEPRECATED", 0),
                "sunset": stats.get("SUNSET", 0),
                "archived": stats.get("ARCHIVED", 0),
                "total": sum(stats.values()),
            }
        except Exception as e:
            logger.error(f"Error fetching model registry stats: {e}")
            return {
                "active": 0,
                "deprecated": 0,
                "sunset": 0,
                "archived": 0,
                "total": 0,
            }

    def get_items_with_active_models(self) -> set[int]:
        """Get item IDs that have at least one ACTIVE model.

        Returns:
            Set of item_ids with active models
        """
        query = text(
            """
            SELECT DISTINCT item_id
            FROM model_registry
            WHERE status = 'ACTIVE'
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query).fetchall()
            return {row[0] for row in result}
        except Exception as e:
            logger.error(f"Error fetching items with active models: {e}")
            return set()

    def get_latest_price(self, item_id: int) -> Optional[dict]:
        """Get most recent price tick for an item from 1-minute data.

        Args:
            item_id: OSRS item ID

        Returns:
            Dictionary with keys: timestamp, high, low, high_time, low_time
            Returns None if no data found
        """
        query = text(
            """
            SELECT
                timestamp,
                high,
                low,
                high_time,
                low_time
            FROM prices_latest_1m
            WHERE item_id = :item_id
            ORDER BY timestamp DESC
            LIMIT 1
        """
        )

        try:
            with self.engine.connect() as conn:
                result = conn.execute(query, {"item_id": item_id}).fetchone()

            if result is None:
                return None

            return {
                "timestamp": result[0],
                "high": result[1],
                "low": result[2],
                "high_time": result[3],
                "low_time": result[4],
            }

        except Exception as e:
            logger.error(f"Error fetching latest price for item {item_id}: {e}")
            return None

    def get_predictions_for_items(self, item_ids: list[int]) -> pd.DataFrame:
        """Get all latest predictions for multiple items.

        Args:
            item_ids: List of OSRS item IDs

        Returns:
            DataFrame with all predictions for the items
            (all hour_offset/offset_pct combinations)
        """
        if not item_ids:
            return pd.DataFrame()

        query = text(
            f"""
            SELECT
                item_id,
                item_name,
                hour_offset,
                offset_pct,
                fill_probability,
                expected_value,
                buy_price,
                sell_price,
                current_high,
                current_low,
                confidence,
                time as prediction_time
            FROM predictions
            WHERE time = {self._max_time_subquery()}
              AND item_id = ANY(:item_ids)
            ORDER BY item_id, hour_offset, offset_pct
        """
        )

        params = self._inject_model_params({"item_ids": item_ids})
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Error fetching predictions for items: {e}")
            return pd.DataFrame()
