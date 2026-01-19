#!/usr/bin/env python3
"""
GePT Trainer - Opportunistic, Validation-Driven Training Service

This service runs on Hydra and handles all model training. It:
1. Connects to Ampere DB via Tailscale
2. Pushes any pending models (from pending_push.json)
3. Queries validation data to find items needing retraining
4. Trains items in priority order
5. Pushes models to Ampere after each successful training
6. Sleeps when queue is empty

Resilience:
- If Ampere is unreachable, retries every 5 minutes
- If training fails mid-way, models queue to pending_push.json
- Graceful shutdown: finishes current model, then stops

Usage:
    python -m src.gept_trainer

Environment:
    DB_CONNECTION_STRING  - Connection to Ampere PostgreSQL
    AMPERE_HOST          - Ampere SSH host for model push
    GEPT_DIR             - Base directory (default: /home/ubuntu/gept)
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import psycopg2

# Configuration
GEPT_DIR = Path(os.getenv("GEPT_DIR", "/home/ubuntu/gept"))
MODELS_DIR = GEPT_DIR / "models"
PENDING_PUSH_FILE = GEPT_DIR / "pending_push.json"
AMPERE_HOST = os.getenv("AMPERE_HOST", "ubuntu@150.136.170.128")
AMPERE_SSH_KEY = os.getenv("AMPERE_SSH_KEY", str(GEPT_DIR / ".secrets" / "oracle_key.pem"))
AMPERE_MODELS_DIR = "/home/ubuntu/gept/models"

# Training settings
MAX_ITEMS_PER_CYCLE = 50  # Train up to 50 items before checking for updates
IDLE_SLEEP_HOURS = 1  # Sleep when no items need training
CONNECTION_RETRY_MINUTES = 5  # Retry DB connection every 5 min
STALE_MODEL_DAYS = 30  # Models older than this are candidates for retraining
HIGH_MISS_THRESHOLD = 0.4  # Miss rate above this triggers priority retraining

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(GEPT_DIR / "logs" / "trainer.log"),
    ],
)
logger = logging.getLogger("gept-trainer")

# Graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    global shutdown_requested
    logger.info(f"Received signal {signum}, will shutdown after current task")
    shutdown_requested = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def ensure_db_tunnel() -> bool:
    """Ensure SSH tunnel to Ampere database is available."""
    import socket

    # Check if port 5432 is already available
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 5432))
    sock.close()

    if result == 0:
        logger.debug("DB tunnel already available on localhost:5432")
        return True

    logger.info("DB tunnel not available, establishing SSH tunnel...")

    try:
        # Establish SSH tunnel in background
        cmd = [
            "ssh", "-f", "-N",
            "-i", AMPERE_SSH_KEY,
            "-L", "5432:localhost:5432",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=60",
            AMPERE_HOST
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)

        # Wait for tunnel to be ready
        time.sleep(2)

        # Verify tunnel
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 5432))
        sock.close()

        if result == 0:
            logger.info("SSH tunnel established successfully")
            return True
        else:
            logger.error("SSH tunnel created but port not reachable")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Timeout establishing SSH tunnel")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to establish SSH tunnel: {e.stderr.decode() if e.stderr else e}")
        return False
    except Exception as e:
        logger.error(f"Error establishing SSH tunnel: {e}")
        return False


@dataclass
class TrainingPriority:
    """Item training priority based on validation data."""
    item_id: int
    item_name: str
    priority: str  # 'high_miss', 'no_model', 'stale', 'healthy'
    miss_rate: Optional[float] = None
    model_age_days: Optional[int] = None
    last_trained: Optional[datetime] = None


def get_db_connection() -> Optional[psycopg2.extensions.connection]:
    """Connect to Ampere PostgreSQL via Tailscale."""
    conn_string = os.getenv("DB_CONNECTION_STRING")
    if not conn_string:
        # Fallback to individual params
        conn_string = (
            f"postgresql://{os.getenv('DB_USER', 'osrs_user')}:"
            f"{os.getenv('DB_PASS')}@"
            f"{os.getenv('DB_HOST', 'localhost')}:"
            f"{os.getenv('DB_PORT', '5432')}/"
            f"{os.getenv('DB_NAME', 'osrs_data')}"
        )

    try:
        conn = psycopg2.connect(conn_string)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return None


def load_pending_push() -> list[dict]:
    """Load models pending push from JSON file."""
    if not PENDING_PUSH_FILE.exists():
        return []

    try:
        with open(PENDING_PUSH_FILE) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load pending_push.json: {e}")
        return []


def save_pending_push(pending: list[dict]):
    """Save models pending push to JSON file."""
    try:
        with open(PENDING_PUSH_FILE, "w") as f:
            json.dump(pending, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save pending_push.json: {e}")


def push_model_to_ampere(run_id: str, item_id: int) -> bool:
    """Push a trained model to Ampere via rsync."""
    model_path = MODELS_DIR / run_id / str(item_id)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return False

    remote_path = f"{AMPERE_HOST}:{AMPERE_MODELS_DIR}/{run_id}/{item_id}/"

    try:
        # Create remote directory (with SSH key)
        subprocess.run(
            ["ssh", "-i", AMPERE_SSH_KEY, "-o", "StrictHostKeyChecking=no",
             AMPERE_HOST, f"mkdir -p {AMPERE_MODELS_DIR}/{run_id}/{item_id}"],
            check=True,
            capture_output=True,
            timeout=30,
        )

        # Rsync model files (with SSH key)
        result = subprocess.run(
            ["rsync", "-avz", "-e", f"ssh -i {AMPERE_SSH_KEY} -o StrictHostKeyChecking=no",
             f"{model_path}/", remote_path],
            check=True,
            capture_output=True,
            timeout=120,
        )

        logger.info(f"Pushed model {run_id}/{item_id} to Ampere")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout pushing model {run_id}/{item_id}")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to push model {run_id}/{item_id}: {e.stderr.decode()}")
        return False


def push_pending_models() -> int:
    """Push any pending models to Ampere. Returns count of successful pushes."""
    pending = load_pending_push()
    if not pending:
        return 0

    logger.info(f"Found {len(pending)} models pending push")

    successful = []
    failed = []

    for model in pending:
        if shutdown_requested:
            break

        run_id = model.get("run_id")
        item_id = model.get("item_id")

        if push_model_to_ampere(run_id, item_id):
            successful.append(model)
        else:
            failed.append(model)

    # Save remaining failed models back to pending
    if failed:
        save_pending_push(failed)
    elif PENDING_PUSH_FILE.exists():
        PENDING_PUSH_FILE.unlink()

    return len(successful)


def get_training_priorities(conn) -> list[TrainingPriority]:
    """Query validation data to determine training priorities."""
    priorities = []

    with conn.cursor() as cur:
        # Get items with high miss rates (from validation data)
        cur.execute("""
            WITH recent_outcomes AS (
                SELECT
                    item_id,
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'CLEAR_MISS' THEN 1 ELSE 0 END) as misses
                FROM prediction_outcomes
                WHERE created_at > NOW() - INTERVAL '7 days'
                GROUP BY item_id
                HAVING COUNT(*) >= 10
            )
            SELECT
                i.item_id as item_id,
                i.name as item_name,
                ro.misses::float / ro.total as miss_rate
            FROM recent_outcomes ro
            JOIN items i ON i.item_id = ro.item_id
            WHERE ro.misses::float / ro.total > %s
            ORDER BY ro.misses::float / ro.total DESC
            LIMIT 100
        """, (HIGH_MISS_THRESHOLD,))

        for row in cur.fetchall():
            priorities.append(TrainingPriority(
                item_id=row[0],
                item_name=row[1],
                priority='high_miss',
                miss_rate=row[2],
            ))

        # Get items without models (require sufficient historical data)
        cur.execute("""
            SELECT i.item_id, i.name
            FROM items i
            LEFT JOIN model_registry mr ON mr.item_id = i.item_id AND mr.status = 'ACTIVE'
            WHERE mr.id IS NULL
            AND i.item_id IN (
                SELECT item_id
                FROM price_data_5min
                WHERE timestamp > NOW() - INTERVAL '6 months'
                GROUP BY item_id
                HAVING COUNT(*) >= 5000
            )
            LIMIT 50
        """)

        for row in cur.fetchall():
            # Don't add if already in high_miss list
            if not any(p.item_id == row[0] for p in priorities):
                priorities.append(TrainingPriority(
                    item_id=row[0],
                    item_name=row[1],
                    priority='no_model',
                ))

        # Get stale models (older than STALE_MODEL_DAYS)
        cur.execute("""
            SELECT
                mr.item_id,
                i.name,
                mr.created_at,
                EXTRACT(DAY FROM NOW() - mr.created_at) as age_days
            FROM model_registry mr
            JOIN items i ON i.item_id = mr.item_id
            WHERE mr.status = 'ACTIVE'
            AND mr.created_at < NOW() - INTERVAL '%s days'
            ORDER BY mr.created_at ASC
            LIMIT 50
        """, (STALE_MODEL_DAYS,))

        for row in cur.fetchall():
            # Don't add if already in list
            if not any(p.item_id == row[0] for p in priorities):
                priorities.append(TrainingPriority(
                    item_id=row[0],
                    item_name=row[1],
                    priority='stale',
                    model_age_days=int(row[3]),
                    last_trained=row[2],
                ))

    return priorities


def prepare_training_data(item_id: int, run_id: str) -> bool:
    """Prepare training data for a single item. Returns True if successful."""
    logger.info(f"Preparing data for item {item_id} (run_id: {run_id})")

    # Hydra uses /home/ubuntu/gept/cloud/ not packages/model/cloud/
    prepare_script = GEPT_DIR / "cloud" / "prepare_runpod_data.py"
    cache_dir = GEPT_DIR / "data_cache"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(prepare_script),
                "--output-dir", str(cache_dir),
                "--run-id", run_id,
                "--items", str(item_id),
                "--months", "6",
                "--min-rows", "1000",  # Lower threshold for individual items
                "--max-items", "2000",  # Increase to include lower-volume items
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for data prep
            cwd=str(GEPT_DIR),
            env={**os.environ, "PYTHONPATH": str(GEPT_DIR / "src")},
        )

        if result.returncode == 0:
            logger.info(f"Data prepared for item {item_id}")
            return True
        else:
            logger.error(f"Data prep failed for item {item_id}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Data prep timed out for item {item_id}")
        return False
    except Exception as e:
        logger.error(f"Data prep error for item {item_id}: {e}")
        return False


def train_item(item_id: int, run_id: str) -> bool:
    """Train a single item model. Returns True if successful."""
    logger.info(f"Training item {item_id} (run_id: {run_id})")

    # Step 1: Prepare training data (features, parquet files)
    if not prepare_training_data(item_id, run_id):
        logger.error(f"Cannot train item {item_id}: data preparation failed")
        return False

    # Step 2: Run training
    # Hydra uses /home/ubuntu/gept/cloud/ not packages/model/cloud/
    train_script = GEPT_DIR / "cloud" / "train_runpod_multitarget.py"
    cache_dir = GEPT_DIR / "data_cache"
    output_dir = MODELS_DIR

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(train_script),
                "--run-id", run_id,
                "--items", str(item_id),
                "--local",
                "--cache-dir", str(cache_dir),
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per item
            cwd=str(GEPT_DIR),
            env={**os.environ, "PYTHONPATH": str(GEPT_DIR / "src")},
        )

        if result.returncode == 0:
            logger.info(f"Successfully trained item {item_id}")
            return True
        else:
            logger.error(f"Training failed for item {item_id}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Training timed out for item {item_id}")
        return False
    except Exception as e:
        logger.error(f"Training error for item {item_id}: {e}")
        return False


def update_model_registry(conn, item_id: int, item_name: str, run_id: str):
    """Update model registry with new model."""
    model_path = f"{AMPERE_MODELS_DIR}/{run_id}/{item_id}/model.cbm"

    with conn.cursor() as cur:
        # Deprecate old active model
        cur.execute("""
            UPDATE model_registry
            SET status = 'DEPRECATED', deprecated_at = NOW()
            WHERE item_id = %s AND status = 'ACTIVE'
        """, (item_id,))

        # Insert new model
        cur.execute("""
            INSERT INTO model_registry (item_id, item_name, run_id, model_path, status, created_at, trained_at)
            VALUES (%s, %s, %s, %s, 'ACTIVE', NOW(), NOW())
        """, (item_id, item_name, run_id, model_path))

        conn.commit()


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    """Main training loop."""
    logger.info("=" * 60)
    logger.info("GePT Trainer starting")
    logger.info(f"GEPT_DIR: {GEPT_DIR}")
    logger.info(f"MODELS_DIR: {MODELS_DIR}")
    logger.info(f"AMPERE_HOST: {AMPERE_HOST}")
    logger.info("=" * 60)

    # Ensure directories exist
    (GEPT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    while not shutdown_requested:
        # Step 1: Ensure DB tunnel is available
        if not ensure_db_tunnel():
            logger.warning(f"Cannot establish DB tunnel, retrying in {CONNECTION_RETRY_MINUTES} minutes")
            time.sleep(CONNECTION_RETRY_MINUTES * 60)
            continue

        # Step 2: Connect to database
        logger.info("Connecting to Ampere database...")
        conn = get_db_connection()

        if conn is None:
            logger.warning(f"Database unavailable, retrying in {CONNECTION_RETRY_MINUTES} minutes")
            time.sleep(CONNECTION_RETRY_MINUTES * 60)
            continue

        try:
            # Step 3: Push any pending models
            pushed = push_pending_models()
            if pushed > 0:
                logger.info(f"Pushed {pushed} pending models")

            if shutdown_requested:
                break

            # Step 4: Get training priorities from validation data
            logger.info("Querying training priorities...")
            priorities = get_training_priorities(conn)

            if not priorities:
                logger.info(f"No items need training. Sleeping for {IDLE_SLEEP_HOURS} hour(s)")
                conn.close()

                # Sleep in small intervals to check for shutdown
                for _ in range(IDLE_SLEEP_HOURS * 60):
                    if shutdown_requested:
                        break
                    time.sleep(60)
                continue

            # Log priorities
            high_miss = [p for p in priorities if p.priority == 'high_miss']
            no_model = [p for p in priorities if p.priority == 'no_model']
            stale = [p for p in priorities if p.priority == 'stale']

            logger.info(f"Training queue: {len(high_miss)} high-miss, {len(no_model)} no-model, {len(stale)} stale")

            # Step 5: Train items
            trained_count = 0

            for priority in priorities[:MAX_ITEMS_PER_CYCLE]:
                if shutdown_requested:
                    logger.info("Shutdown requested, stopping training loop")
                    break

                item_id = priority.item_id
                item_name = priority.item_name
                # Use unique run_id per item to avoid config.json overwrites
                run_id = f"{generate_run_id()}_{item_id}"
                logger.info(f"Training {item_name} (id={item_id}, priority={priority.priority})")

                success = train_item(item_id, run_id)

                if success:
                    # Push model to Ampere
                    if push_model_to_ampere(run_id, item_id):
                        update_model_registry(conn, item_id, item_name, run_id)
                        trained_count += 1
                    else:
                        # Queue for later push
                        pending = load_pending_push()
                        pending.append({"run_id": run_id, "item_id": item_id})
                        save_pending_push(pending)
                        logger.warning(f"Queued model {item_id} for later push")

            logger.info(f"Training cycle complete: {trained_count} models trained and pushed")

        except Exception as e:
            logger.error(f"Error in training loop: {e}")
        finally:
            if conn:
                conn.close()

        # Brief pause between cycles
        if not shutdown_requested:
            time.sleep(60)

    logger.info("GePT Trainer shutdown complete")


if __name__ == "__main__":
    main()
