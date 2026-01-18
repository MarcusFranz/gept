"""
GePT Public API - FastAPI service for webapp integration

Exposes prediction data and item information for the gept.gg frontend.
Runs on port 8000 by default.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

    # Development with auto-reload:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from db_utils import get_db_cursor, get_connection, release_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_VERSION = "1.0.0"
ALLOWED_ORIGINS = [
    "https://gept.gg",
    "https://www.gept.gg",
    "http://localhost:3000",  # Local development
    "http://localhost:5173",  # Vite dev server
]


# Response models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    database: str
    last_prediction: Optional[str] = None


class PredictionResponse(BaseModel):
    item_id: int
    item_name: str
    hour_offset: int
    target_hour: str
    offset_pct: float
    fill_probability: float
    expected_value: float
    buy_price: float
    sell_price: float
    current_high: Optional[float] = None
    current_low: Optional[float] = None
    confidence: str
    time: str


class OpportunityResponse(BaseModel):
    item_id: int
    item_name: str
    hour_offset: int
    offset_pct: float
    fill_probability: float
    expected_value: float
    buy_price: float
    sell_price: float
    confidence: str
    potential_profit: float
    roi_pct: float


class ItemResponse(BaseModel):
    item_id: int
    name: str
    examine: Optional[str] = None
    members: Optional[bool] = None
    high_alch: Optional[int] = None
    low_alch: Optional[int] = None
    buy_limit: Optional[int] = None
    icon_url: Optional[str] = None


class PredictionsListResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    prediction_time: str


class OpportunitiesListResponse(BaseModel):
    opportunities: List[OpportunityResponse]
    count: int
    prediction_time: str


class StatsResponse(BaseModel):
    total_items: int
    total_predictions: int
    last_inference: str
    avg_fill_probability: float
    top_opportunity_ev: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("GePT API starting up...")
    # Test database connection
    try:
        conn = get_connection()
        release_connection(conn)
        logger.info("Database connection verified")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
    yield
    logger.info("GePT API shutting down...")


app = FastAPI(
    title="GePT API",
    description="Grand Exchange Price Prediction API for OSRS",
    version=API_VERSION,
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring and Vercel."""
    db_status = "healthy"
    last_pred = None

    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT 1")
            cur.execute("SELECT MAX(time) FROM predictions")
            result = cur.fetchone()
            if result and result[0]:
                last_pred = result[0].isoformat()
    except Exception as e:
        logger.error(f"Health check DB error: {e}")
        db_status = "unhealthy"

    return HealthResponse(
        status="ok" if db_status == "healthy" else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        version=API_VERSION,
        database=db_status,
        last_prediction=last_pred,
    )


@app.get("/api/predictions", response_model=PredictionsListResponse)
async def get_predictions(
    item_id: Optional[int] = Query(None, description="Filter by item ID"),
    min_probability: float = Query(0.05, ge=0, le=1, description="Minimum fill probability"),
    max_probability: float = Query(0.30, ge=0, le=1, description="Maximum fill probability (filter out noise)"),
    hour_offset: Optional[int] = Query(None, ge=1, le=24, description="Filter by specific hour offset"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results to return"),
):
    """Get latest predictions with optional filters."""
    try:
        with get_db_cursor() as cur:
            # Get latest prediction time
            cur.execute("SELECT MAX(time) FROM predictions")
            latest_time = cur.fetchone()[0]

            if not latest_time:
                return PredictionsListResponse(predictions=[], count=0, prediction_time="")

            # Build query with filters
            query = """
                SELECT item_id, item_name, hour_offset, target_hour, offset_pct,
                       fill_probability, expected_value, buy_price, sell_price,
                       current_high, current_low, confidence, time
                FROM predictions
                WHERE time = %s
                  AND fill_probability >= %s
                  AND fill_probability <= %s
            """
            params = [latest_time, min_probability, max_probability]

            if item_id is not None:
                query += " AND item_id = %s"
                params.append(item_id)

            if hour_offset is not None:
                query += " AND hour_offset = %s"
                params.append(hour_offset)

            query += " ORDER BY expected_value DESC LIMIT %s"
            params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()

            predictions = [
                PredictionResponse(
                    item_id=row[0],
                    item_name=row[1],
                    hour_offset=row[2],
                    target_hour=row[3].isoformat() if row[3] else "",
                    offset_pct=float(row[4]),
                    fill_probability=float(row[5]),
                    expected_value=float(row[6]),
                    buy_price=float(row[7]),
                    sell_price=float(row[8]),
                    current_high=float(row[9]) if row[9] else None,
                    current_low=float(row[10]) if row[10] else None,
                    confidence=row[11],
                    time=row[12].isoformat() if row[12] else "",
                )
                for row in rows
            ]

            return PredictionsListResponse(
                predictions=predictions,
                count=len(predictions),
                prediction_time=latest_time.isoformat(),
            )

    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch predictions")


@app.get("/api/opportunities", response_model=OpportunitiesListResponse)
async def get_opportunities(
    min_ev: float = Query(0.001, description="Minimum expected value"),
    min_probability: float = Query(0.05, ge=0, le=1),
    max_probability: float = Query(0.30, ge=0, le=1),
    limit: int = Query(50, ge=1, le=200),
):
    """Get top trading opportunities ranked by expected value."""
    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT MAX(time) FROM predictions")
            latest_time = cur.fetchone()[0]

            if not latest_time:
                return OpportunitiesListResponse(opportunities=[], count=0, prediction_time="")

            cur.execute("""
                SELECT item_id, item_name, hour_offset, offset_pct,
                       fill_probability, expected_value, buy_price, sell_price,
                       confidence
                FROM predictions
                WHERE time = %s
                  AND expected_value >= %s
                  AND fill_probability >= %s
                  AND fill_probability <= %s
                ORDER BY expected_value DESC
                LIMIT %s
            """, (latest_time, min_ev, min_probability, max_probability, limit))

            rows = cur.fetchall()

            opportunities = [
                OpportunityResponse(
                    item_id=row[0],
                    item_name=row[1],
                    hour_offset=row[2],
                    offset_pct=float(row[3]),
                    fill_probability=float(row[4]),
                    expected_value=float(row[5]),
                    buy_price=float(row[6]),
                    sell_price=float(row[7]),
                    confidence=row[8],
                    potential_profit=float(row[7]) - float(row[6]),
                    roi_pct=((float(row[7]) - float(row[6])) / float(row[6]) * 100) if float(row[6]) > 0 else 0,
                )
                for row in rows
            ]

            return OpportunitiesListResponse(
                opportunities=opportunities,
                count=len(opportunities),
                prediction_time=latest_time.isoformat(),
            )

    except Exception as e:
        logger.error(f"Error fetching opportunities: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch opportunities")


@app.get("/api/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int):
    """Get item metadata by ID."""
    try:
        with get_db_cursor() as cur:
            cur.execute("""
                SELECT id, name, examine, members, highalch, lowalch, buy_limit, icon
                FROM items
                WHERE id = %s
            """, (item_id,))
            row = cur.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

            return ItemResponse(
                item_id=row[0],
                name=row[1],
                examine=row[2],
                members=row[3],
                high_alch=row[4],
                low_alch=row[5],
                buy_limit=row[6],
                icon_url=f"https://oldschool.runescape.wiki/images/{row[7]}" if row[7] else None,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching item {item_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch item")


@app.get("/api/items/{item_id}/predictions", response_model=PredictionsListResponse)
async def get_item_predictions(
    item_id: int,
    hours: int = Query(24, ge=1, le=168, description="Hours of predictions to return"),
):
    """Get all predictions for a specific item (current forecast)."""
    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT MAX(time) FROM predictions")
            latest_time = cur.fetchone()[0]

            if not latest_time:
                return PredictionsListResponse(predictions=[], count=0, prediction_time="")

            cur.execute("""
                SELECT item_id, item_name, hour_offset, target_hour, offset_pct,
                       fill_probability, expected_value, buy_price, sell_price,
                       current_high, current_low, confidence, time
                FROM predictions
                WHERE time = %s AND item_id = %s
                ORDER BY hour_offset, offset_pct
            """, (latest_time, item_id))

            rows = cur.fetchall()

            if not rows:
                raise HTTPException(status_code=404, detail=f"No predictions for item {item_id}")

            predictions = [
                PredictionResponse(
                    item_id=row[0],
                    item_name=row[1],
                    hour_offset=row[2],
                    target_hour=row[3].isoformat() if row[3] else "",
                    offset_pct=float(row[4]),
                    fill_probability=float(row[5]),
                    expected_value=float(row[6]),
                    buy_price=float(row[7]),
                    sell_price=float(row[8]),
                    current_high=float(row[9]) if row[9] else None,
                    current_low=float(row[10]) if row[10] else None,
                    confidence=row[11],
                    time=row[12].isoformat() if row[12] else "",
                )
                for row in rows
            ]

            return PredictionsListResponse(
                predictions=predictions,
                count=len(predictions),
                prediction_time=latest_time.isoformat(),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching predictions for item {item_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch item predictions")


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get summary statistics about predictions."""
    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT MAX(time) FROM predictions")
            latest_time = cur.fetchone()[0]

            if not latest_time:
                raise HTTPException(status_code=503, detail="No predictions available")

            cur.execute("""
                SELECT
                    COUNT(DISTINCT item_id) as total_items,
                    COUNT(*) as total_predictions,
                    AVG(fill_probability) as avg_prob,
                    MAX(expected_value) as top_ev
                FROM predictions
                WHERE time = %s
                  AND fill_probability BETWEEN 0.05 AND 0.30
            """, (latest_time,))

            row = cur.fetchone()

            return StatsResponse(
                total_items=row[0] or 0,
                total_predictions=row[1] or 0,
                last_inference=latest_time.isoformat(),
                avg_fill_probability=float(row[2]) if row[2] else 0.0,
                top_opportunity_ev=float(row[3]) if row[3] else 0.0,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch stats")


@app.get("/api/items/search")
async def search_items(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search items by name."""
    try:
        with get_db_cursor() as cur:
            cur.execute("""
                SELECT id, name, members, buy_limit
                FROM items
                WHERE LOWER(name) LIKE LOWER(%s)
                ORDER BY name
                LIMIT %s
            """, (f"%{q}%", limit))

            rows = cur.fetchall()

            return {
                "items": [
                    {
                        "item_id": row[0],
                        "name": row[1],
                        "members": row[2],
                        "buy_limit": row[3],
                    }
                    for row in rows
                ],
                "count": len(rows),
            }

    except Exception as e:
        logger.error(f"Error searching items: {e}")
        raise HTTPException(status_code=500, detail="Failed to search items")


# Root redirect to docs
@app.get("/")
async def root():
    """Redirect to API documentation."""
    return {"message": "GePT API", "version": API_VERSION, "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
