"""Root endpoint with API info."""


from fastapi import APIRouter

from ..app_metadata import APP_VERSION

router = APIRouter()


@router.get("/")
async def root():
    return {
        "name": "GePT Recommendation Engine",
        "version": APP_VERSION,
        "description": "Transforms raw predictions into trade recommendations",
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "recommendations": "POST/GET /api/v1/recommendations",
            "by_id": "GET /api/v1/recommendations/{rec_id}",
            "by_item_id": "GET /api/v1/recommendations/item/{item_id}",
            "by_item_name": "GET /api/v1/recommendations/item?name={name}",
            "evaluate_order": "POST /api/v1/recommendations/update",
            "guidance": "POST /api/v1/guidance",
            "predictions": "GET /api/v1/predictions/{item_id}",
            "item_search": "GET /api/v1/items/search?q=<query>&limit=<limit>",
            "report_outcome": "POST /api/v1/recommendations/{rec_id}/outcome",
            "submit_feedback": "POST /api/v1/feedback",
            "feedback_analytics": "GET /api/v1/feedback/analytics",
        },
        "web_frontend_endpoints": {
            "recommendations": "GET /recommendations",
            "item": "GET /item/{item_id}",
            "search": "GET /search-items",
            "trade_outcome": "POST /trade-outcome",
            "health": "GET /health",
        },
        "webhooks": {
            "trades": "POST /webhooks/trades",
        },
    }
