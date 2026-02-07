"""Webhook endpoints (events from the web application)."""


from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request

from ..api_dependencies import get_trade_event_handler
from ..api_models import TradeWebhookRequest, TradeWebhookResponse
from ..logging_config import get_logger
from ..trade_events import TradeEvent, TradeEventHandler, TradeEventType, TradePayload
from ..webhook import WebhookSignatureError, verify_webhook_signature

logger = get_logger(__name__)

router = APIRouter()


@router.post("/webhooks/trades", response_model=TradeWebhookResponse)
async def receive_trade_webhook(
    request: Request,
    body: TradeWebhookRequest,
    trade_event_handler: TradeEventHandler = Depends(get_trade_event_handler),
):
    """Receive trade lifecycle events from the web application."""
    timestamp = request.headers.get("X-Webhook-Timestamp")
    signature = request.headers.get("X-Webhook-Signature")

    if not timestamp or not signature:
        logger.warning("Webhook request missing signature headers")
        raise HTTPException(
            status_code=401,
            detail="Missing X-Webhook-Timestamp or X-Webhook-Signature header",
        )

    try:
        raw_body = await request.body()
        verify_webhook_signature(
            body=raw_body.decode("utf-8"),
            timestamp=timestamp,
            signature=signature,
        )
    except WebhookSignatureError as e:
        logger.warning("Webhook signature verification failed", error=str(e))
        raise HTTPException(status_code=401, detail=str(e))

    try:
        event_timestamp = datetime.fromisoformat(body.timestamp.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")

    event = TradeEvent(
        event_type=TradeEventType(body.eventType),
        timestamp=event_timestamp,
        user_id=body.userId,
        trade_id=body.tradeId,
        payload=TradePayload(
            item_id=body.payload.itemId,
            item_name=body.payload.itemName,
            buy_price=body.payload.buyPrice,
            sell_price=body.payload.sellPrice,
            quantity=body.payload.quantity,
            profit=body.payload.profit,
            rec_id=body.payload.recId,
            model_id=body.payload.modelId,
            expected_hours=body.payload.expectedHours,
            created_at=(
                datetime.fromisoformat(body.payload.createdAt.replace("Z", "+00:00"))
                if body.payload.createdAt
                else None
            ),
        ),
    )

    try:
        await trade_event_handler.handle_event(event)
    except Exception as e:
        logger.error("Error processing trade event", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process event")

    logger.info("Trade webhook processed", event_type=body.eventType, trade_id=body.tradeId)
    return TradeWebhookResponse(success=True)

