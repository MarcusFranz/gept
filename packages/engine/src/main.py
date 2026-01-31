"""Main entry point for the prediction engine."""

import logging
import sys

import uvicorn
from dotenv import load_dotenv

from .config import config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/engine.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """Run the prediction engine API server."""
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.error(f"Configuration errors: {errors}")
        sys.exit(1)

    logger.info("Starting GePT Prediction Engine...")
    logger.info(f"API endpoint: http://{config.api_host}:{config.api_port}")

    # Run the server
    uvicorn.run(
        "src.api:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
