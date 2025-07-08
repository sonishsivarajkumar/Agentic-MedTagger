"""
Main application entry point.
"""

import asyncio
import logging
from pathlib import Path

from .web.app import create_app
from .core.pipeline import Pipeline


logger = logging.getLogger(__name__)


async def main():
    """Main application entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Agentic MedTagger...")
    
    # Create FastAPI app
    app = create_app()
    
    # Start the server
    import uvicorn
    
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
