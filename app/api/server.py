"""FastAPI application factory and instance per SPEC.md section 16."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db.base import init_db
from app.logging import get_logger

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context initializing DB schema on startup."""
    logger.info("api_server_startup")
    try:
        init_db()
    except Exception as exc:
        logger.warn("api_db_init_warning", error=str(exc))
    yield
    logger.info("api_server_shutdown")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Multi-Agent Assistant API",
        description="REST API for Multi-Agent Assistant with Human-in-the-Loop approvals",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
