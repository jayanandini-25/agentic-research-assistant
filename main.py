from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from core.logger import setup_logger
from config.settings import get_settings

settings = get_settings()
logger = setup_logger(__name__)

app = FastAPI(
    title="Agentic Research Assistant",
    description="AI-powered research assistant with multi-agent orchestration and RAG.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["Research"])


@app.on_event("startup")
async def on_startup():
    logger.info("=" * 50)
    logger.info("Agentic Research Assistant started")
    logger.info(f"Environment : {settings.app_env}")
    logger.info(f"Model       : {settings.openai_model}")
    logger.info("Docs at     : http://localhost:8000/docs")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Server shutting down.")
