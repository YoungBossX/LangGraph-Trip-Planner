"""FastAPI主应用"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config import get_settings, print_config, validate_config
from .middleware import RequestBodyLimitMiddleware
from .routes import poi, trip

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("\n" + "="*60)
    print(f"-> {settings.app_name} v{settings.app_version}")
    print("="*60)

    print_config()
    try:
        validate_config()
        print("\n[OK] 配置验证通过")
    except ValueError as e:
        print(f"\n[ERROR] 配置验证失败:\n{e}")
        print("\n请检查.env文件并确保所有必要的配置项都已设置")
        raise

    print("\n" + "="*60)
    print("[DOC] API文档: http://localhost:8000/docs")
    print("[DOC] ReDoc文档: http://localhost:8000/redoc")
    print("="*60 + "\n")

    yield

    # shutdown
    print("\n" + "="*60)
    print("[BYE] 应用正在关闭...")
    print("="*60 + "\n")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="基于LangChain/LangGraph框架的智能旅行规划助手API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=settings.max_request_body_bytes)

app.include_router(trip.router, prefix="/api")
app.include_router(poi.router, prefix="/api")

@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version
    }
