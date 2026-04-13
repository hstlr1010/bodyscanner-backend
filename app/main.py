from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, engine
from app.routers.health import router as health_router
from app.routers.scans import router as contacts_router
from app.routers.scans import scan_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables on startup (use Alembic migrations for production)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(
    title="BodyScanner API",
    version="0.1.0",
    description="Body measurement inference backend for BodyScanner iOS app and bizbozz React widget.",
    lifespan=lifespan,
)

# Allow bizbozz web app + iOS app (iOS uses direct HTTPS, not a browser origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.gymheros.com",   # bizbozz web app — adjust to real domain
        "https://bizbozz.com",
        "http://localhost:3000",       # local bizbozz dev
        "http://localhost:5173",       # local Vite dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(contacts_router)
app.include_router(scan_router)
