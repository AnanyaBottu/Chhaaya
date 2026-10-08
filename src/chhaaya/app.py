from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from chhaaya.webhook import get_settings, router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    get_settings()
    yield


app = FastAPI(title="Chhaaya", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
