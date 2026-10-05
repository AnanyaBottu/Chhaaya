from fastapi import FastAPI

from chhaaya.webhook import router

app = FastAPI(title="Chhaaya")
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
