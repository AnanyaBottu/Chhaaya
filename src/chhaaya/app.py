from fastapi import FastAPI

app = FastAPI(title="Chhaaya")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
