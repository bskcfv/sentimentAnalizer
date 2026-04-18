from fastapi import FastAPI
from app.db.connect import init_db
from app.api.routes import predict, health, updload

app = FastAPI(
    title="Emotion Analyzer API",
    version="1.0"
)

@app.on_event("startup")
async def startup():
    await init_db()

app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(updload.router, prefix="/api/upload", tags=["Upload"])