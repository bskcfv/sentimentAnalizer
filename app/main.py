from fastapi import FastAPI
from app.db.connect import init_db
from app.api.routes import predict, health, updload
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Emotion Analyzer API",
    version="1.0"
)

origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await init_db()

app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(updload.router, prefix="/api/upload", tags=["Upload"])