from fastapi import APIRouter, Depends
from app.db.connect import get_db
from app.models.request_models import TextRequest
from app.services.ml_service import predict_emotion
from app.services.comments_service import save_prediction
router = APIRouter()

@router.post("/")
def predict(request: TextRequest):
    result = predict_emotion(request.text)
    return {"emotion": result}


@router.post("/post")
async def predict(request: TextRequest, conn = Depends(get_db)):

    return await save_prediction(request.text, predict_emotion(request.text), conn)

@router.get("/history")
async def history(conn = Depends(get_db)):
    from app.services.comments_service import get_predictions
    return await get_predictions(conn)

