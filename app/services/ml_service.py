import joblib
from app.nlp.preprocessing import clean_text
from pathlib import Path
import sys
from app import nlp


sys.modules['nlp'] = nlp

BASE_DIR = Path(__file__).resolve().parents[2]

model = joblib.load(BASE_DIR / "app/ml/artifacts/model.pkl")
vectorizer = joblib.load(BASE_DIR / "app/ml/artifacts/vectorizer.pkl")


def predict_emotion(text: str):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]

