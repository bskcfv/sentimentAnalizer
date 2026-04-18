import joblib
from app.nlp.preprocessing import clean_text

# cargar modelo y vectorizer
model = joblib.load("app/ml/artifacts/model.pkl")
vectorizer = joblib.load("app/ml/artifacts/vectorizer.pkl")

def _prepare_texts(texts):
    return [
        clean_text(str(t)).strip()
        for t in texts
        if isinstance(t, str) and t.strip()
    ]


def predict_sentiment(text: str):
    texts = _prepare_texts([text])
    X = vectorizer.transform(texts)
    return model.predict(X)[0]


def predict_text_batch(texts: list[str]):
    texts = _prepare_texts(texts)
    X = vectorizer.transform(texts)
    return model.predict(X)


if __name__ == "__main__":
    print(predict_sentiment("I am very happy today!"))  # should be positive
    print(predict_sentiment("I am sad and I hate this."))  # should be negative
    print(predict_sentiment("Im not feeling anything."))  # should be neutral
    print(predict_sentiment("I love this!"))  # should be positive
    print(predict_sentiment("I am so angry right now."))  # should be negative