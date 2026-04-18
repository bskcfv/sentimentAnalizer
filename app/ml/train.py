import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from app.nlp.preprocessing import clean_text
from app.nlp.vectorizer import TextVectorizer


# =========================
# 1. PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH1 = os.path.join(BASE_DIR, "data", "twitter_training.csv")
DATA_PATH2 = os.path.join(BASE_DIR, "data", "Tweets.csv")

# =========================
# 2. DATASET 1 (twitter_training)
# =========================
df1 = pd.read_csv(DATA_PATH1, header=None)
df1.columns = ["id", "entity", "label", "text"]

df1 = df1[["text", "label"]]
df1 = df1.dropna()

df1["label"] = df1["label"].str.lower()


# =========================
# 3. DATASET 2 (Tweets.csv)
# =========================
df2 = pd.read_csv(DATA_PATH2)

# este dataset usa columnas distintas
df2 = df2[["text", "airline_sentiment"]]
df2 = df2.dropna()

df2 = df2.rename(columns={
    "airline_sentiment": "label"
})

df2["label"] = df2["label"].str.lower()


# DATASET 3
positive_emotions = [
    "admiration", "amusement", "approval", "caring",
    "excitement", "gratitude", "joy", "love",
    "optimism", "pride", "relief"
]

negative_emotions = [
    "anger", "annoyance", "disappointment", "disapproval",
    "disgust", "embarrassment", "fear", "grief",
    "nervousness", "remorse", "sadness"
]

neutral_emotions = [
    "confusion", "curiosity", "realization", "surprise", "neutral"
]


# =========================
# 4. UNIR DATASETS
# =========================
df = pd.concat([df1, df2], ignore_index=True)


df["label"] = df["label"].astype(str).str.lower().str.strip()

df["label"] = df["label"].replace({
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positve": "positive",
    "negativ": "negative"
})

df = df[df["text"].str.len() > 3]
df = df[df["label"].isin(["positive", "negative", "neutral"])]

# =========================
# 5. LIMPIEZA DE TEXTO
# =========================
df["clean_text"] = df["text"].astype(str).apply(clean_text)

X = df["clean_text"]
y = df["label"]


# =========================
# 6. VECTORIZACIÓN
# =========================
vectorizer = TextVectorizer()
X_vec = vectorizer.fit_transform(X)


# =========================
# 7. TRAIN / TEST
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y,
    test_size=0.2,
    random_state=42,
    stratify=y  
)


# =========================
# 8. MODELO
# =========================
model = LinearSVC(
    C=1.2,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# =========================
# 9. EVALUACIÓN
# =========================
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)


# =========================
# 10. GUARDAR MODELO
# =========================
ARTIFACTS_DIR = os.path.join(BASE_DIR, "app/ml/artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

joblib.dump(model, os.path.join(ARTIFACTS_DIR, "model.pkl"))
joblib.dump(vectorizer, os.path.join(ARTIFACTS_DIR, "vectorizer.pkl"))