import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))  # puedes cambiar a spanish si quieres

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # menciones
    text = re.sub(r"@\w+", "", text)

    # links
    text = re.sub(r"http\S+", "", text)

    text = text.replace("sad", "very_sad")
    text = text.replace("death", "very_negative")
    text = text.replace("died", "very_negative")
    text = text.replace("hate", "very_negative")
    text = text.replace("love", "very_positive")

    # solo limpia ruido fuerte
    text = re.sub(r"\s+", " ", text).strip()

    return text