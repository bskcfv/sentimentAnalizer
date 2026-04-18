from sklearn.feature_extraction.text import TfidfVectorizer

class TextVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=None,             
            ngram_range=(1,3),           
            max_features=30000,           
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)