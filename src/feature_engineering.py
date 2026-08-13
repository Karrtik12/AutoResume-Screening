import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from src.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE

TABULAR_FEATURE_COLS = [
    "experience_gap",
    "education_gap",
    "skill_overlap_ratio_req",
    "missing_required_skill_count",
    "matched_score",
    "meets_all_requirements",
]

def build_tfidf_vectorizer(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple = TFIDF_NGRAM_RANGE,
    stop_words: str = "english"
) -> TfidfVectorizer:
    """Builds TF-IDF vectorizer instance."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=stop_words,
    )

class HybridFeatureTransformer:
    """
    Combines sparse TF-IDF text features with scaled dense tabular requirement features
    to build a hybrid representation that resists keyword-stuffing attacks.
    """
    def __init__(self, vectorizer=None, scaler=None, tabular_cols=None):
        self.vectorizer = vectorizer if vectorizer is not None else build_tfidf_vectorizer()
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.tabular_cols = tabular_cols if tabular_cols is not None else TABULAR_FEATURE_COLS
        self.is_fitted = False

    def fit(self, X_text: pd.Series, df_tabular: pd.DataFrame):
        self.vectorizer.fit(X_text)
        tabular_data = df_tabular[self.tabular_cols].fillna(0.0).values
        self.scaler.fit(tabular_data)
        self.is_fitted = True
        return self

    def transform(self, X_text: pd.Series, df_tabular: pd.DataFrame):
        if not self.is_fitted:
            raise ValueError("HybridFeatureTransformer must be fitted before transforming.")
        
        X_tfidf = self.vectorizer.transform(X_text)
        tabular_data = df_tabular[self.tabular_cols].fillna(0.0).values
        X_tab_scaled = csr_matrix(self.scaler.transform(tabular_data))
        
        return hstack([X_tfidf, X_tab_scaled]).tocsr()

    def fit_transform(self, X_text: pd.Series, df_tabular: pd.DataFrame):
        self.fit(X_text, df_tabular)
        return self.transform(X_text, df_tabular)

    def get_feature_names(self):
        text_features = list(self.vectorizer.get_feature_names_out())
        return np.array(text_features + self.tabular_cols)
