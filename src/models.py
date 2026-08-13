import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from src.config import RANDOM_STATE
from src.feature_engineering import build_tfidf_vectorizer, HybridFeatureTransformer

def evaluate_model(name: str, y_true, y_pred, verbose: bool = False) -> dict:
    """Evaluates classification model and returns standardized performance metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    if verbose:
        print(f"\n===== {name} =====")
        print("Accuracy :", acc)
        print("Precision:", prec)
        print("Recall   :", rec)
        print("F1-score :", f1)
        print("FNR      :", fnr)
        print("FPR      :", fpr)
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "FNR": fnr,
        "FPR": fpr,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }

def train_linear_svm(X_train, y_train, C: float = 1.0) -> LinearSVC:
    model = LinearSVC(random_state=RANDOM_STATE, C=C)
    model.fit(X_train, y_train)
    return model

def train_hybrid_classifier(X_train, y_train, C: float = 1.0) -> LogisticRegression:
    """Trains a Logistic Regression model on hybrid features (Text + Structured Requirements)."""
    model = LogisticRegression(random_state=RANDOM_STATE, C=C, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_rbf_svm(X_train, y_train, C: float = 1.0) -> SVC:
    model = SVC(kernel="rbf", C=C, gamma="scale", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

def train_gaussian_nb(X_train, y_train) -> GaussianNB:
    X_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    model = GaussianNB()
    model.fit(X_dense, y_train)
    return model

def cross_validate_model(model, X_train, y_train, cv: int = 5) -> np.ndarray:
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
