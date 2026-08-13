#!/usr/bin/env python3
"""
AutoResume-Screening Main Pipeline Entrypoint.
Executes data preprocessing, hybrid feature engineering, model training,
demographic fairness auditing, adversarial attack suite, and diagnostic visualization exports.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import OUTPUT_DIR, RANDOM_STATE, LABEL_COL
from src.preprocessing import (
    load_dataset,
    filter_demographics,
    combine_text_fields,
    extract_requirement_features,
)
from src.feature_engineering import HybridFeatureTransformer, build_tfidf_vectorizer
from src.models import (
    train_linear_svm,
    train_hybrid_classifier,
    train_rbf_svm,
    train_gaussian_nb,
    evaluate_model,
    cross_validate_model,
)
from src.fairness import audit_demographic_fairness
from src.robustness import run_adversarial_attack_suite
from src.explainability import explain_candidate, get_top_features
from src.visualization import (
    save_confusion_matrix_plot,
    save_top_features_plot,
    save_robustness_plot,
    save_dashboard_summary,
)

def run_pipeline():
    print("=====================================================")
    print("🚀 AutoResume-Screening Modular ML & Fairness Pipeline")
    print("=====================================================")

    # 1. Load Data
    print("\n[1/6] Loading dataset...")
    df = load_dataset()
    print(f"   Dataset loaded. Initial shape: {df.shape}")

    # 2. Preprocess & Feature Extraction
    print("\n[2/6] Preprocessing text and extracting requirement features...")
    df = filter_demographics(df)
    df["combined_text"] = combine_text_fields(df)
    df = extract_requirement_features(df)
    print(f"   Cleaned dataset shape: {df.shape}")
    print(f"   Good candidates: {df['good_candidate'].sum()} | Weak candidates: {df['weak_candidate'].sum()}")

    # 3. Train/Test Split & Feature Engineering
    print("\n[3/6] Splitting data & engineering hybrid features...")
    X_train_text, X_test_text, y_train, y_test, df_train, df_test = train_test_split(
        df["combined_text"],
        df[LABEL_COL],
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[LABEL_COL],
    )

    hybrid_transformer = HybridFeatureTransformer()
    X_train_hybrid = hybrid_transformer.fit_transform(X_train_text, df_train)
    X_test_hybrid = hybrid_transformer.transform(X_test_text, df_test)
    feature_names = hybrid_transformer.get_feature_names()

    print(f"   Hybrid Feature Matrix shape: {X_train_hybrid.shape}")

    # Also fit standard TF-IDF for baseline comparison
    tfidf_vec = build_tfidf_vectorizer()
    X_train_tfidf = tfidf_vec.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vec.transform(X_test_text)
    text_feature_names = np.array(tfidf_vec.get_feature_names_out()) if 'np' in globals() else hybrid_transformer.vectorizer.get_feature_names_out()

    # 4. Model Training & Evaluation
    print("\n[4/6] Training & benchmarking models...")
    metrics_list = []

    # 4.1 Hybrid Logistic Regression (Production Model)
    hybrid_clf = train_hybrid_classifier(X_train_hybrid, y_train)
    y_pred_hybrid = hybrid_clf.predict(X_test_hybrid)
    metrics_list.append(evaluate_model("Hybrid Classifier (Prod)", y_test, y_pred_hybrid, verbose=True))

    # 4.2 Baseline Linear SVM
    linear_svm = train_linear_svm(X_train_tfidf, y_train)
    y_pred_linear = linear_svm.predict(X_test_tfidf)
    metrics_list.append(evaluate_model("Linear SVM (Baseline)", y_test, y_pred_linear, verbose=False))

    # 4.3 RBF SVM
    rbf_svm = train_rbf_svm(X_train_tfidf, y_train)
    y_pred_rbf = rbf_svm.predict(X_test_tfidf)
    metrics_list.append(evaluate_model("RBF SVM", y_test, y_pred_rbf, verbose=False))

    # 4.4 Gaussian NB
    gnb = train_gaussian_nb(X_train_tfidf, y_train)
    y_pred_gnb = gnb.predict(X_test_tfidf.toarray())
    metrics_list.append(evaluate_model("Gaussian NB", y_test, y_pred_gnb, verbose=False))

    # Cross-Validation
    cv_scores = cross_validate_model(train_hybrid_classifier(X_train_hybrid, y_train), X_train_hybrid, y_train, cv=5)
    print(f"\n   5-Fold Cross-Validation F1: {cv_scores.mean():.3f} (± {cv_scores.std():.3f})")

    # 5. Fairness & Robustness Audit
    print("\n[5/6] Auditing demographic fairness & adversarial robustness...")
    fairness_df = audit_demographic_fairness(df_test, y_test, y_pred_hybrid)
    print("\nDemographic Fairness Audit (Hybrid Classifier):")
    print(fairness_df)

    top_pos, _ = get_top_features(hybrid_clf.coef_[0], feature_names, top_k=10)
    top_keywords = [f[0] for f in top_pos if f[0] in text_feature_names]

    robustness_df = run_adversarial_attack_suite(
        model=hybrid_clf,
        feature_transformer=hybrid_transformer,
        X_text_list=X_test_text.tolist(),
        df_test=df_test,
        top_keywords=top_keywords,
    )
    print("\nAdversarial Robustness Results:")
    print(robustness_df)

    # 6. Visualization Export
    print("\n[6/6] Generating visual dashboard artifacts...")
    save_confusion_matrix_plot("Hybrid Classifier", y_test, y_pred_hybrid)
    save_confusion_matrix_plot("Linear SVM Baseline", y_test, y_pred_linear)
    save_top_features_plot(feature_names, hybrid_clf.coef_[0])
    save_robustness_plot(robustness_df)
    save_dashboard_summary(
        fairness_df=fairness_df,
        robustness_table=robustness_df,
        feature_names=feature_names,
        coefs=hybrid_clf.coef_[0],
        y_true=y_test,
        y_pred=y_pred_hybrid,
    )
    print(f"   Visual artifacts saved to: {OUTPUT_DIR}")

    # Sample Explainer Demonstration
    explain_candidate(
        model=hybrid_clf,
        feature_transformer=hybrid_transformer,
        feature_names=feature_names,
        df=df_test,
        X_text=X_test_text,
        y_true=y_test,
        gender_series=df_test["gender_group"],
        fairness_df=fairness_df,
    )

    print("\n✅ Pipeline execution finished successfully!")

if __name__ == "__main__":
    import numpy as np
    run_pipeline()
