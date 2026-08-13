import numpy as np
import pandas as pd
from src.config import NAME_COL

def get_top_features(coefs: np.ndarray, feature_names: np.ndarray, top_k: int = 20) -> tuple:
    """Returns top positive and negative features based on model coefficients."""
    top_pos_idx = np.argsort(coefs)[-top_k:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_k]
    
    pos_features = list(zip(feature_names[top_pos_idx], coefs[top_pos_idx]))
    neg_features = list(zip(feature_names[top_neg_idx], coefs[top_neg_idx]))
    return pos_features, neg_features

def explain_candidate(
    model,
    feature_transformer,
    feature_names: np.ndarray,
    df: pd.DataFrame,
    X_text: pd.Series,
    y_true: pd.Series,
    gender_series: pd.Series,
    fairness_df: pd.DataFrame = None,
    candidate_idx: int = None,
    name_col: str = NAME_COL,
    top_local_words: int = 8,
):
    """Generates detailed diagnostic explanation for a single candidate profile."""
    if candidate_idx is None:
        rand_pos = np.random.randint(0, len(X_text))
        idx = X_text.index[rand_pos]
    else:
        idx = candidate_idx

    text = X_text.loc[idx]
    true_label = int(y_true.loc[idx])
    gender_group = gender_series.loc[idx]
    row = df.loc[idx]

    if hasattr(feature_transformer, "transform"):
        vec = feature_transformer.transform(pd.Series([text]), pd.DataFrame([row]))
    else:
        vec = feature_transformer.transform([text])

    if hasattr(model, "decision_function"):
        decision = model.decision_function(vec)[0]
    elif hasattr(model, "predict_proba"):
        decision = model.predict_proba(vec)[0][1]
    else:
        decision = 0.0

    pred_label = int(model.predict(vec)[0])

    coefs = model.coef_[0] if hasattr(model, "coef_") else np.zeros(len(feature_names))
    vec_dense = vec.toarray()[0] if hasattr(vec, "toarray") else vec[0]
    contributions = coefs * vec_dense
    
    non_zero_idx = np.where(vec_dense != 0)[0]
    non_zero_contribs = contributions[non_zero_idx]
    non_zero_features = feature_names[non_zero_idx]

    pos_mask = non_zero_contribs > 0
    neg_mask = non_zero_contribs < 0
    
    pos_features = non_zero_features[pos_mask]
    pos_values = non_zero_contribs[pos_mask]
    neg_features = non_zero_features[neg_mask]
    neg_values = non_zero_contribs[neg_mask]

    pos_order = np.argsort(-pos_values)[:top_local_words]
    neg_order = np.argsort(neg_values)[:top_local_words]

    top_help = list(zip(pos_features[pos_order], pos_values[pos_order]))
    top_hurt = list(zip(neg_features[neg_order], neg_values[neg_order]))

    name = row.get(name_col, "Unknown")
    jp = row.get("job_position_name", "Unknown position")

    skills_meet = int(row.get("skills_meet_all_required", 0))
    exp_meet = int(row.get("experience_meets", 0))
    edu_meet = int(row.get("education_meets", 0))
    good_candidate = int(row.get("good_candidate", 0))
    matched_score = row.get("matched_score", np.nan)
    exp_gap = row.get("experience_gap", np.nan)
    edu_gap = row.get("education_gap", np.nan)
    skill_overlap = row.get("skill_overlap_ratio_req", np.nan)
    missing_skills = int(row.get("missing_required_skill_count", 0))

    print("\n================= CANDIDATE EXPLANATION =================\n")
    print(f"Candidate: {name}  (index {idx})")
    print(f"Applied for: {jp}")
    print(f"Inferred group: {gender_group}")
    print(f"True label    : {'SHORTLIST' if true_label == 1 else 'REJECT'}")
    print(f"Model predict : {'SHORTLIST' if pred_label == 1 else 'REJECT'}")
    print(f"Decision score: {decision:.3f}\n")

    print("Requirement satisfaction:")
    print(f"  • Skills requirement met?       {'YES' if skills_meet else 'NO'}")
    print(f"  • Experience requirement met?   {'YES' if exp_meet else 'NO'}")
    print(f"  • Education requirement met?    {'YES' if edu_meet else 'NO'}")
    if not np.isnan(exp_gap):
        print(f"  • Experience gap (cand - req):  {exp_gap:.1f} years")
    if not np.isnan(edu_gap):
        print(f"  • Education gap (cand - req):   {edu_gap:.1f} level(s)")
    if not np.isnan(skill_overlap):
        print(f"  • Skill overlap (of required):  {skill_overlap*100:.1f}%")
        print(f"  • Missing required skills:      {missing_skills}")

    print("\nOverall requirement-based verdict:")
    if good_candidate:
        print("  → Candidate looks GOOD based on requirements / matched_score.")
    else:
        print("  → Candidate looks WEAK / borderline based on requirements / matched_score.")

    print("\nTop features helping prediction:")
    for w, v in top_help:
        print(f"   + '{w}' (contribution: {v:.4f})")

    print("\nHurtful features:")
    for w, v in top_hurt:
        print(f"   - '{w}' (contribution: {v:.4f})")

    print("\n================= END CANDIDATE EXPLANATION =================\n")
