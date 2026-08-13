import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def group_metrics(y_true, y_pred) -> dict:
    """Computes confusion matrix metrics for a single demographic subgroup."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    selection_rate = (tp + fp) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "Selection_Rate": selection_rate,
        "FPR": fpr,
        "FNR": fnr,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Support": len(y_true),
    }

def audit_demographic_fairness(df_test: pd.DataFrame, y_true, y_pred, group_col: str = "gender_group") -> pd.DataFrame:
    """Audits fairness metrics across demographic groups."""
    test_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "group": df_test[group_col].values,
    })

    fairness_results = []
    for g in sorted(test_df["group"].unique()):
        sub = test_df[test_df["group"] == g]
        m = group_metrics(sub["y_true"], sub["y_pred"])
        m["gender_group"] = g
        fairness_results.append(m)

    return pd.DataFrame(fairness_results).set_index("gender_group")
