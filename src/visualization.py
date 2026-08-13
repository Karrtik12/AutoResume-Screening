import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.config import OUTPUT_DIR

def save_confusion_matrix_plot(name: str, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp.plot(ax=ax, cmap="viridis", colorbar=True)
    ax.set_title(f"Confusion Matrix ({name})")
    plt.tight_layout()
    cm_file = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_{name.split()[0].lower().replace('(', '').replace(')', '')}.png"
    )
    fig.savefig(cm_file, bbox_inches="tight")
    plt.close(fig)

def save_top_features_plot(feature_names: np.ndarray, coefs: np.ndarray, top_k: int = 15):
    top_idx = np.argsort(np.abs(coefs))[-top_k:]
    top_feats = feature_names[top_idx]
    top_vals = coefs[top_idx]
    order = np.argsort(top_vals)
    
    top_feats = top_feats[order]
    top_vals = top_vals[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_feats, top_vals)
    ax.set_title("Top 15 Model Features (Coef Magnitude)")
    ax.set_xlabel("Coefficient")
    ax.invert_yaxis()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "top_features.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

def save_robustness_plot(robustness_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(robustness_table["attack"], robustness_table["flip_rate"])
    for i, v in enumerate(robustness_table["flip_rate"].values):
        ax.text(i, v + 0.0005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Flip Rate")
    ax.set_title("Robustness: Flip Rates by Attack")
    ax.set_xticks(range(len(robustness_table)))
    ax.set_xticklabels(robustness_table["attack"], rotation=25, ha="right")
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "robustness_flip_rates_pretty.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

def save_dashboard_summary(
    fairness_df: pd.DataFrame,
    robustness_table: pd.DataFrame,
    feature_names: np.ndarray,
    coefs: np.ndarray,
    y_true,
    y_pred
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1,1) Fairness
    fairness_plot = fairness_df[["FPR", "FNR"]]
    groups = fairness_plot.index.tolist()
    x = np.arange(len(groups))
    width = 0.35

    axes[0, 0].bar(x - width/2, fairness_plot["FPR"].values, width, label="FPR")
    axes[0, 0].bar(x + width/2, fairness_plot["FNR"].values, width, label="FNR")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(groups)
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_title("Fairness by Group (FPR & FNR)")
    axes[0, 0].legend()

    # (1,2) Robustness
    axes[0, 1].bar(robustness_table["attack"], robustness_table["flip_rate"])
    for i, v in enumerate(robustness_table["flip_rate"].values):
        axes[0, 1].text(i, v + 0.0005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0, 1].set_ylabel("Flip Rate")
    axes[0, 1].set_title("Robustness: Flip Rates by Attack")
    axes[0, 1].set_xticks(range(len(robustness_table)))
    axes[0, 1].set_xticklabels(robustness_table["attack"], rotation=25, ha="right")

    # (2,1) Top Features
    top_k = 15
    top_idx = np.argsort(np.abs(coefs))[-top_k:]
    top_feats = feature_names[top_idx]
    top_vals = coefs[top_idx]
    order = np.argsort(top_vals)
    axes[1, 0].barh(top_feats[order], top_vals[order])
    axes[1, 0].set_title("Top 15 Model Features (Coef Magnitude)")
    axes[1, 0].set_xlabel("Coefficient")
    axes[1, 0].invert_yaxis()

    # (2,2) Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    im = axes[1, 1].imshow(cm, interpolation="nearest")
    axes[1, 1].set_title("Confusion Matrix (Production Model)")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("Actual")
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 1].text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    dashboard_path = os.path.join(OUTPUT_DIR, "dashboard_summary.png")
    fig.savefig(dashboard_path, bbox_inches="tight")
    plt.close(fig)
