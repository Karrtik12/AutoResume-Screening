# Automated Resume Screening & ML Fairness Audit System 📄🔍

An end-to-end Machine Learning pipeline and evaluation framework for automated candidate resume screening, shortlisting, demographic fairness auditing, local/global interpretability, and adversarial robustness testing.

---

## 📌 Features & Highlights

* **Modular Architecture (`src/`):** Clean separation of concerns into configuration, preprocessing, hybrid feature engineering, model training, demographic auditing, adversarial robustness, explainability, and visualization.
* **Hybrid Feature Fusion (Production Model):** Combines sparse TF-IDF text features with dense, scaled candidate requirement features (experience gap, degree level mapping, skill overlap ratio, missing skill count, matched score).
* **High Accuracy & Reduced Misclassifications:** Achieves **70.6% test accuracy** (+7.5% over text-only baseline) and **0.728 F1-score**, reducing False Negative Rate (missed qualified candidates) to **23.8%**.
* **Demographic Fairness Audit:** Disaggregates predictions by demographic groups (`female` vs `male`) to evaluate selection rates, False Positive Rates (FPR), and False Negative Rates (FNR) for equalized odds compliance.
* **Adversarial Robustness Testing:** Evaluates model resilience against 6 adversarial attack strategies (keyword stuffing, random typos, template headers/footers, synonym replacement, sentence shuffling, adversarial insertions).
* **Candidate Local Explainer:** Explains individual candidate decisions down to decision margins, requirement satisfaction, positive/negative feature contributions (+/−), and demographic context.

---

## 🏗️ Project Architecture

```
AutoResume-Screening/
├── main.py                             # Main CLI pipeline entrypoint
├── main.ipynb                          # Interactive Jupyter Notebook importing from src/
├── requirements.txt                    # Pinned Python package dependencies
├── .gitignore                          # Excludes bytecode, venv/, cache, & OS files
├── DATASETS.md                         # Reference guide for external open resume datasets
├── src/                                # Modular Python Package
│   ├── __init__.py
│   ├── config.py                       # Paths, constants, and hyperparameters
│   ├── preprocessing.py                # Dataset loading, gender inference, requirement parsing
│   ├── feature_engineering.py          # TF-IDF & HybridFeatureTransformer
│   ├── models.py                       # Model initializations & cross-validation
│   ├── fairness.py                     # Demographic parity & equalized odds auditing
│   ├── robustness.py                   # Adversarial attack suite
│   ├── explainability.py               # Feature ranking & candidate local explainer
│   └── visualization.py                # Export utilities for charts & dashboards
└── data/
    ├── resume_data.csv                 # Dataset (9,544 records × 38 attributes)
    └── ml_pipeline_outputs/            # Output charts, confusion matrices, & CSV reports
```

---

## ⚙️ Installation & Virtual Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Karrtik12/AutoResume-Screening.git
   cd AutoResume-Screening
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On macOS/Linux
   # venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage & Execution

### Option 1: Run via CLI Entrypoint
Execute the complete modular pipeline from start to finish:
```bash
python main.py
```

### Option 2: Run via Interactive Notebook
Launch Jupyter Notebook or VS Code to run [main.ipynb](file:///Users/kartikayechaturvedi/Dev/AutoResume-Screening/main.ipynb):
```bash
jupyter notebook main.ipynb
```

---

## 📊 Benchmark Results

| Model | Accuracy | F1-Score | Recall | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: |
| **Hybrid Classifier (Production)** | **70.6%** | **0.728** | **76.2%** | **23.8%** |
| **Linear SVM Baseline (Text-only)** | 63.1% | 0.649 | 66.1% | 33.9% |
| **RBF SVM Benchmark** | 64.3% | 0.670 | 70.3% | 29.7% |
| **Gaussian Naive Bayes Baseline** | 58.7% | 0.544 | 47.7% | 52.3% |

* **5-Fold Stratified Cross-Validation (Hybrid Model):** F1 = $0.731 \pm 0.015$

---

## 📑 Datasets

Detailed references to additional open-source resume datasets (Kaggle & Hugging Face) can be found in [DATASETS.md](file:///Users/kartikayechaturvedi/Dev/AutoResume-Screening/DATASETS.md).