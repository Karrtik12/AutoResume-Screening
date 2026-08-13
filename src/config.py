import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_PATH = os.path.join(DATA_DIR, "resume_data.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "ml_pipeline_outputs")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Random Seed
RANDOM_STATE = 42

# Column names
NAME_COL = "name"
LABEL_COL = "shortlist"

TEXT_COLS = [
    "skills",
    "responsibilities.1",
    "educational_institution_name",
    "degree_names",
    "major_field_of_studies",
    "educational_results",
    "result_types",
    "career_objective",
    "responsibilities",
    "extra_curricular_activity_types",
    "languages",
    "certification_skills",
]

# TF-IDF Settings
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
