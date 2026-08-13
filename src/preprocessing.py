import re
import numpy as np
import pandas as pd
import gender_guesser.detector as gender
from src.config import DATA_PATH, LABEL_COL, NAME_COL, TEXT_COLS

detector = gender.Detector(case_sensitive=False)

def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Loads dataset and ensures label column is integer type."""
    df = pd.read_csv(path)
    if LABEL_COL in df.columns:
        df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df

def infer_gender_name(name: str) -> str:
    """Infers demographic gender from candidate first name using gender_guesser."""
    if pd.isna(name) or not isinstance(name, str) or name.strip() == "":
        return "unknown"
    first_name = name.strip().split()[0]
    g = detector.get_gender(first_name)
    if g in ["male", "mostly_male"]:
        return "male"
    elif g in ["female", "mostly_female"]:
        return "female"
    else:
        return "unknown"

def filter_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Filters dataset to contain only clean male and female inferred records."""
    df = df.copy()
    df["gender_group"] = df[NAME_COL].apply(infer_gender_name)
    df = df[df["gender_group"].isin(["male", "female"])].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def combine_text_fields(df: pd.DataFrame, cols: list = None) -> pd.Series:
    """Combines unstructured resume text columns into a single text document per applicant."""
    if cols is None:
        cols = TEXT_COLS
    cols = [c for c in cols if c in df.columns]
    
    def row_combiner(row):
        parts = []
        for c in cols:
            val = row.get(c, "")
            if pd.notna(val) and str(val).strip() != "":
                parts.append(str(val))
        return " ".join(parts)

    return df.apply(row_combiner, axis=1).fillna("").astype(str)

def parse_skill_list(s) -> set:
    """Splits skill string into a set of lowercased tokens."""
    if pd.isna(s):
        return set()
    tokens = re.split(r"[,\;/\|]", str(s).lower())
    return set(t.strip() for t in tokens if t.strip() != "")

def parse_years_from_text(s) -> float:
    """Extracts numeric years from experience requirement strings."""
    if pd.isna(s):
        return np.nan
    nums = re.findall(r"\d+\.?\d*", str(s))
    if not nums:
        return np.nan
    return float(nums[0])

def degree_level(text) -> int:
    """Maps education degree text into ordinal hierarchy level (0-4)."""
    if pd.isna(text):
        return 0
    t = str(text).lower()
    if "phd" in t or "doctor" in t:
        return 4
    if "master" in t or "m.tech" in t or "mtech" in t or "m.sc" in t or "mba" in t:
        return 3
    if "bachelor" in t or "b.tech" in t or "btech" in t or "b.e" in t or "bsc" in t or "b.sc" in t:
        return 2
    if "diploma" in t:
        return 1
    return 0

def parse_age_requirement(text) -> tuple:
    """Extracts min and max age constraints from requirement text."""
    if pd.isna(text):
        return (np.nan, np.nan)
    t = str(text).lower()
    nums = re.findall(r"\d+", t)
    if ("between" in t or "-" in t) and len(nums) >= 2:
        return (float(nums[0]), float(nums[1]))
    if "below" in t or "under" in t or "upto" in t:
        return (np.nan, float(nums[0])) if nums else (np.nan, np.nan)
    if "above" in t or "over" in t:
        return (float(nums[0]), np.nan) if nums else (np.nan, np.nan)
    if len(nums) == 1:
        return (float(nums[0]), np.nan)
    return (np.nan, np.nan)

def extract_requirement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts structured requirement gaps, skill overlap ratios, and rule verdicts."""
    df = df.copy()

    # Skills overlap
    cand_skills = df["skills"].fillna("")
    req_skills = df["skills_required"].fillna("") if "skills_required" in df.columns else pd.Series([""] * len(df))

    skill_overlap_count = []
    skill_overlap_ratio_req = []
    missing_required_skill_count = []
    skills_meet = []

    for cs, rs in zip(cand_skills, req_skills):
        cs_set = parse_skill_list(cs)
        rs_set = parse_skill_list(rs)
        inter = cs_set & rs_set

        overlap = len(inter)
        missing = max(len(rs_set) - overlap, 0)
        ratio_req = overlap / len(rs_set) if len(rs_set) > 0 else 0.0
        meet_flag = 1 if missing == 0 and len(rs_set) > 0 else 0

        skill_overlap_count.append(overlap)
        skill_overlap_ratio_req.append(ratio_req)
        missing_required_skill_count.append(missing)
        skills_meet.append(meet_flag)

    df["skill_overlap_count"] = skill_overlap_count
    df["skill_overlap_ratio_req"] = skill_overlap_ratio_req
    df["missing_required_skill_count"] = missing_required_skill_count
    df["skills_meet_all_required"] = skills_meet

    # Experience Gap
    cand_exp = pd.to_numeric(df.get("years_of_experience", pd.Series()), errors="coerce").fillna(0.0)
    req_exp_raw = df.get("experiencere_requirement", pd.Series()).fillna("")
    req_exp = req_exp_raw.apply(parse_years_from_text).fillna(0.0)

    df["candidate_experience_years"] = cand_exp
    df["required_experience_years"] = req_exp
    df["experience_gap"] = df["candidate_experience_years"] - df["required_experience_years"]
    df["experience_meets"] = (df["experience_gap"] >= 0).astype(int)

    # Education Gap
    cand_edu_text = (
        df.get("educational_institution_name", pd.Series()).fillna("").astype(str)
        + " " + df.get("degree_names", pd.Series()).fillna("").astype(str)
        + " " + df.get("major_field_of_studies", pd.Series()).fillna("").astype(str)
    )
    req_edu_text = df.get("educationaL_requirements", pd.Series()).fillna("")

    df["candidate_education_level"] = cand_edu_text.apply(degree_level)
    df["required_education_level"] = req_edu_text.apply(degree_level)
    df["education_gap"] = df["candidate_education_level"] - df["required_education_level"]
    df["education_meets"] = (df["education_gap"] >= 0).astype(int)

    # Age Requirement
    age_req_text = df.get("age_requirement", pd.Series()).fillna("")
    age_mins, age_maxs = [], []
    for t in age_req_text:
        mn, mx = parse_age_requirement(t)
        age_mins.append(mn)
        age_maxs.append(mx)

    df["age_min_required"] = age_mins
    df["age_max_required"] = age_maxs
    df["has_age_requirement"] = (~df.get("age_requirement", pd.Series()).isna()).astype(int)

    # Matched Score
    df["matched_score"] = pd.to_numeric(df.get("matched_score", pd.Series()), errors="coerce").fillna(0.0)

    # Candidate Classification by Rule Engine
    ms_valid = df["matched_score"]
    ms_threshold = ms_valid.quantile(0.7) if not ms_valid.empty else 0.7

    df["high_matched_score"] = (df["matched_score"] >= ms_threshold).astype(int)
    df["meets_all_requirements"] = (
        (df["skills_meet_all_required"] == 1)
        & (df["experience_meets"] == 1)
        & (df["education_meets"] == 1)
    ).astype(int)

    df["good_candidate"] = (
        (df["meets_all_requirements"] == 1) | (df["high_matched_score"] == 1)
    ).astype(int)
    df["weak_candidate"] = 1 - df["good_candidate"]

    if LABEL_COL in df.columns:
        df["is_shortlisted"] = df[LABEL_COL].astype(int)
        df["good_but_rejected"] = ((df["good_candidate"] == 1) & (df["is_shortlisted"] == 0)).astype(int)
        df["weak_but_shortlisted"] = ((df["good_candidate"] == 0) & (df["is_shortlisted"] == 1)).astype(int)

    return df
