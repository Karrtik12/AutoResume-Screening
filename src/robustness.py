import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import RANDOM_STATE, OUTPUT_DIR

def attack_keyword_stuffing(text: str, keywords: list, repeat: int = 3) -> str:
    stuffing = " ".join(list(keywords) * repeat)
    return text + " " + stuffing

def attack_random_typos(text: str, prob: float = 0.02) -> str:
    chars = list(text)
    for i in range(len(chars)):
        if np.random.rand() < prob and chars[i].isalpha():
            chars.insert(i, chars[i])
    return "".join(chars)

def attack_template_header_footer(text: str) -> str:
    header = "Highly motivated candidate seeking challenging role. "
    footer = " Proven track record of excellence across multiple domains."
    return header + text + footer

def attack_synonym_replacement(text: str) -> str:
    mapping = {
        "good": "excellent",
        "great": "outstanding",
        "hardworking": "diligent",
        "team": "group",
        "leader": "lead",
    }
    words = text.split()
    new_words = [mapping.get(w.lower(), w) for w in words]
    return " ".join(new_words)

def attack_sentence_shuffle(text: str) -> str:
    sentences = [s.strip() for s in text.split(".") if s.strip() != ""]
    if len(sentences) <= 1:
        return text
    np.random.shuffle(sentences)
    return ". ".join(sentences) + "."

def attack_adversarial_insert(text: str) -> str:
    adv = (
        " consistently rated top performer with strong problem-solving, "
        "stakeholder management, and leadership skills "
    )
    mid = len(text) // 2
    return text[:mid] + adv + text[mid:]

def evaluate_attack(
    model,
    feature_transformer,
    X_text_list: list,
    df_test: pd.DataFrame,
    attack_fn,
    attack_name: str,
    **kwargs
) -> dict:
    """Evaluates adversarial attack flip rates on initially rejected candidate profiles."""
    if hasattr(feature_transformer, "transform"):
        X_base = feature_transformer.transform(pd.Series(X_text_list), df_test)
    else:
        X_base = feature_transformer.transform(X_text_list)

    base_pred = model.predict(X_base)
    rejected_mask = (base_pred == 0)
    rejected_indices = np.where(rejected_mask)[0]

    if len(rejected_indices) == 0:
        return {"attack": attack_name, "flip_rate": 0.0, "flips": 0, "total": 0}

    rejected_texts = [X_text_list[i] for i in rejected_indices]
    adv_texts = [attack_fn(t, **kwargs) for t in rejected_texts]

    df_rejected = df_test.iloc[rejected_indices]

    if hasattr(feature_transformer, "transform"):
        X_adv = feature_transformer.transform(pd.Series(adv_texts), df_rejected)
    else:
        X_adv = feature_transformer.transform(adv_texts)

    adv_pred = model.predict(X_adv)
    flips = int(np.sum(adv_pred == 1))
    flip_rate = flips / len(adv_pred)

    return {
        "attack": attack_name,
        "flip_rate": flip_rate,
        "flips": flips,
        "total": len(adv_pred),
    }

def run_adversarial_attack_suite(
    model,
    feature_transformer,
    X_text_list: list,
    df_test: pd.DataFrame,
    top_keywords: list
) -> pd.DataFrame:
    """Runs standard suite of 6 adversarial attacks and returns summary DataFrame."""
    np.random.seed(RANDOM_STATE)
    attacks = [
        ("Keyword stuffing",       attack_keyword_stuffing,       {"keywords": top_keywords, "repeat": 3}),
        ("Random typos",           attack_random_typos,           {"prob": 0.02}),
        ("Template header/footer", attack_template_header_footer, {}),
        ("Synonym replacement",    attack_synonym_replacement,    {}),
        ("Sentence shuffle",       attack_sentence_shuffle,       {}),
        ("Adversarial insert",     attack_adversarial_insert,     {}),
    ]

    results = []
    for name, fn, kw in attacks:
        res = evaluate_attack(
            model=model,
            feature_transformer=feature_transformer,
            X_text_list=X_text_list,
            df_test=df_test,
            attack_fn=fn,
            attack_name=name,
            **kw
        )
        results.append(res)

    res_df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "robustness_report.csv")
    res_df.to_csv(csv_path, index=False)
    return res_df
