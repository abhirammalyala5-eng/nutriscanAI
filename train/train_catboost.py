# -*- coding: utf-8 -*-
"""
train_catboost.py - Reproducible CatBoost Nutrition Training Pipeline
======================================================================
Trains the 5 CatBoost regression models and associated preprocessing
artifacts (TF-IDF vectorizer, TruncatedSVD, keyword dictionaries)
expected by the NutraVision inference service (ML/app.py).

Targets:
  - energy_kcal
  - carb_g
  - protein_g
  - fat_g
  - fibre_g

Features (75 total):
  - text_0 .. text_63: TruncatedSVD(64) on char_wb n-gram TF-IDF (2-5)
  - category: Categorical feature (12 food groups + unknown)
  - preparation: Categorical feature (8 cooking methods + unknown)
  - attr_skin_on .. attr_lean: 9 binary food attributes
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Default Constants & Keywords Matching Inference Schema
# ---------------------------------------------------------------------------
TARGETS = ["energy_kcal", "carb_g", "protein_g", "fat_g", "fibre_g"]

PREPARATION_KEYWORDS: Dict[str, List[str]] = {
    "baked": ["baked"],
    "boiled": ["boiled"],
    "dried": ["dried"],
    "fried": ["fried"],
    "grilled": ["grilled", "barbecued"],
    "raw": ["raw", "uncooked"],
    "roasted": ["roasted"],
    "steamed": ["steamed"],
    "unknown": [],
}

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "dairy": ["milk", "cheese", "yogurt", "curd", "butter", "cream", "paneer"],
    "egg": ["egg"],
    "fruit": ["apple", "banana", "mango", "orange", "grape", "papaya", "pineapple", "berry"],
    "grain": ["rice", "wheat", "oat", "barley", "millet", "bread", "flour", "pasta"],
    "legume": ["lentil", "dal", "bean", "chickpea", "soybean", "pea"],
    "meat": ["mutton", "lamb", "beef", "pork", "goat", "meat"],
    "nut": ["almond", "cashew", "walnut", "peanut", "pistachio", "hazelnut"],
    "oil": ["oil", "ghee"],
    "poultry": ["chicken", "turkey", "duck", "quail"],
    "seafood": ["fish", "salmon", "tuna", "prawn", "shrimp", "crab", "lobster", "sardine"],
    "sweetener": ["sugar", "honey", "syrup", "jaggery"],
    "vegetable": [
        "broccoli", "potato", "tomato", "onion", "carrot", "spinach",
        "cabbage", "chilli", "cauliflower", "cucumber", "peas",
    ],
}

ATTRIBUTE_KEYWORDS: Dict[str, List[str]] = {
    "skin_on": ["skin-on", "with skin"],
    "skinless": ["skinless"],
    "boneless": ["boneless"],
    "bone_in": ["bone-in", "with bone"],
    "peeled": ["peeled"],
    "whole": ["whole"],
    "sliced": ["sliced"],
    "chopped": ["chopped"],
    "lean": ["lean"],
}
ATTRIBUTE_NAMES: List[str] = list(ATTRIBUTE_KEYWORDS.keys())
ATTR_COLS: List[str] = [f"attr_{a}" for a in ATTRIBUTE_NAMES]
CAT_FEATURES: List[str] = ["category", "preparation"]


def detect_preparation(text: str) -> str:
    text_l = text.lower()
    for prep, keywords in PREPARATION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_l:
                return prep
    return "unknown"


def detect_category(text: str) -> str:
    text_l = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_l:
                return cat
    return "unknown"


def detect_attributes(text: str) -> List[int]:
    text_l = text.lower()
    flags = []
    for attr in ATTRIBUTE_NAMES:
        kws = ATTRIBUTE_KEYWORDS.get(attr, [])
        flags.append(1 if any(kw in text_l for kw in kws) else 0)
    return flags


def clean_food_name(text: str) -> str:
    # Strip weight patterns if present
    cleaned = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:g|grams?|kg|kilograms?|ml|millilitres?|oz|ounces?|lb|lbs?|pounds?)\b",
        "",
        str(text),
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ,-")


def build_features(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    svd: TruncatedSVD,
    is_training: bool = False,
) -> Tuple[pd.DataFrame, TfidfVectorizer, TruncatedSVD]:
    """Transform raw ingredients dataframe into the 75 CatBoost features."""
    names = df["cleaned_name"].tolist()

    if is_training:
        tfidf_mat = vectorizer.fit_transform(names)
        svd_mat = svd.fit_transform(tfidf_mat)
    else:
        tfidf_mat = vectorizer.transform(names)
        svd_mat = svd.transform(tfidf_mat)

    # 64 text SVD features
    svd_df = pd.DataFrame(svd_mat, columns=[f"text_{i}" for i in range(svd.n_components)], index=df.index)

    # Categorical and attribute features
    categories = df["cleaned_name"].apply(detect_category)
    preparations = df["cleaned_name"].apply(detect_preparation)
    attrs = np.array([detect_attributes(n) for n in names])
    attr_df = pd.DataFrame(attrs, columns=ATTR_COLS, index=df.index)

    feat_df = pd.concat([
        svd_df,
        pd.DataFrame({"category": categories, "preparation": preparations}, index=df.index),
        attr_df,
    ], axis=1)

    return feat_df, vectorizer, svd


def train_pipeline(
    data_path: Path,
    export_dir: Path,
    test_size: float = 0.10,
    iterations: int = 1500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run full training pipeline and export artifacts."""
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path).dropna(subset=["ingredient_name"]).copy()
    print(f"Loaded {len(df)} records.")

    df["cleaned_name"] = df["ingredient_name"].apply(clean_food_name)

    # Clean targets: ensure numeric, replace NaNs with 0
    for col in TARGETS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 90/10 Train-Test Split matching historical evaluation
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)
    print(f"Train set: {len(train_df)} rows | Test set: {len(test_df)} rows")

    # Fit TF-IDF and SVD on training set only (prevent data leakage)
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=30000,
        sublinear_tf=True,
    )
    svd = TruncatedSVD(n_components=64, random_state=random_seed)

    print("Extracting features on training split...")
    X_train, vectorizer, svd = build_features(train_df, vectorizer, svd, is_training=True)
    print("Extracting features on test split...")
    X_test, _, _ = build_features(test_df, vectorizer, svd, is_training=False)

    train_pool = Pool(X_train, cat_features=CAT_FEATURES)
    test_pool = Pool(X_test, cat_features=CAT_FEATURES)

    models: Dict[str, CatBoostRegressor] = {}
    eval_records = []

    export_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        print(f"\nTraining CatBoostRegressor for: {target} (depth=8, lr=0.03, l2=5)...")
        cb = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=5,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=random_seed,
            verbose=250,
        )
        cb.fit(
            X_train,
            train_df[target],
            eval_set=(X_test, test_df[target]),
            cat_features=CAT_FEATURES,
            early_stopping_rounds=100,
        )

        preds = cb.predict(X_test)
        preds = np.maximum(preds, 0.0)  # clamp negatives

        actual = test_df[target].values
        mae = mean_absolute_error(actual, preds)
        rmse = float(np.sqrt(mean_squared_error(actual, preds)))
        r2 = r2_score(actual, preds)

        print(f"[{target}] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")
        eval_records.append({
            "nutrient": target,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "test_rows": len(test_df),
        })

        model_path = export_dir / f"{target}.cbm"
        cb.save_model(str(model_path))
        print(f"Saved: {model_path}")
        models[target] = cb

    # Save preprocessing joblib artifacts
    print("\nSaving preprocessing joblib artifacts...")
    joblib.dump(vectorizer, export_dir / "vectorizer.joblib")
    joblib.dump(svd, export_dir / "svd.joblib")
    joblib.dump(CATEGORY_KEYWORDS, export_dir / "category_keywords.joblib")
    joblib.dump(PREPARATION_KEYWORDS, export_dir / "preparation_keywords.joblib")
    joblib.dump(ATTRIBUTE_KEYWORDS, export_dir / "attribute_keywords.joblib")
    joblib.dump(ATTRIBUTE_NAMES, export_dir / "attribute_names.joblib")

    metadata = {
        "model_type": "CatBoostRegressor",
        "targets": TARGETS,
        "categorical_features": CAT_FEATURES,
        "input_schema": ["food_name", "category", "preparation", "attributes", "weight_g"],
        "prediction_unit": "per_100g",
    }
    joblib.dump(metadata, export_dir / "metadata.joblib")

    eval_df = pd.DataFrame(eval_records)
    eval_df.to_csv(export_dir / "evaluation.csv", index=False)
    print("\nTraining Complete! Evaluation Results:")
    print(eval_df.to_string(index=False))
    return eval_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NutraVision CatBoost Models")
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).parent / "data" / "model1_dataset.csv"),
        help="Path to training CSV",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "ML"),
        help="Directory to save .cbm and .joblib artifacts",
    )
    parser.add_argument("--iterations", type=int, default=1500, help="Number of trees")
    parser.add_argument("--quick", action="store_true", help="Quick run with 50 iterations for testing")

    args = parser.parse_args()
    iters = 50 if args.quick else args.iterations
    train_pipeline(Path(args.data), Path(args.export_dir), iterations=iters)

