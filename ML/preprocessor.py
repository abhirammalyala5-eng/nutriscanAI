# -*- coding: utf-8 -*-
"""
ML/preprocessor.py
==================
Feature engineering, text parsing, and 3-tier preparation resolution for NutraVision.
"""

import re
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from catboost import Pool

_WEIGHT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:g|grams?|kg|kilograms?|ml|millilitres?|oz|ounces?|lb|lbs?|pounds?)",
    re.IGNORECASE,
)
_G_PER_UNIT = {
    "g": 1, "gram": 1, "grams": 1,
    "kg": 1000, "kilogram": 1000, "kilograms": 1000,
    "ml": 1,
    "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
    "lb": 453.59, "lbs": 453.59, "pound": 453.59, "pounds": 453.59,
}


def parse_weight(text: str) -> tuple[float, str]:
    """Return (weight_g, text_without_weight_token). Defaults to 100g if omitted."""
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(g|grams?|kg|kilograms?|ml|millilitres?|oz|ounces?|lb|lbs?|pounds?)\b",
        text, re.IGNORECASE,
    )
    if m:
        qty = float(m.group(1))
        unit = m.group(2).lower().rstrip("s") if m.group(2).lower() not in _G_PER_UNIT else m.group(2).lower()
        for key in _G_PER_UNIT:
            if m.group(2).lower().startswith(key[:3]):
                weight_g = qty * _G_PER_UNIT[key]
                break
        else:
            weight_g = qty
        cleaned = (text[: m.start()] + text[m.end():]).strip(" ,")
        return weight_g, cleaned
    return 100.0, text


def detect_preparation(text: str, prep_keywords: Dict[str, List[str]]) -> str:
    text_l = text.lower()
    for prep, keywords in prep_keywords.items():
        for kw in keywords:
            if kw in text_l:
                return prep
    return "unknown"


def detect_category(text: str, category_keywords: List[str]) -> str:
    text_l = text.lower()
    for cat in category_keywords:
        if cat in text_l:
            return cat
    return "unknown"


def detect_attributes(text: str, attr_names: List[str], attr_keywords: Dict[str, List[str]]) -> List[int]:
    text_l = text.lower()
    flags = []
    for attr in attr_names:
        kws = attr_keywords.get(attr, [])
        flags.append(1 if any(kw in text_l for kw in kws) else 0)
    return flags


# Contextual preparation mappings
DISH_PREP_CONTEXT: Dict[str, Dict[str, str]] = {
    # Indian Rice & Biryanis
    "biryani":      {"rice": "boiled", "chicken": "roasted", "mutton": "roasted",
                     "lamb": "roasted", "egg": "boiled", "onion": "fried",
                     "potato": "boiled"},
    "pulao":        {"rice": "boiled", "chicken": "boiled", "vegetable": "boiled",
                     "pea": "boiled", "paneer": "fried"},
    "khichdi":      {"rice": "boiled", "dal": "boiled", "lentil": "boiled", "moong": "boiled"},
    "bisi bele":    {"rice": "boiled", "dal": "boiled", "vegetable": "boiled"},
    "fried rice":   {"rice": "boiled", "egg": "fried", "chicken": "fried",
                     "vegetable": "fried", "onion": "fried"},

    # Indian Thalis & Combos
    "thali":        {"rice": "boiled", "phulka": "baked", "roti": "baked", "dal": "boiled",
                     "curd": "raw", "sambar": "boiled", "papad": "roasted", "salad": "raw"},

    # Indian Dals & Soups
    "dal":          {"lentil": "boiled", "dal": "boiled", "onion": "fried", "tomato": "boiled"},
    "dal makhani":  {"lentil": "boiled", "dal": "boiled", "rajma": "boiled", "butter": "raw", "cream": "raw"},
    "sambar":       {"dal": "boiled", "lentil": "boiled", "vegetable": "boiled", "drumstick": "boiled"},
    "rasam":        {"tomato": "boiled", "pepper": "boiled", "tamarind": "boiled"},
    "pappu":        {"dal": "boiled", "toor": "boiled", "tomato": "boiled", "spinach": "boiled"},

    # Indian Curries & Gravies
    "curry":        {"chicken": "boiled", "mutton": "boiled", "lamb": "boiled",
                     "paneer": "boiled", "potato": "boiled", "rice": "boiled", "fish": "boiled"},
    "palak paneer": {"spinach": "boiled", "paneer": "boiled", "cream": "raw"},
    "butter chicken":{"chicken": "roasted", "butter": "raw", "cream": "raw", "tomato": "boiled"},
    "tikka masala": {"chicken": "roasted", "paneer": "roasted", "cream": "raw"},
    "rajma":        {"rajma": "boiled", "bean": "boiled", "onion": "fried", "tomato": "boiled"},
    "chole":        {"chickpea": "boiled", "chana": "boiled", "onion": "fried"},
    "chana masala": {"chickpea": "boiled", "chana": "boiled", "onion": "fried"},
    "kofta":        {"kofta": "fried", "potato": "boiled", "paneer": "fried"},

    # Indian Dry Sabzis & Fries
    "aloo fry":     {"potato": "fried", "aloo": "fried"},
    "aloo gobi":    {"potato": "fried", "cauliflower": "fried", "gobi": "fried"},
    "poriyal":      {"beetroot": "fried", "cabbage": "fried", "beans": "fried", "coconut": "raw"},
    "fry":          {"potato": "fried", "dondakaya": "fried", "kakarakaya": "fried", "cabbage": "fried"},

    # Indian Breads
    "paratha":      {"flour": "fried", "potato": "boiled", "paneer": "raw", "gobi": "boiled"},
    "roti":         {"wheat": "baked", "flour": "baked", "ghee": "raw"},
    "phulka":       {"wheat": "baked", "flour": "baked"},
    "naan":         {"flour": "baked", "butter": "raw", "garlic": "baked"},

    # South Indian Tiffin
    "dosa":         {"potato": "boiled", "onion": "fried", "crepe": "fried"},
    "idli":         {"rice": "steamed", "dal": "steamed"},

    # Western & General
    "salad":        {"chicken": "grilled", "egg": "boiled", "tuna": "raw",
                     "vegetable": "raw", "lettuce": "raw", "tomato": "raw"},
    "sandwich":     {"chicken": "grilled", "egg": "boiled", "vegetable": "raw"},
    "burger":       {"chicken": "fried", "beef": "fried", "patty": "fried"},
    "stir fry":     {"chicken": "fried", "beef": "fried", "vegetable": "fried",
                     "noodle": "boiled", "rice": "boiled"},
    "soup":         {"chicken": "boiled", "vegetable": "boiled", "potato": "boiled"},
    "pasta":        {"pasta": "boiled", "chicken": "grilled", "vegetable": "boiled"},
    "pizza":        {"chicken": "roasted", "cheese": "raw", "vegetable": "roasted"},
}

INGREDIENT_DEFAULT_PREP: Dict[str, str] = {
    "rice": "boiled",    "basmati": "boiled", "noodle": "boiled",
    "pasta": "boiled",   "quinoa": "boiled",  "oat": "boiled",
    "bread": "baked",    "roti": "baked",     "naan": "baked",
    "chapati": "baked",  "paratha": "fried",  "phulka": "baked",
    "puri": "fried",     "bhatura": "fried",  "dosa": "fried",
    "idli": "steamed",   "upma": "boiled",    "poha": "fried",
    "khichdi": "boiled", "papad": "roasted",
    "chicken": "boiled", "mutton": "boiled",  "lamb": "boiled",
    "beef": "boiled",    "pork": "boiled",    "fish": "boiled",
    "prawn": "boiled",   "shrimp": "boiled",  "egg": "boiled",
    "tuna": "raw",       "salmon": "raw",
    "dal": "boiled",     "lentil": "boiled",  "bean": "boiled",
    "chickpea": "boiled","pea": "boiled",     "sambar": "boiled",
    "rasam": "boiled",   "rajma": "boiled",   "chana": "boiled",
    "chole": "boiled",   "toor": "boiled",    "moong": "boiled",
    "urad": "boiled",    "masoor": "boiled",
    "potato": "boiled",  "carrot": "boiled",  "spinach": "boiled",
    "onion": "boiled",   "tomato": "boiled",  "broccoli": "boiled",
    "cauliflower": "boiled", "cabbage": "boiled", "corn": "boiled",
    "aloo": "boiled",    "gobi": "boiled",    "bhindi": "fried",
    "baingan": "roasted","poriyal": "fried",
    "apple": "raw",   "banana": "raw",  "mango": "raw",
    "orange": "raw",  "grape": "raw",   "berry": "raw",
    "watermelon": "raw", "papaya": "raw", "muskmelon": "raw",
    "milk": "raw",    "cheese": "raw",  "paneer": "raw",
    "yogurt": "raw",  "curd": "raw",    "dahi": "raw",
    "raita": "raw",   "butter": "raw",  "ghee": "raw",
    "chutney": "raw", "pickle": "raw",
}


def resolve_preparation(dish: Optional[str], food_name: str, prep_keywords: Dict[str, List[str]]) -> str:
    """3-Tier Preparation Resolution: Explicit keyword > Dish context > Ingredient smart default."""
    prep = detect_preparation(food_name, prep_keywords)
    if prep != "unknown":
        return prep
    if dish:
        dish_l = dish.lower()
        food_l = food_name.lower()
        for dish_key, ingr_map in DISH_PREP_CONTEXT.items():
            if dish_key in dish_l:
                for ingr_key, d_prep in ingr_map.items():
                    if ingr_key in food_l:
                        return d_prep
    food_l = food_name.lower()
    for ingr_key, default_prep in INGREDIENT_DEFAULT_PREP.items():
        if ingr_key in food_l:
            return default_prep
    return "boiled"


def build_catboost_pool(
    food_name: str,
    category: str,
    preparation: str,
    attributes: List[int],
    vectorizer: Any,
    svd: Any,
    attr_cols: List[str],
    cat_features: List[str]
) -> Pool:
    """Build a CatBoost Pool matching the model's feature schema."""
    tfidf_vec = vectorizer.transform([food_name])
    svd_vec   = svd.transform(tfidf_vec)[0]        # shape (64,)
    text_cols = {f"text_{i}": float(svd_vec[i]) for i in range(len(svd_vec))}
    cat_col   = {"category": str(category), "preparation": str(preparation)}
    attr_col  = {col: int(val) for col, val in zip(attr_cols, attributes)}
    df = pd.DataFrame([{**text_cols, **cat_col, **attr_col}])
    return Pool(df, cat_features=cat_features)
