# -*- coding: utf-8 -*-
"""
ML/predictor.py
===============
Model registry, artifact loader, and multi-target CatBoost prediction engine for NutraVision.
"""

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
if str(_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_DIR.parent))

try:
    from ML.schemas import (
        IngredientIn, NutrientResult, RecipeTotals, PredictResponse,
        VisionNutrientResult, VisionPredictResponse
    )
    from ML.preprocessor import (
        parse_weight, detect_category, detect_attributes, resolve_preparation,
        build_catboost_pool
    )
except ModuleNotFoundError:
    from schemas import (
        IngredientIn, NutrientResult, RecipeTotals, PredictResponse,
        VisionNutrientResult, VisionPredictResponse
    )
    from preprocessor import (
        parse_weight, detect_category, detect_attributes, resolve_preparation,
        build_catboost_pool
    )

TARGETS = ["energy_kcal", "carb_g", "protein_g", "fat_g", "fibre_g"]
CAT_FEATURES = ["category", "preparation"]


class NutritionPredictor:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        self.svd = joblib.load(model_dir / "svd.joblib")
        self.attribute_keywords = joblib.load(model_dir / "attribute_keywords.joblib")
        self.attribute_names = joblib.load(model_dir / "attribute_names.joblib")
        self.category_keywords = joblib.load(model_dir / "category_keywords.joblib")
        self.preparation_keywords = joblib.load(model_dir / "preparation_keywords.joblib")
        self.metadata = joblib.load(model_dir / "metadata.joblib")

        self.attr_cols = [f"attr_{a}" for a in self.attribute_names]

        self.catboost_models: Dict[str, CatBoostRegressor] = {}
        for target in TARGETS:
            cb = CatBoostRegressor()
            cb.load_model(str(model_dir / f"{target}.cbm"))
            self.catboost_models[target] = cb

    def predict_ingredients(self, ingredients: List[IngredientIn]) -> PredictResponse:
        results: List[NutrientResult] = []
        totals = {t: 0.0 for t in TARGETS}
        total_weight = 0.0

        for item in ingredients:
            weight_g, food_name = parse_weight(item.name)
            if item.weight_g is not None:
                weight_g = item.weight_g

            preparation = item.preparation or resolve_preparation(None, food_name, self.preparation_keywords)
            category    = item.category    or detect_category(item.name, self.category_keywords)
            attributes  = detect_attributes(item.name, self.attribute_names, self.attribute_keywords)

            try:
                pool = build_catboost_pool(
                    food_name, category, preparation, attributes,
                    self.vectorizer, self.svd, self.attr_cols, CAT_FEATURES
                )
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Feature extraction failed for '{item.name}': {exc}")

            preds_per100g: Dict[str, float] = {}
            for target, model in self.catboost_models.items():
                try:
                    val = float(model.predict(pool)[0])
                    preds_per100g[target] = max(val, 0.0)
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Model error for '{target}': {exc}")

            scale = weight_g / 100.0
            scaled = {t: round(v * scale, 2) for t, v in preds_per100g.items()}

            results.append(NutrientResult(
                entered_item=item.name,
                parsed_food=food_name,
                weight_g=weight_g,
                category=category,
                preparation=preparation,
                **scaled,
            ))

            total_weight += weight_g
            for t in TARGETS:
                totals[t] += scaled[t]

        return PredictResponse(
            ingredients=results,
            recipe_totals=RecipeTotals(
                total_weight_g=round(total_weight, 2),
                **{t: round(v, 2) for t, v in totals.items()},
            ),
        )

    def predict_vision_pairs(self, dish: Optional[str], pairs: List[Tuple[str, float]]) -> VisionPredictResponse:
        results: List[VisionNutrientResult] = []
        totals = {t: 0.0 for t in TARGETS}
        total_weight = 0.0

        for food_name, weight_g in pairs:
            preparation = resolve_preparation(dish, food_name, self.preparation_keywords)
            category    = detect_category(food_name, self.category_keywords)
            attributes  = detect_attributes(food_name, self.attribute_names, self.attribute_keywords)

            try:
                pool = build_catboost_pool(
                    food_name, category, preparation, attributes,
                    self.vectorizer, self.svd, self.attr_cols, CAT_FEATURES
                )
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Feature error for '{food_name}': {exc}")

            preds_per100g: Dict[str, float] = {}
            for target, model in self.catboost_models.items():
                try:
                    val = float(model.predict(pool)[0])
                    preds_per100g[target] = max(val, 0.0)
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"Model error for '{target}': {exc}")

            scale  = weight_g / 100.0
            scaled = {t: round(v * scale, 2) for t, v in preds_per100g.items()}

            results.append(VisionNutrientResult(
                ingredient=food_name,
                weight_g=weight_g,
                **scaled,
            ))

            total_weight += weight_g
            for t in TARGETS:
                totals[t] += scaled[t]

        return VisionPredictResponse(
            dish=dish,
            nutrition=results,
            recipe_totals=RecipeTotals(
                total_weight_g=round(total_weight, 2),
                **{t: round(v, 2) for t, v in totals.items()},
            ),
        )
