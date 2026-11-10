# -*- coding: utf-8 -*-
"""
ML/schemas.py
=============
Pydantic data schemas for NutraVision Nutritional Regression API requests and responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IngredientIn(BaseModel):
    name: str = Field(
        ...,
        description='Free-text ingredient, e.g. "150g grilled chicken" or just "apple"',
        examples=["150g grilled chicken", "50g basmati rice", "apple"],
    )
    weight_g: Optional[float] = Field(
        None,
        description="Explicit weight override in grams (overrides any amount parsed from name)",
        examples=[100, 250],
    )
    category: Optional[str] = Field(
        None,
        description="Food category hint (poultry/meat/seafood/grain/vegetable/fruit/dairy/egg/legume/nut)",
    )
    preparation: Optional[str] = Field(
        None,
        description="Preparation hint (raw/boiled/grilled/roasted/fried/steamed/baked/dried/unknown)",
    )


class NutrientResult(BaseModel):
    entered_item:  str
    parsed_food:   str
    weight_g:      float
    category:      str
    preparation:   str
    energy_kcal:   float
    carb_g:        float
    protein_g:     float
    fat_g:         float
    fibre_g:       float


class RecipeTotals(BaseModel):
    total_weight_g: float
    energy_kcal:    float
    carb_g:         float
    protein_g:      float
    fat_g:          float
    fibre_g:        float


class PredictRequest(BaseModel):
    ingredients: List[IngredientIn] = Field(
        ...,
        description="List of ingredients to analyse",
        min_length=1,
    )


class PredictResponse(BaseModel):
    ingredients:   List[NutrientResult]
    recipe_totals: RecipeTotals


class VisionRequest(BaseModel):
    dish:        Optional[str]                   = None
    ingredients: Optional[Dict[str, Any]]        = None
    estimates:   Optional[List[Dict[str, Any]]]  = None
    model_config = {"extra": "allow"}              # allow unknown fields from vision service


class VisionNutrientResult(BaseModel):
    ingredient:  str
    weight_g:    float
    energy_kcal: float
    carb_g:      float
    protein_g:   float
    fat_g:       float
    fibre_g:     float


class VisionPredictResponse(BaseModel):
    dish:          Optional[str]
    nutrition:     List[VisionNutrientResult]
    recipe_totals: RecipeTotals
