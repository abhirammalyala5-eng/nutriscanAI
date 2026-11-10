# -*- coding: utf-8 -*-
"""
pipeline.py - NutraVision CLI Orchestrator
===========================================
End-to-end command-line runner connecting:
  1. Vision Service (port 8001 / PsTemp)
  2. Nutrition Service (port 8000 / ML)

Usage:
  python pipeline.py --top path/to/top.jpg --side path/to/side.jpg --diameter 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nutravision_cli")

DEFAULT_VISION_URL = os.getenv("VISION_URL", "http://127.0.0.1:8001/vision")
DEFAULT_NUTRITION_URL = os.getenv("NUTRITION_URL", "http://127.0.0.1:8000/predict/vision")


def run_pipeline(
    top_path: str | Path,
    side_path: Optional[str | Path] = None,
    plate_diameter_cm: float = 25.0,
    vision_url: str = DEFAULT_VISION_URL,
    nutrition_url: str = DEFAULT_NUTRITION_URL,
) -> Dict[str, Any]:
    """Execute the end-to-end vision -> nutrition workflow."""
    top_file = Path(top_path)
    if not top_file.is_file():
        raise FileNotFoundError(f"Top-view image not found: {top_file}")

    side_file = Path(side_path) if side_path else top_file
    if not side_file.is_file():
        logger.warning("Side image not found at %s. Falling back to top image.", side_file)
        side_file = top_file

    logger.info("Step 1: Sending images to Vision Service (%s)...", vision_url)
    with open(top_file, "rb") as f_top, open(side_file, "rb") as f_side:
        files = {
            "top_image": (top_file.name, f_top, "image/jpeg"),
            "side_image": (side_file.name, f_side, "image/jpeg"),
        }
        data = {"plate_diameter_cm": str(plate_diameter_cm)}

        try:
            v_resp = requests.post(vision_url, files=files, data=data, timeout=240)
            v_resp.raise_for_status()
            vision_result = v_resp.json()
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Vision Service at %s. Ensure PsTemp is running on port 8001.", vision_url)
            raise
        except Exception as e:
            logger.error("Vision request failed: %s", e)
            raise

    dish = vision_result.get("dish", "Detected Meal")
    ingredients = vision_result.get("ingredients", {})
    logger.info("Vision Result: Dish='%s', Ingredients=%s", dish, ingredients)

    if not ingredients:
        logger.warning("Vision service returned no ingredients.")
        return {
            "dish": dish,
            "ingredients": {},
            "nutrition": [],
            "recipe_totals": {},
            "vision_metadata": vision_result,
        }

    logger.info("Step 2: Sending {dish, ingredients} to Nutrition Service (%s)...", nutrition_url)
    ml_payload = {
        "dish": dish,
        "ingredients": ingredients,
    }

    try:
        n_resp = requests.post(nutrition_url, json=ml_payload, timeout=30)
        n_resp.raise_for_status()
        nutrition_result = n_resp.json()
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to Nutrition Service at %s. Ensure ML is running on port 8000.", nutrition_url)
        raise
    except Exception as e:
        logger.error("Nutrition request failed: %s", e)
        raise

    logger.info("Step 3: Merging results...")
    return {
        "dish": dish,
        "ingredients": ingredients,
        "estimates": vision_result.get("estimates", []),
        "scale_reference": vision_result.get("scale_reference", {}),
        "nutrition": nutrition_result.get("nutrition", []),
        "recipe_totals": nutrition_result.get("recipe_totals", {}),
        "limitations": vision_result.get("limitations", []),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NutraVision End-to-End CLI Pipeline")
    parser.add_argument("--top", "-t", required=False, default=None, help="Path to top-view image")
    parser.add_argument("--side", "-s", required=False, default=None, help="Path to side-view image")
    parser.add_argument("--diameter", "-d", type=float, default=25.0, help="Plate diameter in cm (default: 25)")
    parser.add_argument("--vision-url", default=DEFAULT_VISION_URL, help="Vision API URL")
    parser.add_argument("--nutrition-url", default=DEFAULT_NUTRITION_URL, help="Nutrition API URL")

    args = parser.parse_args()

    # Default to sample image if none provided
    top_arg = args.top
    if not top_arg:
        sample = Path(__file__).parent / "FrontEnd" / "images" / "sample_meal_top.jpg"
        if sample.is_file():
            top_arg = str(sample)
        else:
            top_arg = str(Path(__file__).parent / "FrontEnd" / "images" / "logo.png.jpeg")

    print(f"\n=======================================================")
    print(f"  NUTRAVISION CLI: {top_arg}")
    print(f"  Plate Diameter: {args.diameter} cm")
    print(f"=======================================================\n")

    try:
        output = run_pipeline(
            top_path=top_arg,
            side_path=args.side,
            plate_diameter_cm=args.diameter,
            vision_url=args.vision_url,
            nutrition_url=args.nutrition_url,
        )
        print("\n--- FINAL PREDICTION RESULT ---")
        print(f"🍛 Dish: {output.get('dish')}")
        totals = output.get("recipe_totals", {})
        print(f"⚡ Total Energy:   {totals.get('energy_kcal', 0)} kcal (Weight: {totals.get('total_weight_g', 0)}g)")
        print(f"🍞 Carbohydrates:  {totals.get('carb_g', 0)} g")
        print(f"🥩 Protein:        {totals.get('protein_g', 0)} g")
        print(f"🧈 Fat:            {totals.get('fat_g', 0)} g")
        print(f"🌾 Fibre:          {totals.get('fibre_g', 0)} g\n")

        print("📋 Per-Ingredient Breakdown:")
        for item in output.get("nutrition", []):
            print(f"  * {item.get('ingredient')}: {item.get('weight_g')}g -> {item.get('energy_kcal')} kcal | C:{item.get('carb_g')}g P:{item.get('protein_g')}g F:{item.get('fat_g')}g")

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)
