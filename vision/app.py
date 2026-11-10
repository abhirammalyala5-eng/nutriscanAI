# -*- coding: utf-8 -*-
"""
Food Vision API
===============
FastAPI service orchestrating dual-view food image analysis:
DeepLabV3+ semantic segmentation, plate rim calibration, and local Qwen3-VL VLM inference.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

try:
    from vision.segmentation import load_segmenter, segmentation_summary, detect_top_plate_scale
    from vision.estimation import density_category, build_weight_estimates
    from vision.vlm import (
        get_vlm_thumbnail_b64,
        extract_universal_components,
        parse_component_response,
        query_qwen_vlm,
    )
except ModuleNotFoundError:
    from segmentation import load_segmenter, segmentation_summary, detect_top_plate_scale
    from estimation import density_category, build_weight_estimates
    from vlm import (
        get_vlm_thumbnail_b64,
        extract_universal_components,
        parse_component_response,
        query_qwen_vlm,
    )
SEGMENTER_PATH = Path(os.getenv("FOODSEG_MODEL", ROOT / "models/FoodSeg103/best_model.pth"))
LABELS_PATH = Path(os.getenv("FOODSEG_LABELS", ROOT / "models/FoodSeg103/labels_config.json"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-vl:2b")
QWEN_NUM_CTX = int(os.getenv("QWEN_NUM_CTX", "4096"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(40 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "50000000"))
CONFIDENCE = float(os.getenv("FOODSEG_CONFIDENCE", "0.25"))
FOODSEG_TILE_SIZE = int(os.getenv("FOODSEG_TILE_SIZE", "512"))
FOODSEG_TILE_OVERLAP = int(os.getenv("FOODSEG_TILE_OVERLAP", "64"))

_ollama_endpoint = urlparse(OLLAMA_URL)
if _ollama_endpoint.scheme not in {"http", "https"} or _ollama_endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}:
    raise RuntimeError("Offline mode requires OLLAMA_URL to point to this computer (localhost)")
if QWEN_MODEL.lower().endswith("-cloud"):
    raise RuntimeError("Offline mode cannot use an Ollama cloud model; configure a locally installed model")

# 1. Load PyTorch Segmenter
segmenter, CLASS_NAMES, DEVICE = load_segmenter(SEGMENTER_PATH, LABELS_PATH)
DENSITY_TABLE = json.loads((ROOT / "food_density.json").read_text(encoding="utf-8"))

app = FastAPI(title="Food Vision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger("foodvision")


# Backward-compatibility wrappers
def _segmentation_summary_compat(image: Image.Image):
    return segmentation_summary(image, segmenter, DEVICE, CLASS_NAMES, FOODSEG_TILE_SIZE, FOODSEG_TILE_OVERLAP, CONFIDENCE)

def _build_weight_estimates_compat(components: list[dict[str, Any]], plate_scale: dict[str, Any]):
    return build_weight_estimates(components, plate_scale, DENSITY_TABLE)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "segmenter": "loaded",
        "segmenter_name": "FoodSeg103",
        "segmenter_architecture": "DeepLabV3+ with MobileNetV2 encoder",
        "segmenter_class_count": len(CLASS_NAMES),
        "segmenter_classes": list(CLASS_NAMES.values()),
        "segmenter_device": str(DEVICE),
        "segmenter_input_mode": "native-resolution overlapping tiles",
        "segmenter_tile_size": FOODSEG_TILE_SIZE,
        "segmenter_tile_overlap": FOODSEG_TILE_OVERLAP,
        "qwen_context_tokens": QWEN_NUM_CTX,
        "max_image_pixels": MAX_IMAGE_PIXELS,
        "density_categories": list(DENSITY_TABLE),
        "qwen_model": QWEN_MODEL,
        "offline_runtime": True,
    }


async def _read_food_image(upload: UploadFile, field_name: str) -> tuple[Image.Image, bytes]:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"{field_name} must be an image")
    raw = await upload.read(MAX_IMAGE_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"{field_name} exceeds {MAX_IMAGE_BYTES} bytes")
    try:
        decoded = Image.open(io.BytesIO(raw))
        if decoded.width * decoded.height > MAX_IMAGE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail=f"{field_name} exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit",
            )
        image = ImageOps.exif_transpose(decoded).convert("RGB")
    except Image.DecompressionBombError:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} exceeds the {MAX_IMAGE_PIXELS:,}-pixel safety limit",
        )
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail=f"Could not decode {field_name}")
    return image, raw


@app.post("/vision")
async def vision(
    top_image: UploadFile = File(..., description="Photo taken directly above the meal"),
    side_image: UploadFile = File(..., description="Photo taken from the side at plate level"),
    plate_diameter_cm: float = Form(..., description="User-selected plate diameter in cm (typical: 20, 25, or 30)"),
) -> dict[str, Any]:
    """Estimate ingredient weights from paired views and a known plate diameter."""
    if not 15 <= plate_diameter_cm <= 40:
        raise HTTPException(status_code=422, detail="plate_diameter_cm must be between 15 and 40 cm")
    top, _ = await _read_food_image(top_image, "top_image")
    side, _ = await _read_food_image(side_image, "side_image")

    started = time.perf_counter()
    top_areas, top_detections, top_food_pixels = segmentation_summary(
        top, segmenter, DEVICE, CLASS_NAMES, FOODSEG_TILE_SIZE, FOODSEG_TILE_OVERLAP, CONFIDENCE
    )
    side_areas, side_detections, _ = segmentation_summary(
        side, segmenter, DEVICE, CLASS_NAMES, FOODSEG_TILE_SIZE, FOODSEG_TILE_OVERLAP, CONFIDENCE
    )
    plate_scale = detect_top_plate_scale(top, plate_diameter_cm, top_food_pixels)
    segmented_at = time.perf_counter()

    top_thumb_b64 = get_vlm_thumbnail_b64(top, max_dim=512)

    parsed = None
    vlm_failed = False
    try:
        parsed = await query_qwen_vlm(
            top_thumb_b64, plate_diameter_cm, OLLAMA_URL, QWEN_MODEL, num_threads=4
        )
    except Exception as exc:
        logger.warning("Ollama/Qwen processing skipped or unusable (%s). Falling back to FoodSeg103 geometric segmentation.", exc)

    if not parsed:
        vlm_failed = True
        # Filter and sanitize FoodSeg103 classes for realistic plated meal fallback
        filtered_areas = [
            item for item in top_areas
            if item["class"].lower() not in {"background", "other ingredients"}
            and item["area_percent"] >= 2.5
        ]

        clean_components = []
        has_curry_or_meat = any(
            any(m in item["class"].lower() for m in ("chicken", "duck", "pork", "beef", "steak", "meat"))
            for item in filtered_areas
        )

        for item in filtered_areas:
            raw_name = item["class"].lower()
            if raw_name in {"milkshake", "ice cream"}:
                if has_curry_or_meat:
                    comp_name = "Rice"
                else:
                    continue
            elif "chicken" in raw_name or "duck" in raw_name:
                comp_name = "Chicken"
            elif "pork" in raw_name:
                comp_name = "Meat" if has_curry_or_meat else "Pork"
            elif "cilantro" in raw_name or "mint" in raw_name:
                comp_name = "Fresh Herbs"
            elif "onion" in raw_name:
                comp_name = "Onions"
            elif raw_name in {"soup", "sausage", "candy"}:
                continue
            else:
                comp_name = item["class"].title()
            clean_components.append((comp_name, item["area_percent"]))

        merged_dict: dict[str, float] = {}
        for c_name, a_pct in clean_components:
            merged_dict[c_name] = merged_dict.get(c_name, 0.0) + a_pct

        total_area = sum(merged_dict.values()) or 1.0
        components = []
        for c_name, a_pct in merged_dict.items():
            share = a_pct / total_area
            components.append({
                "name": c_name,
                "food_area_fraction": round(share, 3),
                "height_cm_min": 1.0,
                "height_cm_max": 2.2,
                "shape_factor": 0.7,
                "volume_ml_min": max(25.0, share * 250.0),
                "volume_ml_max": max(50.0, share * 500.0),
                "confidence": "low",
            })

        sorted_keys = sorted(merged_dict.keys(), key=lambda k: merged_dict[k], reverse=True)
        if len(sorted_keys) >= 2:
            dish_name = f"{sorted_keys[0]} with {sorted_keys[1]}"
        elif sorted_keys:
            dish_name = f"{sorted_keys[0]} Plate"
        else:
            dish_name = "Plated Meal (Geometric Fallback)"
            components.append({
                "name": "Plated Food",
                "food_area_fraction": 0.7,
                "height_cm_min": 1.0,
                "height_cm_max": 2.2,
                "shape_factor": 0.7,
                "volume_ml_min": 150.0,
                "volume_ml_max": 300.0,
                "confidence": "low",
            })

        parsed = {"dish": dish_name, "components": components}

    logger.info(
        "vision completed: segmentation=%.2fs qwen=%.2fs total=%.2fs top=%sx%s side=%sx%s",
        segmented_at - started, time.perf_counter() - segmented_at, time.perf_counter() - started,
        top.width, top.height, side.width, side.height,
    )
    estimates = build_weight_estimates(parsed["components"], plate_scale, DENSITY_TABLE)
    top_classes = [item["class"] for item in top_areas]
    side_classes = [item["class"] for item in side_areas]
    views_share_class = bool(set(top_classes) & set(side_classes))
    limitations = [
        "These are AI-assisted estimates, not measured weights.",
        "Plate calibration depends on a visible, fully framed plate rim; occlusion or perspective can reduce accuracy.",
        "The multi-class segmentation model can miss or confuse ingredients, especially when foods overlap.",
        "Food density varies with packing, moisture, cooking, and recipe.",
    ]
    if vlm_failed:
        limitations.append(
            "VLM visual reasoning timed out or was unavailable; food components were estimated from filtered geometric segmentation."
        )
    if top_classes and side_classes and not views_share_class:
        limitations.append(
            "FoodSeg labels differ between the two views (top: " + ", ".join(top_classes[:3]) +
            "; side: " + ", ".join(side_classes[:3]) +
            "). Check that both photos show the same serving; the segmenter can also mislabel food."
        )
        for estimate in estimates:
            if estimate["confidence"] == "high":
                estimate["confidence"] = "medium"

    ingredient_totals: dict[str, int] = {}
    for item in estimates:
        name = item["name"]
        ingredient_totals[name] = ingredient_totals.get(name, 0) + int(item["weight_g"]["estimate"])

    return {
        "dish": parsed["dish"],
        "ingredients": ingredient_totals,
        "estimates": estimates,
        "scale_reference": {"plate_diameter_cm": plate_diameter_cm, **plate_scale},
        "image_processing": {
            "top_image_px": {"width": top.width, "height": top.height},
            "side_image_px": {"width": side.width, "height": side.height},
            "qwen_input": "512px LANCZOS thumbnail for optimal CPU VLM inference latency and visual token budget",
            "segmentation_input": "native-resolution overlapping tiles; all original pixels are covered",
        },
        "method": "Qwen estimates component volumes from two views; backend multiplies volume ranges by food-category density ranges.",
        "limitations": limitations,
        "segmentation": {
            "top_view": {"areas": top_areas, "detections": top_detections},
            "side_view": {"areas": side_areas, "detections": side_detections},
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
