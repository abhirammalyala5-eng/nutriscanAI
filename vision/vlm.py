# -*- coding: utf-8 -*-
"""
vision/vlm.py
=============
Local Vision-Language Model (Qwen3-VL:2B) client, thread-optimized Ollama payload,
and universal regex parser for multi-component plated meals.
"""

import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List
from PIL import Image
import httpx

try:
    from vision.estimation import density_category
except ModuleNotFoundError:
    from estimation import density_category

logger = logging.getLogger("foodvision")


def get_vlm_thumbnail_b64(image: Image.Image, max_dim: int = 512) -> str:
    """Create a compact JPEG thumbnail for fast, accurate local VLM visual processing."""
    thumb = image.copy()
    thumb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def extract_universal_components(text: str) -> Dict[str, Any]:
    """Universal regex extraction for any dish and components from raw, truncated, or thinking model output."""
    dish = ""
    dish_match = re.search(r'"dish"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if not dish_match:
        dish_match = re.search(r'(?:dish is|dish name is|meal is)\s*[:"]?\s*([A-Za-z0-9\s\-/&]+?)(?:["\n,.]|components)', text, re.IGNORECASE)
    if dish_match:
        dish = dish_match.group(1).strip().title()

    comp_matches = re.findall(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"(?:fraction|food_area_fraction)"\s*:\s*([0-9.]+)\s*\}',
        text,
        re.IGNORECASE,
    )
    if not comp_matches:
        comp_matches = [
            (name, frac)
            for frac, name in re.findall(
                r'\{\s*"(?:fraction|food_area_fraction)"\s*:\s*([0-9.]+)\s*,\s*"name"\s*:\s*"([^"]+)"\s*\}',
                text,
                re.IGNORECASE,
            )
        ]
    if not comp_matches:
        bullet_matches = re.findall(r'[-*•]?\s*["\']?([A-Za-z\s]+?)["\']?\s*:\s*(?:about\s*)?(0\.\d+|\d+%)', text)
        comp_matches = [(name.strip(), val.replace('%', '') if '%' not in val else str(float(val.replace('%', ''))/100)) for name, val in bullet_matches if len(name.strip()) > 2]
    if not comp_matches:
        loose_matches = re.findall(r'([A-Za-z\s]{3,20}?)\s*(?:with fraction|\sis|\s*\(|:\s*)\s*(0\.\d+|\d+%)', text)
        comp_matches = [(name.strip(), val.replace('%', '') if '%' not in val else str(float(val.replace('%', ''))/100)) for name, val in loose_matches if len(name.strip()) > 2]

    components = []
    seen = set()
    stopwords = {"dish", "fraction", "component", "total", "schema", "first", "second", "third", "proportion", "observation", "image", "plate"}
    for name, frac_str in comp_matches:
        clean_name = name.strip().title()
        clean_lower = clean_name.lower()
        if clean_lower in seen or any(w in clean_lower for w in stopwords):
            continue
        seen.add(clean_lower)
        try:
            val = float(frac_str)
            if val > 1.0:
                val = val / 100.0
            components.append({"name": clean_name, "fraction": max(0.01, min(val, 1.0))})
        except ValueError:
            pass

    for prefix in ("The ", "A ", "An "):
        if dish.startswith(prefix):
            dish = dish[len(prefix):].strip()

    if (not dish or dish.lower() in ("curry", "food", "meal", "plated meal")) and components:
        sorted_comps = sorted(components, key=lambda c: c["fraction"], reverse=True)
        if len(sorted_comps) >= 2:
            dish = f"{sorted_comps[0]['name']} with {sorted_comps[1]['name']}"
        elif sorted_comps:
            dish = f"{sorted_comps[0]['name']} Plate"

    if not dish:
        dish = "Plated Meal"
    return {"dish": dish, "components": components}


def parse_component_response(text: str, plate_diameter_cm: float) -> Dict[str, Any]:
    """Parse JSON or reasoning stream into geometric meal components."""
    text = text.strip()
    if "</think>" in text:
        text_after_think = text.split("</think>", 1)[1].strip()
        if "{" in text_after_think:
            text = text_after_think
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    value = None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        repaired = re.sub(r',\s*\{[^{}]*$', '', text.strip())
        if repaired.count('[') > repaired.count(']'):
            repaired += ']'
        if repaired.count('{') > repaired.count('}'):
            repaired += '}'
        try:
            value = json.loads(repaired)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    value = json.loads(match.group(0))
                except Exception:
                    pass
        if not value:
            value = extract_universal_components(text)

    if not isinstance(value, dict) or not isinstance(value.get("components"), list) or not value.get("components"):
        value = extract_universal_components(text)
        if not value.get("components"):
            raise ValueError("Could not extract structured meal components from VLM output")

    dish = str(value.get("dish", "Plated Meal")).strip()
    components = []
    for item in value["components"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        try:
            name = item["name"].strip().title()
            area_val = item.get("fraction", item.get("food_area_fraction", 0.3))
            area_share = max(0.01, min(float(area_val), 1.0))

            cat = density_category(name)
            if cat == "bread":
                default_hmin, default_hmax, default_shape = 0.015, 0.035, 0.5
            elif cat == "legume_or_curry":
                default_hmin, default_hmax, default_shape = 0.05, 0.12, 0.8
            elif cat == "cooked_grain":
                default_hmin, default_hmax, default_shape = 0.06, 0.15, 0.85
            elif cat == "cooked_meat":
                default_hmin, default_hmax, default_shape = 0.04, 0.10, 0.75
            elif cat == "papad":
                default_hmin, default_hmax, default_shape = 0.01, 0.02, 0.3
            else:
                default_hmin, default_hmax, default_shape = 0.03, 0.08, 0.65

            h_min_ratio = float(item.get("height_fraction_min", default_hmin))
            h_max_ratio = float(item.get("height_fraction_max", default_hmax))
            height_fraction_min = max(0.005, min(h_min_ratio, 0.2))
            height_fraction_max = max(height_fraction_min, min(h_max_ratio, 0.2))
            height_min = height_fraction_min * plate_diameter_cm
            height_max = height_fraction_max * plate_diameter_cm

            shape_factor = max(0.2, min(float(item.get("shape_factor", default_shape)), 1.0))
            low = float(item.get("volume_ml_min", max(20.0, area_share * 180.0)))
            high = float(item.get("volume_ml_max", max(low * 1.3, area_share * 400.0)))
            if high < 5.0:
                low = max(20.0, area_share * 180.0)
                high = max(low * 1.3, area_share * 400.0)
        except (KeyError, TypeError, ValueError):
            continue
        confidence = item.get("confidence", "high")
        if confidence not in ("low", "medium", "high"):
            confidence = "high"
        components.append({"name": name, "food_area_fraction": area_share,
                           "height_cm_min": height_min, "height_cm_max": height_max,
                           "shape_factor": shape_factor, "volume_ml_min": low,
                           "volume_ml_max": high, "confidence": confidence})
    return {"dish": dish, "components": components}


async def query_qwen_vlm(
    top_thumb_b64: str,
    plate_diameter_cm: float,
    ollama_url: str,
    qwen_model: str,
    num_threads: int = 4,
) -> Dict[str, Any]:
    """Execute optimized CPU VLM chat inference via Ollama."""
    prompt = (
        f"You are a professional culinary vision AI analyzing a plated meal (plate diameter: {plate_diameter_cm:g} cm).\n"
        "Identify the exact dish name and visible meal components (e.g. Rice, Curry, Roti, Chicken, Meat, Onions, Salad) with area fractions (summing to 1.0).\n"
        "Keep your reasoning under 3 brief sentences, then immediately output the final JSON schema:\n"
        "{\n"
        '  "dish": "Exact Dish Name",\n'
        '  "components": [\n'
        '    {"name": "Food Component Name", "fraction": 0.5}\n'
        '  ]\n'
        "}\n"
    )
    payload = {
        "model": qwen_model,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [top_thumb_b64],
        }],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 1200,
            "num_ctx": 4096,
            "num_thread": num_threads,
        },
        "keep_alive": "10m",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as client:
        response = await client.post(f"{ollama_url.rstrip('/')}/api/chat", json=payload)
        response.raise_for_status()
    ollama_result = response.json()
    message = ollama_result.get("message", {})
    answer = message.get("content") or message.get("thinking", "")
    return parse_component_response(answer, plate_diameter_cm)
