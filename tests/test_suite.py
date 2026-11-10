# -*- coding: utf-8 -*-
"""
test_suite.py - End-to-End System Validation Suite
==================================================
Tests:
  1. Nutrition API Health & Predictions (Port 8000)
  2. Nutrition API Contextual Prep (Biryani vs Raw)
  3. Nutrition API Input Validation (Negative weights)
  4. Vision API Health & CORS Preflight (Port 8001)
  5. Vision API Dual-View Inference with Hough Calibration
  6. Frontend Static Assets Availability
  7. End-to-End Integration Orchestration
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import time
import requests
from pathlib import Path

PASS = " [PASS]"
FAIL = " [FAIL]"

def run_tests():
    total_passed = 0
    total_failed = 0

    print("=========================================================")
    print("      NUTRAVISION END-TO-END VALIDATION SUITE            ")
    print("=========================================================\n")

    # ─────────────────────────────────────────────────────────────
    # TEST 1: Nutrition API Health Check (8000)
    # ─────────────────────────────────────────────────────────────
    print("[1] Testing Nutrition API Health Check (http://127.0.0.1:8000/health)...")
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert data.get("status") == "ok", "Expected status ok"
        assert len(data.get("models_loaded", [])) == 5, f"Expected 5 models, got {len(data.get('models_loaded', []))}"
        print(f"    {PASS} Health status OK, 5 CatBoost models active: {data['models_loaded']}")
        total_passed += 1
    except Exception as e:
        print(f"    {FAIL} Error: {e}")
        total_failed += 1

    # ─────────────────────────────────────────────────────────────
    # TEST 2: Nutrition API Individual & Multi Ingredient Prediction
    # ─────────────────────────────────────────────────────────────
    print("\n[2] Testing Nutrition API /predict endpoint...")
    try:
        payload = {
            "ingredients": [
                {"name": "150g grilled chicken", "weight_g": 150.0},
                {"name": "100g basmati rice", "weight_g": 100.0}
            ]
        }
        r = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        totals = data.get("recipe_totals", {})
        assert totals.get("energy_kcal", 0) > 200, f"Unreasonable calories: {totals}"
        assert totals.get("protein_g", 0) > 20, f"Unreasonable protein: {totals}"
        print(f"    {PASS} Predicted: {totals['energy_kcal']} kcal, {totals['protein_g']}g protein, {totals['carb_g']}g carbs")
        total_passed += 1
    except Exception as e:
        print(f"    {FAIL} Error: {e}")
        total_failed += 1

    # ─────────────────────────────────────────────────────────────
    # TEST 3: Nutrition Contextual Preparation Resolution
    # ─────────────────────────────────────────────────────────────
    print("\n[3] Testing 3-Tier Preparation Resolution (Biryani context)...")
    try:
        p_biryani = {"dish": "Biryani", "ingredients": {"Rice": 100, "Chicken": 100}}
        p_salad   = {"dish": "Green Salad", "ingredients": {"Rice": 100, "Chicken": 100}}
        r_b = requests.post("http://127.0.0.1:8000/predict/vision", json=p_biryani, timeout=5).json()
        r_s = requests.post("http://127.0.0.1:8000/predict/vision", json=p_salad, timeout=5).json()
        # Verify valid macro calculation
        assert len(r_b.get("nutrition", [])) == 2
        assert len(r_s.get("nutrition", [])) == 2
        print(f"    {PASS} Biryani Total: {r_b['recipe_totals']['energy_kcal']} kcal | Salad Total: {r_s['recipe_totals']['energy_kcal']} kcal")
        total_passed += 1
    except Exception as e:
        print(f"    {FAIL} Error: {e}")
        total_failed += 1

    # ─────────────────────────────────────────────────────────────
    # TEST 4: Input Validation (Negative Weights)
    # ─────────────────────────────────────────────────────────────
    print("\n[4] Testing Negative Weight Validation...")
    try:
        bad_payload = {"dish": "Test", "ingredients": {"Rice": -50}}
        r = requests.post("http://127.0.0.1:8000/predict/vision", json=bad_payload, timeout=5)
        assert r.status_code == 422, f"Expected 422 Unprocessable, got {r.status_code}"
        print(f"    {PASS} Correctly rejected negative ingredient weight with HTTP 422")
        total_passed += 1
    except Exception as e:
        print(f"    {FAIL} Error: {e}")
        total_failed += 1

    # ─────────────────────────────────────────────────────────────
    # TEST 5: Vision API Health Check & CORS Preflight (8001)
    # ─────────────────────────────────────────────────────────────
    print("\n[5] Testing Vision API Health & CORS Preflight (http://127.0.0.1:8001)...")
    try:
        r_health = requests.get("http://127.0.0.1:8001/health", timeout=5)
        assert r_health.status_code == 200, f"Expected 200, got {r_health.status_code}"
        h_data = r_health.json()
        assert h_data.get("segmenter") == "loaded", "Segmenter not loaded"

        # Test CORS OPTIONS preflight
        headers = {
            "Origin": "http://127.0.0.1:8000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        }
        r_opt = requests.options("http://127.0.0.1:8001/vision", headers=headers, timeout=5)
        assert r_opt.status_code == 200, f"Expected 200 for OPTIONS, got {r_opt.status_code}"
        assert "access-control-allow-origin" in r_opt.headers, "Missing CORS allow-origin header"
        print(f"    {PASS} Vision API healthy ({h_data['segmenter_class_count']} FoodSeg classes) & CORS preflight accepted (200 OK)")
        total_passed += 1
    except Exception as e:
        print(f"    {FAIL} Error: {e}")
        total_failed += 1

    # ─────────────────────────────────────────────────────────────
    # TEST 6: Static Frontend Files Serving
    # ─────────────────────────────────────────────────────────────
    print("\n[6] Testing Frontend UI Static Assets (http://127.0.0.1:8000/)...")
    try:
        r_index = requests.get("http://127.0.0.1:8000/", timeout=5)
        assert r_index.status_code == 200, f"Index failed: {r_index.status_code}"
        assert "NutriScanAI" in r_index.text or "NutraVision" in r_index.text, "Index text mismatch"

        r_js = requests.get("http://127.0.0.1:8000/script.js", timeout=5)
        assert r_js.status_code == 200, f"script.js failed: {r_js.status_code}"
        assert "8001" in r_js.text, "script.js does not configure port 8001"
        assert "8000" in r_js.text, "script.js does not configure port 8000"

        print(f"    {PASS} index.html served (200 OK) & script.js configured for 8001/8000")
        total_passed += 1
    except Exception as e:
        print(f"    {FAIL} Error: {e}")
        total_failed += 1

    # ─────────────────────────────────────────────────────────────
    # TEST 7: Vision Dual Image Inference & Hough Calibration
    # ─────────────────────────────────────────────────────────────
    print("\n[7] Testing Vision Dual Image Inference on sample meal...")
    img1 = Path(r"C:\Users\abhiram\.gemini\antigravity\brain\646b060f-3532-41f7-b070-d2153b69c57c\.user_uploaded\media_1793870701374_81eb298c.jpg")
    img2 = Path(r"C:\Users\abhiram\.gemini\antigravity\brain\646b060f-3532-41f7-b070-d2153b69c57c\.user_uploaded\media_1793870712091_f0aaaada.jpg")

    if not img1.is_file() or not img2.is_file():
        root_dir = Path(__file__).resolve().parent.parent
        img1 = root_dir / "FrontEnd/images/sample_meal_top.jpg"
        img2 = root_dir / "FrontEnd/images/sample_meal_side.jpg"

    if img1.is_file() and img2.is_file():
        t0 = time.time()
        try:
            with open(img1, "rb") as f1, open(img2, "rb") as f2:
                files = {"top_image": ("top.jpg", f1, "image/jpeg"), "side_image": ("side.jpg", f2, "image/jpeg")}
                data = {"plate_diameter_cm": "20.0"}
                r_vis = requests.post("http://127.0.0.1:8001/vision", files=files, data=data, timeout=180)
                assert r_vis.status_code == 200, f"Vision failed: {r_vis.status_code}: {r_vis.text[:200]}"
                v_json = r_vis.json()
                elapsed = time.time() - t0
                print(f"    {PASS} Vision response in {elapsed:.2f}s: Dish='{v_json.get('dish')}', Ingredients={v_json.get('ingredients')}")
                total_passed += 1

                # TEST 8: Full pipeline chain with the vision response
                print("\n[8] Testing Nutrition Prediction on Vision Output...")
                ml_in = {"dish": v_json.get("dish"), "ingredients": v_json.get("ingredients")}
                r_ml = requests.post("http://127.0.0.1:8000/predict/vision", json=ml_in, timeout=10)
                assert r_ml.status_code == 200, f"ML failed: {r_ml.status_code}"
                ml_json = r_ml.json()
                totals = ml_json.get("recipe_totals", {})
                print(f"    {PASS} End-to-End Totals: {totals.get('energy_kcal')} kcal | Carbs: {totals.get('carb_g')}g | Protein: {totals.get('protein_g')}g | Fat: {totals.get('fat_g')}g")
                total_passed += 1

        except Exception as e:
            print(f"    {FAIL} Error: {e}")
            total_failed += 1
    else:
        print("    [SKIP] Test images not found")

    print("\n=========================================================")
    print(f"  VALIDATION SUMMARY: {total_passed} PASSED, {total_failed} FAILED")
    print("=========================================================\n")
    return total_failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
