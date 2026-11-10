# 🥗 NutraVision — Dual-View AI Nutrition Analyzer & Food Weight Estimator

NutraVision is an offline-capable, dual-microservice AI platform that estimates food portion weights and complete nutritional profiles from paired top-down and side-view meal photographs.

The project combines **PyTorch DeepLabV3+ semantic segmentation**, local **Qwen3-VL (2B)** multimodal reasoning, **calibrated plate geometry**, and **CatBoost multi-target regression** enriched with authentic Indian food datasets from **CVIT IIIT Hyderabad (ICVGIP 2025)** and **ICMR-NIN**.

---

## 🏗️ Architecture & Processing Flow

```
                      [ User Photos: Top-Down + Side-View ]
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   Vision Service (Port 8001)      │
                     ├───────────────────────────────────┤
                     │ 1. DeepLabV3+ MobileNetV2 (104cls)│
                     │    Native 512px overlapping tiles │
                     │ 2. Hough Circle Plate Calibration │
                     │    Real-world cm-to-pixel ratio   │
                     │ 3. Qwen3-VL:2B Multimodal VLM     │
                     │    Visual component & fraction    │
                     │ 4. Food Category Density & Volume │
                     │    IIIT-H empirical gram bounds   │
                     └─────────────────┬─────────────────┘
                                       │
                {"dish": "Mutton Curry", "ingredients": {"Rice": 190, "Mutton Curry": 140}}
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │  Nutrition Service (Port 8000)    │
                     ├───────────────────────────────────┤
                     │ 1. 3-Tier Preparation Resolution  │
                     │    (e.g., Biryani vs Salad prep)  │
                     │ 2. Char-wb TF-IDF (2-5) + 64D SVD │
                     │ 3. 5× CatBoost Regressors:        │
                     │    - energy_kcal                  │
                     │    - carb_g                       │
                     │    - protein_g                    │
                     │    - fat_g                        │
                     │    - fibre_g                      │
                     │ 4. Weight Scaling & Aggregation   │
                     └─────────────────┬─────────────────┘
                                       │
                                       ▼
                     [ FrontEnd UI / Client Application ]
```

---

## 📂 Project Directory Structure

```text
NutraVision/
├── FrontEnd/                # Web UI (HTML, CSS, JS, Assets)
│   ├── index.html           # Dual-upload interface & nutrition dashboard
│   ├── script.js            # Client-side API orchestrator (Ports 8001 & 8000)
│   ├── style.css            # Responsive layout & styling
│   └── images/              # Logo and sample test meal photos
│
├── vision/                  # Vision Microservice (Port 8001)
│   ├── app.py               # FastAPI entry point & CORS configuration
│   ├── segmentation.py      # DeepLabV3+ tiled inference & Hough plate scale
│   ├── estimation.py        # Volume-to-gram density & IIIT-H weight bounds
│   ├── vlm.py               # Ollama Qwen3-VL client & universal JSON parser
│   ├── food_density.json    # Food category density mapping
│   └── models/
│       └── FoodSeg103/      # DeepLabV3+ MobileNetV2 weights (17.9 MB) & labels
│
├── ML/                      # Nutrition Regression Microservice (Port 8000)
│   ├── app.py               # FastAPI router, integration endpoints & static UI
│   ├── predictor.py         # Multi-target CatBoost prediction engine
│   ├── preprocessor.py      # 3-tier preparation resolution & TF-IDF/SVD features
│   ├── schemas.py           # Pydantic request/response schemas
│   ├── *.cbm                # 5 trained CatBoost regression models
│   └── *.joblib             # TF-IDF, SVD, and keyword feature artifacts
│
├── train/                   # Dataset generation & model training scripts
│   ├── generate_indian_dataset.py  # IIIT-H & ICMR-NIN dataset builder
│   ├── train_catboost.py           # CatBoost 5-target training pipeline
│   └── data/                       # Enriched training CSVs
│
├── tests/                   # Automated end-to-end test suite
│   ├── test_suite.py        # 8-step full system validation
│   ├── test_indian_foods.py # Indian dish macro prediction benchmarks
│   └── test_nutrition_api.py# API regression tests
│
├── scripts/                 # Utility scripts
│   ├── pipeline.py          # Command-line dual-view meal inference runner
│   └── evaluate_weights.py  # Portion weight accuracy evaluator
│
├── requirements.txt         # Single unified project dependencies
├── start_services.ps1       # Single-command background launcher for both services
└── README.md                # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10 or 3.11** (64-bit)
- **Ollama** installed on your system with `qwen3-vl:2b`:
  ```powershell
  ollama pull qwen3-vl:2b
  ```

### 2. Environment Setup
Install the unified dependencies from the project root:
```powershell
py -3.11 -m venv .venv_run
.\.venv_run\Scripts\python.exe -m pip install --upgrade pip
.\.venv_run\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Launching Services
Run the automated launcher from PowerShell:
```powershell
.\start_services.ps1
```
This script validates Ollama, initializes both microservices, and opens the frontend UI:
* **Frontend UI & Nutrition Service:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Vision Service:** [http://127.0.0.1:8001](http://127.0.0.1:8001)
* **API Documentation (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs)

---

## 🍛 IIIT Hyderabad Dataset Integration

NutraVision integrates empirical data from **CVIT IIIT Hyderabad (ICVGIP 2025 Indian Thali Dataset)** and **ICMR-NIN (National Institute of Nutrition)**:

1. **Physical Portion Bounds:** Over 3,278 physically weighed Indian servings constrain computer-vision volume estimation:
   * Flatbreads (*Phulka / Roti*): bounded to $30\text{--}80\text{ g}$ (preventing 200g loaf hallucinations).
   * *Papad*: bounded to $\le 25\text{ g}$.
   * *Chutneys & Pickles*: bounded to $\le 45\text{ g}$.
2. **Context-Aware Preparation (3-Tier Resolution):**
   * *Explicit:* `"grilled chicken"` $\to$ grilled.
   * *Contextual:* Rice inside a `"Biryani"` $\to$ boiled with ghee/spices; veggies in a `"Salad"` $\to$ raw.
   * *Smart Default:* Inherent food physical state.

---

## 🧪 Running the Test Suite

Validate all endpoints, models, CORS preflight, and dual-image inference:
```powershell
.\.venv_run\Scripts\python.exe tests/test_suite.py
```

Run Indian food macro accuracy benchmarks:
```powershell
.\.venv_run\Scripts\python.exe tests/test_indian_foods.py
```

---

## 🛡️ License
Academic and research use under MIT License. Models and datasets subject to respective upstream licenses (FoodSeg103, IIIT-H CVIT, ICMR-NIN).
