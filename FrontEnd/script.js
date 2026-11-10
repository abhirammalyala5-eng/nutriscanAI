// FrontEnd/script.js - NutraVision Dual-View Frontend Controller

const topImageInput = document.getElementById("topImage");
const sideImageInput = document.getElementById("sideImage");
const topPreview = document.getElementById("topPreview");
const sidePreview = document.getElementById("sidePreview");
const topPlaceholder = document.getElementById("topPlaceholder");
const sidePlaceholder = document.getElementById("sidePlaceholder");
const plateDiameterInput = document.getElementById("plateDiameter");
const analyzeButton = document.getElementById("analyzeButton");
const resultDiv = document.getElementById("result");

// Top-Down Image Preview Handler
if (topImageInput) {
    topImageInput.addEventListener("change", function () {
        const file = topImageInput.files[0];
        if (file) {
            const url = URL.createObjectURL(file);
            topPreview.src = url;
            topPreview.style.display = "block";
            if (topPlaceholder) topPlaceholder.style.display = "none";
        }
    });
}

// Side Image Preview Handler
if (sideImageInput) {
    sideImageInput.addEventListener("change", function () {
        const file = sideImageInput.files[0];
        if (file) {
            const url = URL.createObjectURL(file);
            sidePreview.src = url;
            sidePreview.style.display = "block";
            if (sidePlaceholder) sidePlaceholder.style.display = "none";
        }
    });
}

// Helper to escape HTML characters
function escapeHtml(text) {
    const map = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    };
    return String(text).replace(/[&<>"']/g, (m) => map[m]);
}

// Format confidence badge
function getConfidenceBadge(confidence) {
    const conf = String(confidence || "medium").toLowerCase();
    if (conf.includes("high") || conf === "high") {
        return `<span class="badge badge-high">High</span>`;
    } else if (conf.includes("low") || conf === "low") {
        return `<span class="badge badge-low">Low</span>`;
    }
    return `<span class="badge badge-med">Medium</span>`;
}

// Main Analyze Function
async function analyzeFood() {
    if (!topImageInput || topImageInput.files.length === 0) {
        resultDiv.innerHTML = `
            <div class="initial-prompt" style="color: #b33939;">
                <p>⚠️ <strong>Missing Top-Down Photo:</strong> Please select at least an overhead (top-down) photo of the meal.</p>
            </div>
        `;
        return;
    }

    const topFile = topImageInput.files[0];
    const sideFile = (sideImageInput && sideImageInput.files.length > 0)
        ? sideImageInput.files[0]
        : topFile; // Gracefully fallback to mirroring top image if side is omitted

    const plateDiameter = plateDiameterInput ? parseFloat(plateDiameterInput.value) || 25.0 : 25.0;

    // Build form data
    const formData = new FormData();
    formData.append("top_image", topFile);
    if (sideFile) {
        formData.append("side_image", sideFile);
    }
    formData.append("plate_diameter_cm", plateDiameter.toString());

    // Update UI to loading with live timer
    analyzeButton.disabled = true;
    analyzeButton.innerText = "⏳ Running Segmentation & Regression Models...";

    // Live elapsed-time counter shown while vision model runs (can take 30–120 s)
    let _elapsed = 0;
    const _steps = [
        "📡 Uploading images to Vision server...",
        "🔍 Running DeepLabV3+ food segmentation...",
        "📐 Calibrating plate scale & geometry...",
        "🧠 Running Qwen VLM ingredient estimation...",
        "⚖️  Finalising weight estimates...",
    ];
    function _timerHtml(sec) {
        const stepMsg = _steps[Math.min(Math.floor(sec / 4), _steps.length - 1)];
        const bar = Math.min(Math.round((sec / 20) * 20), 20);
        const filled = "█".repeat(bar) + "░".repeat(20 - bar);
        return `
            <div class="initial-prompt" id="_progress_box">
                <p><strong>${stepMsg}</strong></p>
                <p style="font-family:monospace;letter-spacing:1px;color:#4a90d9">${filled}</p>
                <p style="color:#888;font-size:0.9em">⏱️ Elapsed: <strong>${sec}s</strong> — Processing meal via vision pipeline (typically 10–20s)…</p>
            </div>`;
    }
    resultDiv.innerHTML = _timerHtml(0);
    const _timerInterval = setInterval(() => {
        _elapsed++;
        const box = document.getElementById("_progress_box");
        if (box) box.outerHTML = _timerHtml(_elapsed);
        // also update the outer resultDiv if box was replaced
        const box2 = document.getElementById("_progress_box");
        if (!box2) resultDiv.innerHTML = _timerHtml(_elapsed);
    }, 1000);

    try {
        // ── STEP 1: Send images to Vision Service (Port 8001) ─────────────
        const visionUrl = window.VISION_API_URL || "http://127.0.0.1:8001/vision";

        const visionResponse = await fetch(visionUrl, {
            method: "POST",
            body: formData,
        });

        const visionData = await visionResponse.json();

        if (!visionResponse.ok || visionData.error || visionData.detail) {
            const errMsg = visionData.error || visionData.detail || `Vision server returned status ${visionResponse.status}`;
            resultDiv.innerHTML = `
                <div class="initial-prompt" style="color: #b33939;">
                    <p>❌ <strong>Vision Analysis Failed:</strong> ${escapeHtml(errMsg)}</p>
                </div>
            `;
            return;
        }

        // ── STEP 2: Extract ONLY dish + ingredients → send to local ML endpoint
        // Strip everything else (estimates, scale_reference, segmentation, etc.)
        const mlPayload = {
            dish: visionData.dish || null,
            ingredients: visionData.ingredients || {}
        };

        // Show intermediate status (vision done, ML starting)
        clearInterval(_timerInterval);
        resultDiv.innerHTML = `
            <div class="initial-prompt">
                <p>✅ <strong>Vision done in ${_elapsed}s — ${escapeHtml(visionData.dish || "Unknown dish")} detected.</strong><br>
                🤖 Running CatBoost regression on <strong>${Object.keys(mlPayload.ingredients).length} ingredient(s)</strong>…</p>
            </div>
        `;

        const nutritionUrl = window.NUTRITION_API_URL || "http://127.0.0.1:8000/predict/vision";
        const mlResponse = await fetch(nutritionUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(mlPayload),
        });

        const mlData = await mlResponse.json();

        if (!mlResponse.ok || mlData.error || mlData.detail) {
            const errMsg = mlData.error || mlData.detail || `ML server returned status ${mlResponse.status}`;
            resultDiv.innerHTML = `
                <div class="initial-prompt" style="color: #b33939;">
                    <p>❌ <strong>Nutrition Prediction Failed:</strong> ${escapeHtml(errMsg)}</p>
                </div>
            `;
            return;
        }

        // ── STEP 3: Merge ML results into vision data for rendering ────────
        const data = {
            ...visionData,
            nutrition: mlData.nutrition,
            recipe_totals: mlData.recipe_totals,
        };

        // Render Results
        renderAnalysisResults(data);

    } catch (error) {
        console.error("API error:", error);
        clearInterval(_timerInterval);
        resultDiv.innerHTML = `
            <div class="initial-prompt" style="color: #b33939;">
                <p>❌ <strong>Connection Error:</strong> ${escapeHtml(error.message || "Unknown error")}.<br>
                Make sure both the Vision Service (127.0.0.1:8001) and Nutrition Service (127.0.0.1:8000) are running.</p>
            </div>
        `;
    } finally {
        clearInterval(_timerInterval);
        analyzeButton.disabled = false;
        analyzeButton.innerText = "🔬 Analyze Food & Run Regression Models";
    }
}

// Render Results DOM
function renderAnalysisResults(data) {
    const dishName = escapeHtml(data.dish || "Detected Plated Meal");
    const scale = data.scale_reference || {};
    const scaleDetected = scale.detected ? "✅ Calibrated" : "⚠️ Estimated (Uncalibrated)";
    const footprintText = scale.food_footprint_cm2 ? ` | Footprint: ${scale.food_footprint_cm2} cm²` : "";
    const plateText = `🍽️ ${scaleDetected} (${scale.plate_diameter_cm || 25} cm Plate${footprintText})`;

    const recipeTotals = data.recipe_totals || {
        total_weight_g: 0,
        energy_kcal: 0,
        carb_g: 0,
        protein_g: 0,
        fat_g: 0,
        fibre_g: 0,
    };

    // 1. Build Totals Banner
    const totalsHtml = `
        <div class="totals-summary-banner">
            <div class="total-stat-box">
                <div class="stat-label">Total Weight</div>
                <div class="stat-value">${recipeTotals.total_weight_g} g</div>
            </div>
            <div class="total-stat-box">
                <div class="stat-label">Calories</div>
                <div class="stat-value">${recipeTotals.energy_kcal} kcal</div>
            </div>
            <div class="total-stat-box">
                <div class="stat-label">Carbs</div>
                <div class="stat-value">${recipeTotals.carb_g} g</div>
            </div>
            <div class="total-stat-box">
                <div class="stat-label">Protein</div>
                <div class="stat-value">${recipeTotals.protein_g} g</div>
            </div>
            <div class="total-stat-box">
                <div class="stat-label">Fats</div>
                <div class="stat-value">${recipeTotals.fat_g} g</div>
            </div>
            <div class="total-stat-box">
                <div class="stat-label">Fibre</div>
                <div class="stat-value">${recipeTotals.fibre_g} g</div>
            </div>
        </div>
    `;

    // 2. Build Vision Estimates Table
    let estimatesRows = "";
    const estimates = data.estimates || [];
    if (estimates.length > 0) {
        estimates.forEach((item) => {
            const name = escapeHtml(item.name || "Item");
            const weight = item.weight_g ? item.weight_g.estimate : (data.ingredients ? data.ingredients[item.name] : "--");
            const vol = item.volume_ml ? `${item.volume_ml.min} - ${item.volume_ml.max}` : "--";
            const density = item.density_g_per_ml ? `${item.density_g_per_ml.min} - ${item.density_g_per_ml.max}` : "--";
            const badge = getConfidenceBadge(item.confidence);

            estimatesRows += `
                <tr>
                    <td><strong>${name}</strong></td>
                    <td><strong>${weight} g</strong></td>
                    <td>${vol} ml</td>
                    <td>${density} g/cm³</td>
                    <td>${badge}</td>
                </tr>
            `;
        });
    } else if (data.ingredients) {
        for (const [name, weight] of Object.entries(data.ingredients)) {
            estimatesRows += `
                <tr>
                    <td><strong>${escapeHtml(name)}</strong></td>
                    <td><strong>${weight} g</strong></td>
                    <td>--</td>
                    <td>1.0 g/cm³</td>
                    <td><span class="badge badge-med">Medium</span></td>
                </tr>
            `;
        }
    }

    const estimatesTableHtml = `
        <h4 class="result-section-title">⚖️ Vision Ingredient Weight Estimates</h4>
        <div class="table-responsive">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Ingredient</th>
                        <th>Estimated Weight</th>
                        <th>Estimated Volume</th>
                        <th>Density Range</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    ${estimatesRows || '<tr><td colspan="5">No ingredients estimated.</td></tr>'}
                </tbody>
            </table>
        </div>
    `;

    // 3. Build Nutritional Regression Model Table
    let regressionRows = "";
    const nutritionList = data.nutrition || [];
    if (nutritionList.length > 0) {
        nutritionList.forEach((item) => {
            regressionRows += `
                <tr>
                    <td><strong>${escapeHtml(item.ingredient)}</strong></td>
                    <td>${item.weight_g} g</td>
                    <td><strong>${item.energy_kcal} kcal</strong></td>
                    <td>${item.carb_g} g</td>
                    <td>${item.protein_g} g</td>
                    <td>${item.fat_g} g</td>
                    <td>${item.fibre_g} g</td>
                </tr>
            `;
        });
    }

    const regressionTableHtml = `
        <h4 class="result-section-title">📊 NutraVision Regression Model Predictions</h4>
        <div class="table-responsive">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Ingredient</th>
                        <th>Weight</th>
                        <th>Calories</th>
                        <th>Carbs</th>
                        <th>Protein</th>
                        <th>Fat</th>
                        <th>Fibre</th>
                    </tr>
                </thead>
                <tbody>
                    ${regressionRows || '<tr><td colspan="7">No nutritional regression predictions returned.</td></tr>'}
                </tbody>
            </table>
        </div>
    `;

    // Combine All
    resultDiv.innerHTML = `
        <div class="result-card-header">
            <h3>🍛 ${dishName}</h3>
            <span class="scale-badge">${plateText}</span>
        </div>
        ${totalsHtml}
        ${regressionTableHtml}
        ${estimatesTableHtml}
    `;

    // 4. Update the bottom nutrition grid cards
    updateNutritionCards(recipeTotals, data.micronutrients);
}

// Update Macro & Micronutrient Cards
function updateNutritionCards(totals, micronutrients) {
    if (!totals) return;

    const micro = micronutrients || {};

    const setVal = (id, val, unit) => {
        const el = document.getElementById(id);
        if (el) el.textContent = `${val !== undefined && val !== null ? val : "--"} ${unit}`;
    };

    setVal("calories", totals.energy_kcal, "kcal");
    setVal("carbohydrates", totals.carb_g, "g");
    setVal("protein", totals.protein_g, "g");
    setVal("fats", totals.fat_g, "g");
    setVal("fibre", totals.fibre_g, "g");

    setVal("free-sugar", micro.free_sugar !== undefined ? micro.free_sugar : 0.0, "g");
    setVal("sodium", micro.sodium !== undefined ? micro.sodium : 0.0, "mg");
    setVal("calcium", micro.calcium !== undefined ? micro.calcium : 0.0, "mg");
    setVal("iron", micro.iron !== undefined ? micro.iron : 0.0, "mg");
    setVal("vitamin-c", micro.vitamin_c !== undefined ? micro.vitamin_c : 0.0, "mg");
    setVal("folate", micro.folate !== undefined ? micro.folate : 0.0, "µg");
}
