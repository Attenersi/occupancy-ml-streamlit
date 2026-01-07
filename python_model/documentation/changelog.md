# Project Evolution: Alteryx -> Python -> Full Stack AI Command Center

## Executive Summary
**Result:** F1 Score increased from **~0.68** (Baseline Alteryx) to **~0.91** (XGBoost Python).
**Architecture:** The project has evolved from a static script into a production-grade **Streamlit Application**. It now features a "Champion/Challenger" retraining pipeline, real-time simulation capabilities, and a business-centric dashboard interface inspired by enterprise IoT platforms.

---

## Detailed Change List

### Phase 1: Logic & Algorithm Improvements
* **Physics-Based Feature Engineering:**
    * Implemented **Deltas (Rate of Change):** Calculated the difference between $t$ and $t-1$ for CO2 and Temperature. This allows the model to detect "Events" (people entering/leaving) rather than just static "Levels."
    * **Rolling Averages:** Applied 4-step rolling windows to smooth out sensor noise and capture trends.
    * **Lag Features:** Introduced `CO2_Lag1` to give the model historical context (memory).
* **Cyclical Time Encoding:**
    * Replaced hardcoded hour buckets (e.g., "Morning", "Afternoon") with **Sine/Cosine transformation**. This mathematically preserves the relationship that 23:00 is close to 00:00.
* **Robust Validation:**
    * Switched from random K-Fold to `TimeSeriesSplit`. This ensures we never train on future data to predict the past, preventing data leakage.

### Phase 2: Operational & Code Structure
* **Dynamic Pathing & Robustness:**
    * The script now automatically resolves file paths relative to `__file__`, eliminating "File Not Found" errors across different environments.
    * Added automated dependency checks to ensure libraries like `xgboost` are present.
* **Data Leakage Prevention:**
    * Strictly enforced the removal of the `Light` variable (which is a proxy for occupancy, not a predictor) to ensure the model solves the actual problem using environmental sensors (CO2/Temp/Humidity).

### Phase 3: Application, UI/UX & Model Lifecycle (Major Update)

#### A. The "Smart Building" Dashboard (Tab 1)
* **Enterprise UI Overhaul:**
    * **Dark Mode Command Center:** Implemented a custom `#0e1117` dark theme with high-contrast Electric Cyan (`#00e6ff`) metrics for professional visibility.
    * **Custom CSS Injection:** Overrode default Streamlit styling to create "Card-style" KPI containers with shadow effects, matching the aesthetic of the *Infsoft* dashboard.
    * **White-Labeling:** Hidden Streamlit branding (footer, hamburger menu) to simulate a proprietary SaaS product.
* **Advanced Visualizations:**
    * **Occupancy Heatmap:** Replaced standard line charts with a **Day x Hour Matrix**. Implemented a 'Blues' gradient scale with text annotations inside cells to visualize usage density at a glance.
    * **Dual-Axis Probability Trends:** Created a Plotly chart combining continuous **Model Confidence (Probability)** (Area chart) with binary **Ground Truth** (Line chart) to visualize exactly where the model is uncertain.
    * **Business-Centric KPIs:**
        * **Bad Air %:** Calculates the percentage of occupied time where CO2 > 1000 ppm (Comfort Score).
        * **AI Confidence:** Displays real-time certainty (e.g., "98% Sure").
        * **Actionable Alerts:** Dynamic text that prompts users (e.g., "⚠️ Ventilate Room" or "Turn off AC").
* **Modes:**
    * **Live Simulation:** A mock loop generates synthetic sensor data to demonstrate real-time monitoring capabilities.
    * **Historical Analysis:** Full support for CSV/TXT upload with date range and working-hour filters.
    * **Data Export:** Added functionality to download the filtered sensor log as a CSV report.

#### B. The "AI Lab" & Retraining Pipeline (Tab 2)
* **Champion vs. Challenger Architecture:**
    * Implemented a safe retraining workflow. Users can upload new datasets to train a "Challenger" model.
    * **Gold Standard Validation:** Both the Active Model (Champion) and the New Model (Challenger) are evaluated against a fixed, external test set (`datatest.txt`) to ensure fair, scientific comparison.
    * **Scorecard:** A side-by-side comparison table of F1 Score, Accuracy, Precision, and Recall with delta indicators (e.g., `+0.02`).
* **Safety Mechanisms:**
    * **Baseline Fallback:** The app maintains a read-only "Baseline Model" in a separate folder. Users can toggle the dashboard to view this model's performance at any time, ensuring a safety net if the Active model degrades.
    * **Session State Management:** Uses `st.session_state` to persist training results across UI interactions without losing data.
* **Performance Optimization:**
    * Switched XGBoost `tree_method` to `'hist'` (Histogram-based), speeding up training on large datasets by ~10x.

#### C. Documentation & Usability (Tab 3)
* **Dedicated Help Tab:** Added a full user guide explaining the difference between "Historical" and "Live" modes.
* **Glossary:** Defined technical terms (F1 Score, Confidence) in business language for facility managers.
* **Universal Compatibility:** Implemented a `safe_rerun()` utility to handle deprecated Streamlit commands, ensuring the app works on both old and new Python environments.