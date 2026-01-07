# Global Occupancy Analytics Dashboard | User & Developer Manual

## 1. Project Overview
This documentation covers the installation, execution, and operational usage of the **Global Occupancy Analytics Dashboard**. This application has evolved from a static Python script into a full-stack **Streamlit Web Application** designed for facility managers and data scientists.

It provides real-time monitoring of room occupancy, environmental sensor analysis (CO2, Temperature), and a "Champion vs. Challenger" AI retraining pipeline that allows for continuous model improvement without downtime.

---

## 2. System Prerequisites

Before running the application, ensure your environment meets the following requirements:

* **Operating System:** Windows 10/11, macOS, or Linux.
* **Python Version:** Python 3.8 or higher is strictly required (3.10+ recommended for speed).
* **Browser:** Google Chrome, Firefox, or Edge (for viewing the Dashboard).

---

## 3. Directory Structure (Critical)

The application relies on **relative pathing** to locate the AI models and training datasets. Your folder structure **must** look exactly like this for the application to function correctly:

```text
/occupancy_app_folder
    │
    ├── app_occupancy.py               <-- THE MAIN APPLICATION (Run this)
    ├── occupancy_model_bundle.pkl     <-- The "Active" (Champion) AI Brain
    │
    ├── baseline_model/                <-- FOLDER: Contains the backup model
    │   └── occupancy_model_bundle.pkl <-- The "Baseline" (Safety) AI Brain
    │
    └── training_datasets/             <-- FOLDER: Validation Data
        ├── datatest.txt               <-- REQUIRED: The "Gold Standard" Test Set
        ├── datatraining.txt           <-- (Optional) For historical reference
        └── datatraining2.txt          <-- (Optional) For historical reference

```

**⚠️ Important Note:** If the `baseline_model` folder or `training_datasets/datatest.txt` are missing, the "Retraining" and "Comparison" features will fail to load.

---

## 4. Installation & Setup

Unlike the previous version, this web application requires specific libraries for the User Interface (Streamlit) and Visualization (Plotly).

### Step 1: Open Terminal

Open Command Prompt, PowerShell, or Terminal and navigate to your project folder:

```powershell
cd path\to\occupancy_app_folder

```

### Step 2: Install Dependencies

Run the following command to install all required packages at once. This ensures compatibility with the new dashboard features:

```powershell
pip install streamlit pandas numpy scikit-learn xgboost plotly joblib

```

* **`streamlit`**: Runs the web server and UI.
* **`plotly`**: Renders the interactive heatmaps and charts.
* **`xgboost`**: The core machine learning algorithm.
* **`joblib`**: Handles the saving/loading of the AI model files (`.pkl`).

---

## 5. Execution Instructions

To launch the dashboard, you do not use the standard `python` command. You must use the `streamlit` command.

**Command:**

```powershell
streamlit run app_occupancy.py

```

**What happens next?**

1. The terminal will display "You can now view your Streamlit app in your browser."
2. Your default web browser will automatically open to `http://localhost:8501`.
3. If it does not open automatically, copy that URL and paste it into your browser.

---

## 6. Navigating the Application

The application is divided into three distinct tabs, serving different operational needs.

### Tab 1: 📊 Dashboard (The Command Center)

This is the view for Facility Managers.

* **Brain Selection (Sidebar):** Use the radio buttons to switch between the **Active Model** (latest version) and the **Baseline Model** (original factory version). This allows you to A/B test logic instantly.
* **Data Modes:**
* **📂 Historical Analysis:** Upload raw sensor logs (`.txt` or `.csv`) to audit past performance. You can filter by Date Range and Working Hours (e.g., 9 AM - 5 PM) to exclude night-time noise.
* **📡 Live Monitor:** Runs a simulation loop to demonstrate real-time alerting capabilities.


* **Key Metrics:**
* **Bad Air %:** Monitors employee comfort. It calculates the percentage of time the room was *occupied* while CO2 levels exceeded 1000 ppm.
* **AI Confidence:** Shows how "sure" the model is. If confidence drops below 60%, the dashboard will flag it.


* **Export:** Click the "📥 Download Data as CSV" button at the bottom to generate a report for external stakeholders.

### Tab 2: 🧠 Model Retraining (The AI Lab)

This is the view for Data Scientists and Developers.

* **Concept:** Implements a **"Champion vs. Challenger"** workflow.
* **Step 1:** Upload a new dataset (must contain an `Occupancy` column).
* **Step 2:** Adjust Hyperparameters (Boosting Rounds, Learning Rate) using the sliders.
* **Step 3:** Click **"🥊 Train Challenger"**.
* The system trains a new XGBoost model on your uploaded data.
* It then evaluates **BOTH** the new model (Challenger) and the current active model (Champion) against the **Gold Standard Test Set** (`datatest.txt`).


* **Step 4:** View the "Scorecard." This table compares Accuracy, F1 Score, Precision, and Recall.
* **Step 5:** **Decide.**
* **✅ Promote:** Overwrites the `occupancy_model_bundle.pkl` file with the new model. The Dashboard instantly updates to use this new brain.
* **🗑️ Discard:** Throws away the new model and keeps the old one.



### Tab 3: 📘 Help & Guide

Contains the glossary of terms, user instructions, and definitions of specific alerts (e.g., "Ventilate Room").

---

## 7. Troubleshooting

### "Command 'streamlit' not found"

This means the library didn't install correctly or isn't in your PATH.

* *Fix:* Try running `python -m streamlit run app_occupancy.py`.

### "Gold Standard Test Set not found"

The retraining tab shows an error red box.

* *Fix:* Ensure you have a folder named `training_datasets` next to your script, and it contains the file `datatest.txt`.

### "Sidebar is missing"

You likely clicked the arrow to collapse it.

* *Fix:* Look for a small `>` arrow in the top-left corner of the webpage to expand the sidebar menu.

### "Model performance is 0.0"

This happens if the test set in the retraining tab has no variance (e.g., all rows are 0).

* *Fix:* Ensure your `datatest.txt` contains both Occupied (1) and Empty (0) examples.

---

## 8. Outputs & Persistence

* **`occupancy_model_bundle.pkl`**: This binary file stores the entire AI pipeline (Scaler + XGBoost Model + Feature List). It is automatically updated when you click "Promote Challenger."
* **`occupancy_report.csv`**: Generated on-demand when the user clicks "Download" in the dashboard.

```
