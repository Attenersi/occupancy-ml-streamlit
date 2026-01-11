# Occupancy Detection System | Technical Documentation & Architecture Specification

## 1. Project Overview & Business Value

### 1.1 Executive Summary
This project implements an end-to-end Machine Learning ecosystem designed to detect and predict room occupancy based solely on environmental sensor data (Temperature, Humidity, CO2). Originally conceived as a static Alteryx workflow, the system has been re-architected into a modern, production-ready Python stack. It solves a critical facility management problem: **optimizing energy consumption and space utilization** without the privacy concerns or installation costs of cameras.

### 1.2 Core Capabilities
The system has evolved into a dual-component architecture:
1.  **Training Pipeline (The Backend):** A robust script that ingests raw historical logs, engineers physics-based features, and trains multiple classifier algorithms to identify the optimal "Champion" model.
2.  **Interactive Dashboard (The Frontend):** A Streamlit-based web application serving as the "Command Center." It allows facility managers to monitor real-time status, audit historical data, and retrain the AI models using a Human-in-the-Loop workflow.

**Performance:** The system achieves a **Test F1 Score of ~0.91** on the `datatest.txt` hold-out set using an XGBoost classifier, demonstrating superior capability in handling the non-linear dynamics of air quality changes compared to linear baselines (F1 ~0.87).

---

## 2. System Architecture

The project adheres to the **"Train Once, Deploy Anywhere"** design pattern, ensuring that the logic used to build the model is identical to the logic used during live inference.

### A. The Training Engine (`python_model_occupancy.py`)
This script is the "Factory" that produces the intelligence. It handles the entire lifecycle of model creation:
* **Data Ingestion:** Automatically scans relative directories to merge multiple training files (`datatraining.txt`, `datatraining2.txt`) into a cohesive training set.
* **Leakage Prevention:** Automatically drops the `Light` variable. While `Light` is highly correlated with occupancy (lights on = people present), it is considered a "proxy variable" that fails during daylight hours or automated lighting schedules. The model is forced to learn from harder, more robust signals (CO2 levels).
* **Artifact Generation:**
    * **Metric Logs:** Exports `model_f1_scores.csv` for auditability.
    * **The Model Bundle (`occupancy_model_bundle.pkl`):** This is the critical output. It is a serialized dictionary containing not just the trained model, but also the feature list and column transformers. This ensures the Dashboard doesn't need to "guess" how to process data—it simply loads this brain.

### B. The User Application (`app_occupancy.py`)
This is the "Consumer" of the intelligence. Built on **Streamlit**, it provides a low-latency interface for end-users.
* **Dynamic Loading:** It uses `joblib` to load the `.pkl` bundle. If the Training Engine updates the model, the Dashboard updates instantly upon refresh.
* **Champion vs. Challenger Architecture:** The app includes a sophisticated "AI Lab" tab. It allows users to upload new data, train a candidate model (Challenger), and perform a side-by-side "Face-Off" against the live model (Champion) on a neutral Gold Standard test set.
* **Operational Features:**
    * **Simulation Loop:** A mock data generator that produces synthetic CO2 and Temperature streams to demonstrate system behavior during demos.
    * **Business Intelligence:** Calculates derived metrics like "Bad Air %" (Occupied time with CO2 > 1000ppm) to translate sensor readings into actionable facilities insights.

---

## 3. Data Pipeline & Physics-Based Feature Engineering

Raw sensor data is noisy and static. To achieve high accuracy, we transformed the raw readings into **dynamic features** that capture the *physics* of human presence.

### 3.1 Cyclical Time Encoding
Standard machine learning models misinterpret time. If we treat Hour 23 and Hour 0 as integers, the model thinks they are far apart.
* **Solution:** We map time onto a **Unit Circle**.
* **Implementation:** We calculate `Hour_Sin` and `Hour_Cos`.
* **Result:** This preserves temporal continuity, allowing the model to understand that 11 PM (23:00) is physically adjacent to Midnight (00:00).

### 3.2 Rolling Statistics (Noise Filtering)
Sensors often have micro-fluctuations (jitter).
* **Solution:** Applied a **Rolling Mean** with a window size of 4.
* **Features:** `CO2_Smooth`, `Temp_Smooth`.
* **Result:** This smooths out the signal, allowing the model to focus on the underlying trend rather than outlier spikes.

### 3.3 Delta Features (Rate of Change)
This is the most critical innovation. Occupancy is an "Event" (entry/exit). Events cause *change*.
* **Solution:** Calculated the first-order difference between $t$ and $t-1$.
* **Features:** `CO2_Delta` and `Temp_Delta`.
* **Why it works:** A room might have high CO2 because it was occupied an hour ago (static level). But a *rapidly rising* CO2 level (`CO2_Delta` > +10) definitively proves someone entered the room *right now*.

### 3.4 Lag Features (Memory)
* **Solution:** Created `CO2_Lag1` features.
* **Result:** Provides the model with "short-term memory," enabling it to understand context (e.g., "The CO2 was low 15 minutes ago, now it's high").

---

## 4. Model Strategy & Algorithms

We evaluated four distinct algorithmic approaches to ensure the final solution was robust.

1.  **XGBoost (The Champion):**
    * **Architecture:** Gradient Boosted Decision Trees.
    * **Why it won:** It naturally handles non-linear discontinuities (e.g., HVAC systems turning on/off) and complex interactions (e.g., "If Time is Night AND CO2 is falling, Room is Empty"). It achieved the highest F1 Score (~0.91).
2.  **Random Forest:**
    * **Role:** Stability Benchmark.
    * **Performance:** Very strong, but slightly slower inference time than XGBoost. Used as a fallback if XGBoost overfits.
3.  **Decision Tree:**
    * **Role:** Interpretability.
    * **Performance:** Lower accuracy, but useful for explaining rules to stakeholders (e.g., "See, the tree splits on CO2 > 450").
4.  **Logistic Regression:**
    * **Role:** Linear Baseline.
    * **Performance:** F1 ~0.87. It failed to capture the complex, non-linear relationship between temperature decay and occupancy, proving that a complex model was necessary.

---

## 5. Validation Strategy

To prevent "Data Leakage" (the cardinal sin of time-series modeling), we rejected standard K-Fold Cross Validation.

* **Temporal Integrity:** We utilized **`TimeSeriesSplit`** (3 splits). This ensures that the training set always consists of time periods *prior* to the validation set. We never use future data to predict the past.
* **Gold Standard Hold-Out:** The file `datatest.txt` was completely sequestered from the training process. It serves as the final, unbiased "Exam" for the models.
* **Metric Selection:** We optimized for **F1 Score** rather than Accuracy. Since rooms are empty for long periods (imbalanced classes), a model that simply guessed "Empty" every time would have high Accuracy but zero utility. F1 Score balances Precision (trust) and Recall (coverage).

## 6. Installation (One-Time Setup)

### Prerequisites
* **Python 3.8+** must be installed.
* **Internet Connection** (to download libraries).

### Step 1: Download the Application
Download the repository and extract it. Open your terminal (Command Prompt or PowerShell) and navigate to the project folder:

```powershell
cd occupancy_app/python_model/occupancy-ml-streamlit

```

### Step 2: Install Dependencies

Run this command to install all necessary tools (Streamlit, XGBoost, Plotly, etc.):

```powershell
pip install -r requirements.txt

```

---

## 7. How to Run the Dashboard

To start the application, run the following command in your terminal.
*(Note: We use `python -m` to ensure the system finds the application correctly).*

```powershell
python -m streamlit run app_occupancy.py

```

**What happens next?**

1. You will see a message: "You can now view your Streamlit app in your browser."
2. A local web server will start.
3. Your web browser will automatically open to `http://localhost:8501`.

---

## 8. Using the Dashboard

The application is divided into three tabs:

### 📊 Tab 1: Dashboard (Live Monitor)

* **Main Chart:** Shows the live CO2 levels (Yellow Line) and the AI's Probability of Occupancy (Blue Area).
* **Metric Cards:**
* **Bad Air %:** Percentage of time people were present while CO2 > 1000 ppm.
* **AI Confidence:** How sure the model is about its current prediction.


* **Sidebar Controls:**
* **Select Model:** Switch between the "Active" model and the "Baseline" (Backup) model.
* **Data Source:** Choose "Live Simulation" to see real-time movements.



### 🧠 Tab 2: Model Retraining (AI Lab)

* **Step 1:** Upload a new CSV dataset.
* **Step 2:** Click **"Train Challenger"**. The system builds a new model.
* **Step 3:** Review the **"Scorecard"**. If the new model (Challenger) has a higher F1 Score than the current one, click **"Promote Challenger"**.

### 📘 Tab 3: Help

* Contains a glossary of terms and troubleshooting tips.

---

## 9. Troubleshooting

**Error: "background_gradient requires matplotlib"**

* **Fix:** Run `pip install -r requirements.txt` again to install the missing visualization tools.

**The Dashboard is blank or slow**

* **Fix:** Refresh your web browser page (F5).

```

```