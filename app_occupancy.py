import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
import traceback
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

# ==========================================
# 0. CONFIG & DARK THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Analytics | Global Occupancy",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    
    /* KPI Cards */
    div[data-testid="metric-container"] {
        background-color: #262730; border: 1px solid #464b5c; padding: 15px;
        border-radius: 5px; box-shadow: 0px 1px 3px rgba(0,0,0,0.5);
        text-align: center; margin-bottom: 10px;
    }
    div[data-testid="metric-container"] label { color: #b0b0b0; font-size: 0.8rem; text-transform: uppercase; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #00e6ff; font-size: 2rem; font-weight: 700; }
    
    /* Top Bar */
    .top-bar {
        background-color: #d81e05; color: white; padding: 10px 20px;
        font-weight: bold; font-size: 1.2rem; border-radius: 5px 5px 0 0;
        margin-bottom: 20px; display: flex; align-items: center;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 10px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #555; border-radius: 5px; }

    /* UI Fixes */
    div[data-testid="stSelectbox"] > label, div[data-testid="stSelectbox"] div[role="combobox"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"], div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] div[role="radiogroup"] { cursor: pointer !important; }
    
    /* White Labeling */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

SCRIPT_DIR = Path(__file__).resolve().parent
ACTIVE_MODEL_PATH = SCRIPT_DIR / "occupancy_model_bundle.pkl"
BASELINE_MODEL_PATH = SCRIPT_DIR / "baseline_model" / "occupancy_model_bundle.pkl"

# --- LOCATE GOLD STANDARD TEST SET ---
# We look in likely locations relative to the script
possible_paths = [
    SCRIPT_DIR / "training_datasets" / "datatest.txt",        # Inside script folder
    SCRIPT_DIR.parent / "training_datasets" / "datatest.txt", # One level up
    SCRIPT_DIR.parent.parent / "training_datasets" / "datatest.txt" # Two levels up (Likely for repo structure)
]
TEST_DATA_PATH = None
for p in possible_paths:
    if p.exists():
        TEST_DATA_PATH = p
        break

COLUMN_NAMES = ['Id', 'Date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio', 'Occupancy']

# --- UTILS ---
def safe_rerun():
    try: st.rerun() 
    except AttributeError:
        try: st.experimental_rerun() 
        except: st.warning("Manual refresh required.")

@st.cache_data
def load_and_preprocess_file(file):
    try:
        # Handle UploadedFile or Path object
        if isinstance(file, (str, Path)):
            name = str(file)
            is_csv = name.endswith('.csv')
        else:
            name = file.name
            is_csv = name.endswith('.csv')

        if is_csv:
            df = pd.read_csv(file)
            if 'Date' not in df.columns: 
                if not isinstance(file, (str, Path)): file.seek(0)
                df = pd.read_csv(file, names=COLUMN_NAMES, header=None, skiprows=1)
        else:
            df = pd.read_csv(file, names=COLUMN_NAMES, header=None, skiprows=1)
            
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Hour'] = df['Date'].dt.hour
            df['Day'] = df['Date'].dt.date
            df['Weekday'] = df['Date'].dt.day_name()
            df['Minute'] = df['Date'].dt.minute
        return df
    except Exception as e:
        return None

def smart_feature_engineering(df):
    df = df.copy()
    if 'Hour' not in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Hour'] = df['Date'].dt.hour

    df['Is_Weekend'] = df['Date'].dt.weekday.isin([5, 6]).astype(int)
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour']/24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour']/24)
    
    window_size = 4
    if df.empty: return df

    df['CO2_Smooth'] = df['CO2'].rolling(window=window_size, min_periods=1).mean()
    df['Temp_Smooth'] = df['Temperature'].rolling(window=window_size, min_periods=1).mean()
    df['CO2_Delta'] = (df['CO2'] - df['CO2'].shift(window_size)).fillna(0)
    df['Temp_Delta'] = (df['Temperature'] - df['Temperature'].shift(window_size)).fillna(0)
    df['CO2_Lag1'] = df['CO2'].shift(window_size).fillna(df['CO2'])
    
    return df

def fetch_live_data_placeholder():
    import random
    
    # 1. Get the last recorded values (or defaults if starting fresh)
    if 'live_history' in st.session_state and not st.session_state.live_history.empty:
        last_row = st.session_state.live_history.iloc[-1]
        last_co2 = last_row['CO2']
        last_temp = last_row['Temperature']
        last_humid = last_row.get('Humidity', 30)
    else:
        # Starting baselines (Empty room conditions)
        last_co2 = 450
        last_temp = 21.0
        last_humid = 30.0

    # 2. "Random Walk" Logic (Drift slightly instead of jumping)
    # CO2 drifts by -10 to +25 (Bias towards rising slightly to simulate people entering)
    new_co2 = last_co2 + random.randint(-10, 25)
    
    # Temperature drifts very slowly (-0.1 to +0.15)
    new_temp = last_temp + random.uniform(-0.05, 0.1)
    
    # Humidity drifts randomly
    new_humid = last_humid + random.uniform(-0.5, 0.5)

    # 3. Clamp values to keep them realistic
    # Force CO2 to stay between 400 (fresh air) and 2000 (very stuffy)
    new_co2 = max(400, min(new_co2, 2000))
    # Force Temp between 18C and 30C
    new_temp = max(18.0, min(new_temp, 30.0))
    # Force Humidity between 20% and 80%
    new_humid = max(20.0, min(new_humid, 80.0))

    # 4. Calculate Physics (HumidityRatio)
    # Simple approximation: Humidity Ratio relates to Temp and RH
    # (Formula: 0.622 * (RH/100 * Psat) / (Patm - ...)) 
    # We use a simplified linear correlation for the simulation to keep the model happy
    mock_humidity_ratio = (new_temp * 0.0002) + (new_humid * 0.0001)

    return pd.DataFrame([{
        'Date': pd.Timestamp.now(),
        'Temperature': new_temp, 
        'Humidity': new_humid, 
        'Light': 500, # Assume lights are ON
        'CO2': new_co2, 
        'HumidityRatio': mock_humidity_ratio
    }])

def load_model_from_path(path):
    if path.exists():
        try:
            bundle = joblib.load(path)
            models = bundle['models']
            feature_cols = bundle['features']
            model = models.get("XGBoost", models[list(models.keys())[0]])
            return model, feature_cols
        except Exception:
            return None, []
    return None, []

active_model, active_features = load_model_from_path(ACTIVE_MODEL_PATH)

# ==========================================
# 2. DASHBOARD RENDERER
# ==========================================
def render_dashboard_tab():
    st.sidebar.header("📊 Dashboard Settings")
    model_choice = st.sidebar.radio("Select Brain Version", ["🏆 Active Model", "🛡️ Baseline Model"])
    
    if model_choice == "🏆 Active Model":
        dashboard_model, dashboard_features = active_model, active_features
        if not dashboard_model: st.sidebar.error("Active model not found.")
    else:
        dashboard_model, dashboard_features = load_model_from_path(BASELINE_MODEL_PATH)
        if not dashboard_model: st.sidebar.error("Baseline model not found.")

    st.sidebar.divider()
    data_mode = st.sidebar.radio("Data Mode", ["📂 Historical Analysis", "📡 Live Monitor"], key="dash_mode")

    if data_mode == "📂 Historical Analysis":
        uploaded_file = st.sidebar.file_uploader("Upload Data", type=['txt', 'csv'], key="dash_file")
        if uploaded_file:
            df_loaded = load_and_preprocess_file(uploaded_file)
            if df_loaded is not None:
                render_dashboard_charts(st.container(), df_loaded, dashboard_model, dashboard_features, is_live=False)
            else: st.error("Invalid File")
        else: st.info("Please upload your dataset.")
        return

    elif data_mode == "📡 Live Monitor":
        # 1. Initialize Session State for History if it doesn't exist
        if 'live_history' not in st.session_state:
            st.session_state.live_history = pd.DataFrame(columns=['Date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])

        # 2. Controls
        col_controls = st.sidebar.columns(2)
        start_btn = col_controls[0].button("▶️ Start", key="start_sim")
        clear_btn = col_controls[1].button("🗑️ Reset", key="reset_sim")

        if clear_btn:
            st.session_state.live_history = pd.DataFrame(columns=['Date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])
            st.rerun()

        st.write("### 📡 Real-Time Sensor Stream")
        
        # 3. Create a single placeholder that we will update (Prevents full screen flash)
        dashboard_placeholder = st.empty()

        if start_btn:
            # Run for 50 iterations (or use 'while True' if you want it infinite)
            for _ in range(50):
                # A. Fetch single new row
                new_row = fetch_live_data_placeholder()
                
                # B. Add to history (Accumulate data)
                st.session_state.live_history = pd.concat([st.session_state.live_history, new_row], ignore_index=True)
                
                # C. Limit history to last 100 points (to keep app fast)
                if len(st.session_state.live_history) > 100:
                    st.session_state.live_history = st.session_state.live_history.iloc[-100:]

                # D. Render the FULL history into the placeholder
                with dashboard_placeholder.container():
                    render_dashboard_charts(
                        st.container(), 
                        st.session_state.live_history,  # <--- Pass Full History, not just new_row
                        dashboard_model, 
                        dashboard_features, 
                        is_live=True
                    )
                
                # E. Wait slightly
                time.sleep(1)
        else:
            # Show existing data if we are paused
            if not st.session_state.live_history.empty:
                with dashboard_placeholder.container():
                    render_dashboard_charts(st.container(), st.session_state.live_history, dashboard_model, dashboard_features, is_live=True)
            else:
                st.info("Click 'Start' to connect to live sensors.")

def render_dashboard_charts(container, df_raw, model, features, is_live=False):
    title_suffix = "(Baseline)" if "Baseline" in str(model) else ""
    st.markdown(f'<div class="top-bar">📊 ANALYTICS | GLOBAL OCCUPANCY DASHBOARD {title_suffix}</div>', unsafe_allow_html=True)

    if not model or not features: return

    # --- 1. FILTERING ---
    if not is_live:
        min_date = df_raw['Date'].min().date()
        max_date = df_raw['Date'].max().date()
        with st.expander("📅 Filter Settings", expanded=False):
            c_f1, c_f2 = st.columns(2)
            start_date, end_date = c_f1.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
            start_hour, end_hour = c_f2.slider("Working Hours", 0, 23, (7, 19))
        
        mask = (df_raw['Date'].dt.date >= start_date) & (df_raw['Date'].dt.date <= end_date) & \
               (df_raw['Hour'] >= start_hour) & (df_raw['Hour'] <= end_hour)
        df_view = df_raw[mask].copy()
    else:
        df_view = df_raw.copy()

    # --- 2. PREDICTION ---
    try:
        df_processed = smart_feature_engineering(df_view)
        if df_processed.empty: st.warning("No data found."); return
        
        X_input = df_processed[features]
        probs = model.predict_proba(X_input)[:, 1]
        df_processed['Probability'] = probs
        df_processed['Prediction'] = (probs >= 0.5).astype(int)
    except Exception: st.error("Processing Error"); return

    # --- 3. TOP ROW (KPIs + GAUGE/HEATMAP) ---
    col_kpi, col_main = st.columns([1, 4])
    
    with col_kpi:
        st.metric("Average Occupancy", f"{df_processed['Prediction'].mean() * 100:.1f}%")
        st.metric("Peak CO2 Level", f"{df_processed['CO2'].max():.0f}")
        latest = df_processed.iloc[-1]
        
        if is_live:
            status = "Occupied" if latest['Prediction'] == 1 else "Vacant"
            st.metric("Current Status", status)
            conf = latest['Probability'] if latest['Prediction'] == 1 else (1 - latest['Probability'])
        else:
            if 'Weekday' in df_processed.columns:
                st.metric("Busiest Day", df_processed.groupby('Weekday')['Prediction'].mean().idxmax())
            occ_data = df_processed[df_processed['Prediction'] == 1]
            bad_air = (occ_data['CO2'] > 1000).sum() / len(occ_data) * 100 if not occ_data.empty else 0
            st.metric("Bad Air %", f"{bad_air:.1f}%")

    with col_main:
        st.subheader("Occupancy Visualization")
        if not is_live:
            # --- HISTORICAL HEATMAP ---
            heatmap_data = df_processed.groupby(['Hour', 'Weekday'])['Prediction'].mean().reset_index()
            heatmap_data['Occupancy_Pct'] = (heatmap_data['Prediction'] * 100).round(1)
            pivot_grid = heatmap_data.pivot(index='Hour', columns='Weekday', values='Occupancy_Pct')
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot_grid = pivot_grid.reindex(columns=days_order)
            
            fig_heatmap = px.imshow(pivot_grid, text_auto=True, color_continuous_scale="Blues", aspect="auto", template="plotly_dark", labels=dict(color="Probability"))
            fig_heatmap.update_layout(coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            # --- LIVE GAUGE (UPDATED TO SHOW PROBABILITY) ---
            
            # 1. Get raw probability (0.0 to 1.0)
            raw_prob = latest['Probability']

            # 2. Draw Gauge showing "Chance of Occupancy" instead of "Confidence"
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = raw_prob * 100,    # <--- Now shows 0% if empty, 100% if full
                title = {'text': "Occupancy Probability"},
                gauge = {
                    'axis': {'range': [None, 100]}, 
                    'bar': {'color': "#00e6ff"}, # Blue bar to match trend line
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(0, 255, 0, 0.1)"},   # Green zone (likely empty)
                        {'range': [50, 100], 'color': "rgba(255, 0, 0, 0.1)"}  # Red zone (likely full)
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            
            import time
            st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_live_{time.time()}")

    st.markdown("---")
    
    # --- 4. BOTTOM ROW (TRENDS + DATA) ---
    b_col1, b_col2 = st.columns(2)
    
    with b_col1:
        st.subheader("Occupancy Probability Trends")
        
        # We create a specific figure for trends (fig_trend) to avoid conflicts
        fig_trend = go.Figure()
        
        # This now runs in BOTH Live and History modes
        fig_trend.add_trace(go.Scatter(
            x=df_processed['Date'], 
            y=df_processed['Probability'], 
            mode='lines', 
            name='Model Confidence (%)', 
            line=dict(color='#00e6ff', width=2), 
            fill='tozeroy', 
            fillcolor='rgba(0, 230, 255, 0.1)'
        ))
        
        if 'Occupancy' in df_processed.columns:
            fig_trend.add_trace(go.Scatter(
                x=df_processed['Date'], 
                y=df_processed['Occupancy'], 
                mode='lines', 
                name='Real Occupancy (0/1)', 
                line=dict(color="#39d043", width=1)
            ))
            
        fig_trend.update_layout(
            template="plotly_dark", 
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            margin=dict(t=10, l=0, r=0, b=0), 
            yaxis_tickformat=".0%", 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1)
        )
        
        import time
        st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_chart_{time.time()}")
            
    with b_col2:
        st.subheader("Detailed Sensor Log")
        table_df = df_processed.sort_values('Date', ascending=False)
        
        # 1. Select the columns you want to see (Add engineered features here)
        # We check if they exist first to avoid errors if features haven't generated yet
        columns_to_show = ['Date', 'Temperature', 'Humidity', 'CO2', 'CO2_Delta', 'Probability', 'Prediction']
        final_cols = [c for c in columns_to_show if c in table_df.columns]
        
        # 2. Render with fancy formatting
        st.dataframe(
            table_df[final_cols].style
            .background_gradient(subset=['CO2'], cmap='Reds')  # Red for high CO2
            .background_gradient(subset=['Probability'], cmap='Blues')  # Blue for high confidence
            .format({
                'Probability': '{:.1%}',  # Show as 95.0%
                'Temperature': '{:.1f}°C', 
                'Humidity': '{:.1f}%',
                'CO2': '{:.0f} ppm',
                'CO2_Delta': '{:+.1f}'    # Show signs like +5.0 or -2.0
            }), 
            use_container_width=True, 
            height=300
        )
        
        if not is_live:
            st.download_button(
                label="📥 Download Data as CSV",
                data=table_df.to_csv(index=False).encode('utf-8'),
                file_name='occupancy_report.csv',
                mime='text/csv',
                key="final_download_btn"
            )

# ==========================================
# 3. RETRAINING (UPDATED WITH GOLD STANDARD)
# ==========================================
def calculate_metrics(model, X, y, features):
    try:
        # Filter to required features
        X_in = X[features]
        probs = model.predict_proba(X_in)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return {
            "Accuracy": accuracy_score(y, preds),
            "F1 Score": f1_score(y, preds),
            "Precision": precision_score(y, preds, zero_division=0),
            "Recall": recall_score(y, preds, zero_division=0)
        }
    except Exception:
        return None

def render_retrain_tab():
    st.markdown('<div class="top-bar" style="background-color: #ff9900;">🧠 AI LAB | CHAMPION vs CHALLENGER</div>', unsafe_allow_html=True)
    
    # 1. Check if Gold Standard exists
    if TEST_DATA_PATH:
        st.success(f"✅ Gold Standard Validation Set Found: `{TEST_DATA_PATH.name}`")
    else:
        st.error("❌ Gold Standard Test Set (`datatest.txt`) not found in `training_datasets` folder.")
        st.warning("Please verify the file exists to ensure accurate model scoring.")
        return

    st.write("### 🥊 Train & Compare")
    train_file = st.file_uploader("Upload New Training Data", type=['txt', 'csv'], key="train_file")
    
    if train_file:
        df_train = load_and_preprocess_file(train_file)
        if df_train is not None and 'Occupancy' in df_train.columns:
            st.info(f"Training Data Loaded: {len(df_train):,} rows")
            
            c1, c2 = st.columns(2)
            n_estimators = c1.slider("Boosting Rounds", 50, 500, 100)
            lr = c2.slider("Learning Rate", 0.01, 0.3, 0.1)
            
            if 'challenger_model' not in st.session_state:
                st.session_state.challenger_model = None
                st.session_state.challenger_metrics = {}
                st.session_state.champion_metrics = {}

            if st.button("🥊 Train Challenger Model"):
                with st.spinner("Training & Running Fairness Audit..."):
                    try:
                        # A. PREPARE TRAINING DATA (Upload)
                        df_eng_train = smart_feature_engineering(df_train).dropna()
                        features_to_use = ['Temperature', 'Humidity', 'CO2', 'HumidityRatio', 'CO2_Smooth', 
                                           'Temp_Smooth', 'CO2_Delta', 'Temp_Delta', 'CO2_Lag1', 
                                           'Is_Weekend', 'Hour_Sin', 'Hour_Cos']
                        available_feats = [c for c in features_to_use if c in df_eng_train.columns]
                        
                        # Train on 100% of uploaded data (since we have external test set)
                        X_train = df_eng_train[available_feats]
                        y_train = df_eng_train['Occupancy']
                        
                        challenger = XGBClassifier(n_estimators=n_estimators, learning_rate=lr, use_label_encoder=False, 
                                                   eval_metric='logloss', tree_method='hist')
                        challenger.fit(X_train, y_train)
                        
                        # B. PREPARE GOLD STANDARD TEST DATA
                        df_gold = load_and_preprocess_file(TEST_DATA_PATH)
                        df_eng_gold = smart_feature_engineering(df_gold).dropna()
                        X_gold = df_eng_gold
                        y_gold = df_eng_gold['Occupancy']
                        
                        # C. CALCULATE METRICS (4 vs 4)
                        # Challenger
                        chal_metrics = calculate_metrics(challenger, X_gold, y_gold, available_feats)
                        
                        # Champion
                        champ_metrics = None
                        if active_model:
                             champ_metrics = calculate_metrics(active_model, X_gold, y_gold, active_features)
                        
                        st.session_state.challenger_model = challenger
                        st.session_state.challenger_feats = available_feats
                        st.session_state.challenger_metrics = chal_metrics
                        st.session_state.champion_metrics = champ_metrics
                        
                    except Exception as e:
                        st.error(f"Training Error: {e}"); st.code(traceback.format_exc())

            if st.session_state.challenger_model:
                st.divider()
                st.subheader("🏆 The Scorecard (Evaluated on `datatest.txt`)")
                
                # Create Comparison DataFrame
                metrics_df = pd.DataFrame({
                    "Metric": ["Accuracy", "F1 Score", "Precision", "Recall"],
                    "🏆 Champion": [
                        st.session_state.champion_metrics.get("Accuracy", 0),
                        st.session_state.champion_metrics.get("F1 Score", 0),
                        st.session_state.champion_metrics.get("Precision", 0),
                        st.session_state.champion_metrics.get("Recall", 0)
                    ] if st.session_state.champion_metrics else [0,0,0,0],
                    "🥊 Challenger": [
                        st.session_state.challenger_metrics["Accuracy"],
                        st.session_state.challenger_metrics["F1 Score"],
                        st.session_state.challenger_metrics["Precision"],
                        st.session_state.challenger_metrics["Recall"]
                    ]
                })
                
                # Calculate Deltas
                metrics_df["Delta"] = metrics_df["🥊 Challenger"] - metrics_df["🏆 Champion"]
                
                # Display using nice columns
                c_score, c_decide = st.columns([3, 1])
                
                with c_score:
                    # Formatted Table
                    st.dataframe(
                        metrics_df.style.format({
                            "🏆 Champion": "{:.4f}", 
                            "🥊 Challenger": "{:.4f}", 
                            "Delta": "{:+.4f}"
                        }).background_gradient(subset=["Delta"], cmap="RdYlGn", vmin=-0.1, vmax=0.1),
                        use_container_width=True,
                        height=180
                    )
                
                with c_decide:
                    st.write("### Decision")
                    chal_f1 = st.session_state.challenger_metrics["F1 Score"]
                    champ_f1 = st.session_state.champion_metrics.get("F1 Score", 0) if st.session_state.champion_metrics else 0
                    
                    if chal_f1 > champ_f1:
                        st.success(f"New Model is Better! (+{chal_f1 - champ_f1:.4f})")
                        if st.button("✅ Promote"):
                            bundle = {'models': {'XGBoost': st.session_state.challenger_model}, 'features': st.session_state.challenger_feats}
                            joblib.dump(bundle, ACTIVE_MODEL_PATH)
                            st.success("Promoted!"); time.sleep(1); safe_rerun()
                    else:
                        st.warning("New Model is Worse.")
                        if st.button("🗑️ Discard"):
                            st.session_state.challenger_model = None; safe_rerun()

# ==========================================
# 4. HELP TAB
# ==========================================
def render_help_tab():
    st.markdown('<div class="top-bar" style="background-color: #2e86de;">📘 HELP & USER GUIDE</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. End User Instructions
    Welcome to the Global Occupancy Analytics Dashboard. This tool allows facility managers to monitor room usage and optimize energy consumption.
    
    #### **Getting Started**
    1.  **Select Mode (Sidebar):**
        * **📂 Historical Analysis:** Upload past sensor data (`.csv` or `.txt`) to analyze trends, busy hours, and air quality issues.
        * **📡 Live Monitor:** Simulates a real-time connection to room sensors for instant status checks.
    2.  **Filter Data:**
        * Use the **Date Range** picker to focus on specific weeks or months.
        * Adjust **Working Hours** (e.g., 7 AM - 7 PM) to exclude night-time data.
    3.  **Choose Your Brain:**
        * **🏆 Active Model:** The smartest, most recently trained AI.
        * **🛡️ Baseline Model:** The original "factory setting" AI (useful for comparison).

    ---
    ### 2. Glossary of Metrics
    Understanding the numbers on your dashboard:
    
    | Metric | Definition |
    | :--- | :--- |
    | **AI Confidence** | How sure the model is about its prediction (0-100%). **>90%**: Very Sure **<60%**: Low Confidence (Check manually) |
    | **Action Alerts** | **⚠️ Ventilate!**: CO2 > 1000 ppm. Air is stuffy/unhealthy**Turn off AC**: Room is EMPTY but Temp > 24°C. |
    | **Bad Air %** | The percentage of time the room was **Occupied** but the air quality was poor (CO2 > 1000). A high number indicates a ventilation problem during meetings. |
    | **Busiest Day** | The day of the week with the highest average occupancy rate. Useful for scheduling cleaning crews. |

    ---
    ### 3. Model Management (AI Lab)
    The **"🧠 Model Retraining"** tab allows you to improve the AI over time.
    
    * **Champion vs Challenger:** When you train a new model, it becomes the "Challenger". The app compares it against your current "Champion".
    * **Promotion:** Only click "✅ Promote" if the Challenger's **F1 Score** is higher than the Champion's.
    * **Safety:** The **Baseline Model** is never overwritten, so you always have a backup.
    """)

# ==========================================
# 5. MAIN
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧠 Model Retraining", "📘 Help & Guide"])
with tab1: render_dashboard_tab()
with tab2: render_retrain_tab()
with tab3: render_help_tab()