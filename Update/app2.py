import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
# --- SOPHISTICATION UPGRADE: Use Gradient Boosting for Quantile/High-Accuracy Regression ---
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Daskan Intelligence | AI Project Estimator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK (Daskan Theme: Black/White + Green Accent) ---
# Daskan Green Accent: #049449
st.markdown("""
<style>
    /* Metric Card Border (Daskan Green Accent) */
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #049449;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    /* --- IMPROVED TAB VISIBILITY (Inactive Tabs) --- */
    .stTabs [data-baseweb="tab"] {
        height: 55px; /* Slightly taller for better clickability */
        white-space: pre-wrap;
        background-color: #EAEAEA; /* Increased contrast for inactive tabs */
        border-radius: 6px 6px 0px 0px; /* Slightly more rounded */
        gap: 1px;
        padding-top: 15px; /* Increased padding */
        padding-bottom: 15px; /* Increased padding */
        font-weight: 600; /* Semi-bold text */
        color: #333333; /* Dark text on inactive tab */
    }
    /* --- IMPROVED TAB VISIBILITY (Active Tab) --- */
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF; /* Active background */
        border-bottom: 4px solid #049449; /* Thicker green line indicator */
        color: #049449; /* Active text color is Daskan Green */
        font-weight: 800; /* Bolder text for active tab */
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DATA GENERATOR (Cached) ---
@st.cache_data
def generate_synthetic_data(seed, size):
    np.random.seed(seed)
    project_types = ['Residential', 'Commercial', 'Institutional', 'Industrial']
    materials = ['Wood', 'Steel', 'Concrete', 'Mixed']
    data = []
    
    for i in range(size):
        p_type = np.random.choice(project_types, p=[0.5, 0.25, 0.15, 0.1])
        mat = np.random.choice(materials)
        
        if p_type == 'Residential': area = np.random.normal(300, 100); levels = np.random.randint(1, 4)
        elif p_type == 'Commercial': area = np.random.normal(1200, 400); levels = np.random.randint(2, 8)
        else: area = np.random.normal(3500, 1000); levels = np.random.randint(1, 12)
            
        area = max(50, area)
        is_winter = 1 if np.random.randint(1, 13) in [12, 1, 2, 3] else 0
        complexity = (levels * 0.5) + (1.5 if mat == 'Mixed' else 1.0)
        base_effort = (area * 0.08) * complexity
        if is_winter: base_effort *= 1.25
        if p_type == 'Institutional': base_effort *= 1.4
        actual_effort = abs(np.random.normal(base_effort, base_effort * 0.10))
        
        data.append({
            'project_type': p_type,
            'material_type': mat,
            'surface_area_m2': round(area, 2),
            'num_levels': levels,
            'is_winter': is_winter,
            'floor_area_ratio': round(area / levels, 2),
            'total_project_effort': round(actual_effort, 2)
        })
        
    df = pd.DataFrame(data)
    # Simulate some missing values for the Imputation demonstration
    df.loc[df.sample(frac=0.05).index, 'surface_area_m2'] = np.nan
    df.loc[df.sample(frac=0.02).index, 'material_type'] = np.nan
    return df

# --- SIDEBAR: DATA INPUT SELECTION ---
with st.sidebar:
    st.image("C:\\Users\\wiza\\viktor-apps\\Synthetic Data Generator\\cropped-DaskanLogo.png", width=300)
    st.markdown("###  Daskan Intelligence")
    st.caption("Thesis Version 3.0 - Quantile Regression")
    
    st.divider()
    
    st.subheader("Data Source Selector")
    uploaded_file = st.file_uploader(
        "Upload Daskan's Master CSV (Required columns: surface_area_m2, num_levels, project_type, material_type, is_winter, total_project_effort)", 
        type=['csv']
    )
    
    st.divider()
    
    st.subheader(" Synthetic Parameters")
    random_seed = st.slider("Random Seed", 1, 100, 50)
    sample_size = st.slider("Dataset Size", 100, 1000, 500)
    

# --- CONDITIONAL DATA LOADING ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        data_source = "Daskan Inc. Uploaded Data"
        st.success(" Successfully loaded uploaded file!")
        
        # Feature Engineering on uploaded data
        if 'floor_area_ratio' not in df.columns and 'surface_area_m2' in df.columns and 'num_levels' in df.columns:
            df['floor_area_ratio'] = df['surface_area_m2'] / df['num_levels']
            st.info("Feature Engineered: 'floor_area_ratio' calculated.")
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = generate_synthetic_data(random_seed, sample_size)
        data_source = "Synthetic Data (Error Fallback)"
else:
    # Use synthetic data if no file is uploaded
    df = generate_synthetic_data(random_seed, sample_size)
    data_source = f"Synthetic Data (Seed: {random_seed})"

# --- MAIN APP ---
st.title("Structural Project Analytics Dashboard")
st.markdown(f"**Current Data Source:** `{data_source}`")

tabs = st.tabs([" Deep Dive Analytics", " Data Quality Check", " AI Model Engine", " Decision Support System"])

# ----------------------------------------------------------------------
# --- TAB 1: INTERACTIVE ANALYTICS (Plotly) ---
# ----------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Data Understanding Phase (RQ1)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Complexity Analysis: Effort vs. Area")
        if all(col in df.columns for col in ['surface_area_m2', 'num_levels', 'total_project_effort', 'project_type', 'floor_area_ratio']):
            # --- FIX: Changed from 3D scatter to 2D scatter plot ---
            fig_scatter = px.scatter(
                df, x='surface_area_m2', y='total_project_effort',
                color='project_type', 
                size='num_levels', # Using levels for size now to encode complexity
                hover_data=['floor_area_ratio'],
                title="Effort vs. Area, Categorized by Project Type (Size indicates Levels)",
                labels={'total_project_effort': 'Effort (Hours)', 'surface_area_m2': 'Surface Area (m²)'}
            )
            fig_scatter.update_layout(margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.error("Cannot generate Complexity plot. Missing required columns in dataset.")
        
    with col2:
        st.subheader("Effort Distribution")
        if 'total_project_effort' in df.columns and 'project_type' in df.columns:
            fig_hist = px.histogram(df, x="total_project_effort", nbins=30, color="project_type",
                                   title="Effort Histogram by Type")
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.error("Cannot generate histogram. Missing 'total_project_effort' or 'project_type'.")

        st.subheader("Correlation Matrix")
        corr = df.select_dtypes(include=np.number).corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)


# ----------------------------------------------------------------------
# --- TAB 2: DATA QUALITY CHECK (Cleaned: No Data Leakage) ---
# ----------------------------------------------------------------------
with tabs[1]:
    st.markdown("### Data Preparation Phase (RQ2)")
    st.subheader("Data Validation & Imputation Strategy")
    st.dataframe(df.describe().T)
    
    st.markdown("#### Missing Values Report")
    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ['Feature', 'Missing Count']
    missing_data['Missing Percentage'] = (missing_data['Missing Count'] / len(df)) * 100
    
    fig_missing = px.bar(
        missing_data.sort_values(by='Missing Percentage', ascending=False),
        x='Missing Percentage', y='Feature', orientation='h',
        title='Missing Data by Feature'
    )
    st.plotly_chart(fig_missing, use_container_width=True)
    
    st.markdown("""
    **Imputation Policy (CRISP-DM Guideline):**
    Missing values will be handled by the **preprocessing pipeline** in the next tab to prevent data leakage:
    * **Numerical Features:** Imputed with the **Median**.
    * **Categorical Features:** Imputed with the **Mode** (most frequent category).
    """)
    
    if missing_data['Missing Count'].sum() > 0:
         st.warning(" Missing values detected. They will be handled correctly by the Pipeline in the 'AI Model Engine' tab.")

# ----------------------------------------------------------------------
# --- TAB 3: AI ENGINE (Advanced - Quantile Regression) ---
# ----------------------------------------------------------------------
with tabs[2]:
    st.markdown("### Modeling & Evaluation Phase (RQ3: Ensemble & Quantile)")
    
    required_cols = ['surface_area_m2', 'num_levels', 'project_type', 'material_type', 'is_winter', 'floor_area_ratio', 'total_project_effort']
    
    if not all(col in df.columns for col in required_cols):
        st.error(f" Cannot train model. The dataset is missing critical columns. Please ensure your uploaded file contains: {', '.join(required_cols)}")
    else:
        # Preprocessing Setup
        X = df.drop(['total_project_effort'], axis=1, errors='ignore')
        y = df['total_project_effort']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        col_config, col_metrics = st.columns([1, 3])
        
        with col_config:
            st.markdown("#### Model Configuration")
            
            # --- SOPHISTICATION UPGRADE 1: Model Selection ---
            model_type = st.radio("Ensemble Algorithm", ["Gradient Boosting Regressor", "Random Forest Regressor", "Linear Regression"])
            
            # --- SOPHISTICATION UPGRADE 2: Quantile Selection ---
            prediction_mode = st.radio(
                "Prediction Objective", 
                ["Point Estimate (Median)", "Prediction Interval (90% Quantile)"],
                help="Prediction Interval provides a confidence range for risk management."
            )
            
            # SCALING SELECTION
            scaling_type = st.radio(
                "Numerical Scaling Method", 
                ["Standard Scaler", "MinMax Scaler"], 
                help="Choose scaling technique for numerical features."
            )
            
            params = {}
            if model_type in ["Gradient Boosting Regressor", "Random Forest Regressor"]:
                n_est = st.slider("Trees (n_estimators)", 50, 300, 100)
                depth = st.slider("Max Depth", 5, 50, 20)
                params = {'n_estimators': n_est, 'max_depth': depth, 'random_state': 42}
        
        # --- BUILD PREPROCESSOR PIPELINE ---
        numeric_features = ['surface_area_m2', 'num_levels', 'floor_area_ratio', 'is_winter']
        categorical_features = ['project_type', 'material_type']

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler() if scaling_type == "Standard Scaler" else MinMaxScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_transformer, [f for f in numeric_features if f in X.columns]),
            ('cat', categorical_transformer, [f for f in categorical_features if f in X.columns])
        ])
        
        # --- MODEL TRAINING LOGIC ---
        with col_metrics:
            if st.button("Train Model(s) on Current Data"):
                st.session_state['model_trained'] = False
                
                # --- Quantile Regression Logic (Trains 3 Models) ---
                if prediction_mode == "Prediction Interval (90% Quantile)" and model_type == "Gradient Boosting Regressor":
                    st.session_state['mode'] = 'quantile'
                    
                    # Train three separate GBR models for the quantiles
                    models = {}
                    quantiles = {'Q05': 0.05, 'Q50': 0.50, 'Q95': 0.95}
                    
                    for name, alpha in quantiles.items():
                        gbr = GradientBoostingRegressor(loss='quantile', alpha=alpha, **params)
                        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', gbr)])
                        pipeline.fit(X_train, y_train)
                        models[name] = pipeline
                        
                    # Evaluate using the Q50 (Median) model
                    pipeline_median = models['Q50']
                    preds_median = pipeline_median.predict(X_test)
                    
                    # Calculate R2 and MAE for the median estimate
                    mae = mean_absolute_error(y_test, preds_median)
                    r2 = r2_score(y_test, preds_median)
                    
                    # Calculate Prediction Interval Coverage Probability (PICP) for thesis rigor
                    preds_q05 = models['Q05'].predict(X_test)
                    preds_q95 = models['Q95'].predict(X_test)
                    
                    # PICP: Percentage of actual values falling between Q05 and Q95 predictions
                    coverage = np.mean((y_test >= preds_q05) & (y_test <= preds_q95))
                    
                    # Display metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Median Accuracy (R²)", f"{r2:.3f}", delta="Target: >0.70")
                    m2.metric("MAE (Q50 Error)", f"{mae:.1f} Hours", delta_color="inverse")
                    m3.metric("PICP (Coverage)", f"{coverage:.1%}", delta=f"Target: 90.0%")
                    
                    st.session_state['adv_model'] = models
                    st.session_state['model_trained'] = True
                    st.success(" 90% Quantile Prediction Intervals (Q05, Q50, Q95) Trained Successfully!")
                    
                # --- Point Estimate Logic (Trains 1 Model) ---
                else:
                    st.session_state['mode'] = 'point'
                    
                    if model_type == "Gradient Boosting Regressor":
                        model = GradientBoostingRegressor(**params)
                    elif model_type == "Random Forest Regressor":
                        model = RandomForestRegressor(**params)
                    else:
                        model = LinearRegression()
                        
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
                    pipeline.fit(X_train, y_train)
                    preds = pipeline.predict(X_test)
                    
                    mae = mean_absolute_error(y_test, preds)
                    r2 = r2_score(y_test, preds)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Model Accuracy (R²)", f"{r2:.3f}", delta="Target: >0.70")
                    m2.metric("MAE (Avg Error)", f"{mae:.1f} Hours", delta_color="inverse")
                    m3.metric("Test Samples", len(y_test))
                    
                    st.session_state['adv_model'] = pipeline
                    st.session_state['model_trained'] = True
                    st.success(f" {model_type} Point Estimate Model Trained Successfully!")

                # --- Residual Plot (Always show for evaluation) ---
                if st.session_state.get('model_trained', False):
                    st.subheader("Residual Analysis (Actual vs. Predicted)")
                    
                    plot_preds = preds_median if st.session_state['mode'] == 'quantile' else preds
                    
                    fig_res = go.Figure()
                    # Use Daskan Green for the prediction markers
                    fig_res.add_trace(go.Scatter(x=y_test, y=plot_preds, mode='markers', name='Predictions', marker=dict(color='#049449', opacity=0.6)))
                    fig_res.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', name='Perfect Prediction', line=dict(color='black', dash='dash')))
                    fig_res.update_layout(xaxis_title="Actual Hours", yaxis_title="Predicted Hours")
                    st.plotly_chart(fig_res, use_container_width=True)

            else:
                 st.info("Click 'Train Model(s) on Current Data' to run the pipeline and evaluate performance.")

        # --- SOPHISTICATION UPGRADE 3: Feature Importance & XAI (Conceptual SHAP) ---
        if st.session_state.get('model_trained', False) and st.session_state['mode'] == 'point' and model_type == "Random Forest Regressor":
            st.divider()
            st.subheader(" Model Explainability: Feature Importance")
            
            pipeline_fit = st.session_state['adv_model']
            
            preprocessor_fitted = pipeline_fit.named_steps['preprocessor']
            try:
                ohe_cols = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
            except AttributeError:
                ohe_cols = preprocessor_fitted.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
            
            all_feats = numeric_features + list(ohe_cols)
            importances = pipeline_fit.named_steps['regressor'].feature_importances_
            
            if len(all_feats) == len(importances):
                feat_df = pd.DataFrame({'Feature': all_feats, 'Importance': importances}).sort_values(by='Importance', ascending=True)
                # Use Daskan Green for the bars
                fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Global Feature Importance (What drives the prediction?)", color_discrete_sequence=['#049449'])
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning("Could not display Feature Importance.")
        
        # NOTE: For Gradient Boosting, SHAP is the better explainability tool.
        if st.session_state.get('model_trained', False) and model_type == "Gradient Boosting Regressor":
            st.divider()
            st.markdown(f"###  Model Explainability (XAI)")
            st.info("For the final thesis, replace this placeholder with a **SHAP Summary Plot** from the Q50/Median model to demonstrate feature contribution rigor.")


# ----------------------------------------------------------------------
# --- TAB 4: DECISION SUPPORT (Business Value - Prediction Interval) ---
# ----------------------------------------------------------------------
with tabs[3]:
    st.markdown("### Workforce Planning & Quotation (Business Objective)")
    
    if not st.session_state.get('model_trained', False):
        st.warning(" Please train the model in the 'AI Model Engine' tab first.")
    else:
        col_in, col_out = st.columns([1, 2])
        
        with col_in:
            st.markdown("#### New Project Parameters")
            i_area = st.number_input("Surface Area (m²)", 1, 10000, 1)
            i_levels = st.number_input("Levels", 1, 50, 1)
            i_type = st.selectbox("Type", df['project_type'].unique())
            i_mat = st.selectbox("Material", df['material_type'].unique())
            i_winter = st.checkbox("Winter Schedule (Dec-Mar)?")
            
            # Prepare Input
            input_df = pd.DataFrame({
                'project_type': [i_type],
                'material_type': [i_mat],
                'surface_area_m2': [i_area],
                'num_levels': [i_levels],
                'is_winter': [1 if i_winter else 0],
                'floor_area_ratio': [i_area/i_levels]
            }, index=[0])

        with col_out:
            if st.button("Calculate Resource Plan"):
                
                if st.session_state['mode'] == 'quantile':
                    # Use all three quantile models
                    models = st.session_state['adv_model']
                    pred_q05 = models['Q05'].predict(input_df)[0]
                    pred_q50 = models['Q50'].predict(input_df)[0]
                    pred_q95 = models['Q95'].predict(input_df)[0]
                    
                    pred_hours = pred_q50 # Median is the central estimate
                    risk_margin = pred_q95 - pred_q50
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="margin:0; color:#1A1A1A;">{pred_q50:.1f} Hours</h2> <p style="margin:0;">Median Effort Estimate (50th Percentile)</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("###  Risk-Managed Quotation (RQ4)")
                    c1, c2 = st.columns(2)
                    c1.metric("90% Prediction Interval", f"{pred_q05:.0f} - {pred_q95:.0f} Hours")
                    # Delta color set to 'off' to maintain neutral/black text
                    c2.metric("Worst-Case Risk Margin", f"{risk_margin:.1f} Hours", delta="Buffer needed above median", delta_color='off')
                    
                else: # Point Estimate Mode
                    pipeline = st.session_state['adv_model']
                    pred_hours = pipeline.predict(input_df)[0]
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="margin:0; color:#1A1A1A;">{pred_hours:.1f} Hours</h2> <p style="margin:0;">Point Effort Estimate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # --- Shared Business Logic ---
                avg_hourly_rate = 120 # CAD
                hours_per_week = 40
                
                estimated_cost = pred_hours * avg_hourly_rate
                weeks_needed_1_eng = pred_hours / hours_per_week
                
                st.subheader(" Workforce Planner")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Estimated Cost (CAD)", f"${estimated_cost:,.0f}")
                c2.metric("Time (1 Engineer)", f"{weeks_needed_1_eng:.1f} Weeks")
                
                team_size = st.slider("Team Size (Engineers)", 1, 5, 3, key="team_size_slider")
                real_duration = weeks_needed_1_eng / team_size
                c3.metric(f"Time ({team_size} Engineers)", f"{real_duration:.1f} Weeks", delta="Project Duration")
                
                # Risk Flag
                if real_duration > 10 or (st.session_state['mode'] == 'quantile' and risk_margin > 50):
                    st.error(" **Risk Alert:** Long duration or high uncertainty (Risk Margin). Review resource allocation.")
                elif i_winter:
                    st.info(" **Seasonal Note:** Estimation includes a factor for winter efficiency/delays.")
                else:
                    st.success(" **Standard Project:** Project fits routine procedures.")
