import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge 
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor 
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
import shap
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Daskan Intelligence | AI Project Estimator",
    page_icon="🏗️",
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
        
        # --- SOPHISTICATION UPGRADE: Advanced Feature Engineering (Complexity Index) ---
        material_factor = 1.0
        if mat == 'Mixed': material_factor = 2.0
        elif mat == 'Steel': material_factor = 1.5
        elif mat == 'Concrete': material_factor = 1.2
            
        # Formula balances vertical complexity (levels^1.2) and material difficulty against scale (area)
        complexity_index = (levels ** 1.2) * (area / levels) / 500 * material_factor
        complexity_index = max(0.5, complexity_index) # Set a minimum baseline
        
        base_effort = (area * 0.08) * complexity_index # Use the new feature to drive effort
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
            'complexity_index': round(complexity_index, 2), # ADDED
            'total_project_effort': round(actual_effort, 2)
        })
        
    df = pd.DataFrame(data)
    # Simulate some missing values for the Imputation demonstration
    df.loc[df.sample(frac=0.05).index, 'surface_area_m2'] = np.nan
    df.loc[df.sample(frac=0.02).index, 'material_type'] = np.nan
    return df

# --- SIDEBAR: DATA INPUT SELECTION ---
with st.sidebar:
    # Changed path to general URL for execution
    st.image("https://cdn-icons-png.flaticon.com/512/2682/2682690.png", width=80) 
    st.markdown("### 🏗️ Daskan Intelligence")
    st.caption("Thesis Version 3.0 - Quantile Regression")
    
    st.divider()
    
    st.subheader("Data Source Selector")
    uploaded_file = st.file_uploader(
        "Upload Daskan's Master CSV (Required columns: surface_area_m2, num_levels, project_type, material_type, is_winter, total_project_effort)", 
        type=['csv']
    )
    
    st.divider()
    
    st.subheader("⚙️ Synthetic Parameters")
    random_seed = st.slider("Random Seed", 1, 100, 50)
    sample_size = st.slider("Dataset Size", 100, 1000, 500)
    
    st.divider()
    # FIX: Added 'r' prefix (raw string) to fix the SyntaxWarning for '\%'
    st.info(r"Target: $90\%$ Prediction Interval Coverage (Risk Management)") 

# --- CONDITIONAL DATA LOADING ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        data_source = "Daskan Inc. Uploaded Data"
        st.success("✅ Successfully loaded uploaded file!")
        
        # Feature Engineering on uploaded data (Ensure required features are calculated if missing)
        if 'floor_area_ratio' not in df.columns and 'surface_area_m2' in df.columns and 'num_levels' in df.columns:
            df['floor_area_ratio'] = df['surface_area_m2'] / df['num_levels']
            st.info("Feature Engineered: 'floor_area_ratio' calculated.")
        
        # Calculate complexity_index for uploaded data
        if all(col in df.columns for col in ['surface_area_m2', 'num_levels', 'material_type']):
            # Define material factor function
            def get_material_factor(mat):
                if mat == 'Mixed': return 2.0
                if mat == 'Steel': return 1.5
                if mat == 'Concrete': return 1.2
                return 1.0

            df['material_factor'] = df['material_type'].apply(get_material_factor)
            
            df['complexity_index'] = (df['num_levels'] ** 1.2) * (df['surface_area_m2'] / df['num_levels']) / 500 * df['material_factor']
            df['complexity_index'] = df['complexity_index'].apply(lambda x: max(0.5, x))
            df.drop('material_factor', axis=1, inplace=True)
            st.info("Feature Engineered: 'complexity_index' calculated for uploaded data.")

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

tabs = st.tabs(["📊 Deep Dive Analytics", "🧹 Data Quality Check", "🧠 AI Model Engine", "💼 Decision Support System"])

# ----------------------------------------------------------------------
# --- TAB 1: INTERACTIVE ANALYTICS (Plotly) ---
# ----------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Data Understanding Phase (RQ1)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Complexity Analysis: Effort vs. Area")
        required_plot_cols = ['surface_area_m2', 'num_levels', 'total_project_effort', 'project_type', 'floor_area_ratio', 'complexity_index']
        
        if all(col in df.columns for col in required_plot_cols):
            fig_scatter = px.scatter(
                df, x='surface_area_m2', y='total_project_effort',
                color='project_type', 
                size='num_levels', 
                hover_data=['floor_area_ratio', 'complexity_index'], 
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
    
    # --- 1. DATA OVERVIEW ---
    st.subheader("Dataset Structure & Statistics")
    col_shape, col_types = st.columns(2)
    with col_shape:
        st.metric("Dataset Shape (Rows, Columns)", f"{df.shape[0]:,} rows, {df.shape[1]} columns")
    with col_types:
        st.dataframe(df.dtypes.rename('DataType').to_frame(), use_container_width=True)
    
    st.markdown("#### Descriptive Statistics")
    st.dataframe(df.describe().T)

    # --- 2. MISSING VALUES REPORT ---
    st.divider()
    st.subheader("Missing Values Report (Imputation Strategy)")
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
         st.warning("⚠️ Missing values detected. They will be handled correctly by the Pipeline in the 'AI Model Engine' tab.")

    # --- 3. OUTLIER DETECTION (NEW) ---
    st.divider()
    st.subheader("Outlier Detection (Numerical Features)")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    selected_num_col = st.selectbox("Select Numerical Feature for Box Plot", numerical_cols)
    
    if selected_num_col:
        fig_box = px.box(
            df, 
            y=selected_num_col, 
            title=f'Box Plot of {selected_num_col}',
            labels={selected_num_col: selected_num_col}
        )
        fig_box.update_layout(margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_box, use_container_width=True)
        st.info("Outliers (dots) indicate values far from the bulk of the data (whiskers). High skewness or large outliers may require data transformation or capping in the pipeline.")

    # --- 4. CATEGORICAL CARDINALITY (NEW) ---
    st.divider()
    st.subheader("Categorical Feature Check")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        cardinality_df = pd.DataFrame({
            'Feature': categorical_cols,
            'Unique Values': [df[col].nunique() for col in categorical_cols],
            'Examples': [', '.join(df[col].dropna().unique()[:3].astype(str)) for col in categorical_cols]
        })
        st.dataframe(cardinality_df, use_container_width=True)
        st.info(
            "Features with very high cardinality (many unique values) should be treated carefully before One-Hot Encoding in the modeling phase."
        )
    else:
        st.info("No categorical features detected (type 'object') for cardinality check.")

# ----------------------------------------------------------------------
# --- TAB 3: AI ENGINE (Advanced - Quantile Regression & HPO) ---
# ----------------------------------------------------------------------
with tabs[2]:
    st.markdown("### Modeling & Evaluation Phase (RQ3: Ensemble & Quantile)")
    
    required_cols = ['surface_area_m2', 'num_levels', 'project_type', 'material_type', 'is_winter', 'floor_area_ratio', 'complexity_index', 'total_project_effort'] 
    
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Cannot train model. The dataset is missing critical columns. Please ensure your uploaded file contains: {', '.join([c for c in required_cols if c not in df.columns])}")
    else:
        # Preprocessing Setup
        X = df.drop(['total_project_effort'], axis=1, errors='ignore')
        y = df['total_project_effort']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        col_config, col_metrics = st.columns([1, 4])
        
        with col_config:
            st.markdown("#### Model Configuration")
            
            # --- SUPER SMART IMPROVEMENT 1: Added Stacked Ensemble ---
            model_type = st.radio(
                "Ensemble/Advanced Algorithm", 
                ["Gradient Boosting Regressor", "Random Forest Regressor", "Stacked Ensemble Regressor", "Linear Regression"]
            )
            
            # Check for compatibility with Quantile Mode
            if model_type not in ["Gradient Boosting Regressor"]:
                 st.session_state['quantile_disabled'] = True
                 prediction_mode = st.radio(
                    "Prediction Objective", 
                    ["Point Estimate (Median)"],
                    disabled=True,
                    help="Quantile Prediction is only supported for Gradient Boosting Regressor."
                )
                 prediction_mode = "Point Estimate (Median)" # Enforce point estimate
            else:
                st.session_state['quantile_disabled'] = False
                prediction_mode = st.radio(
                    "Prediction Objective", 
                    ["Point Estimate (Median)", "Prediction Interval (90% Quantile)"],
                    help="Prediction Interval provides a confidence range for risk management."
                )
            
            scaling_type = st.radio(
                "Numerical Scaling Method", 
                ["Standard Scaler", "MinMax Scaler"], 
                help="Choose scaling technique for numerical features."
            )
            
            perform_hpo = st.checkbox("Perform Hyperparameter Optimization (Slow)", value=False, help="Uses Randomized Search CV for 10 iterations to find optimal parameters.")
            
            params = {}
            if model_type in ["Gradient Boosting Regressor", "Random Forest Regressor"] and not perform_hpo:
                n_est = st.slider("Trees (n_estimators)", 50, 300, 100)
                depth = st.slider("Max Depth", 5, 50, 20)
                params = {'n_estimators': n_est, 'max_depth': depth, 'random_state': 42}
        
        # --- BUILD PREPROCESSOR PIPELINE ---
        numeric_features = ['surface_area_m2', 'num_levels', 'floor_area_ratio', 'is_winter', 'complexity_index'] 
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
                
                # --- HPO Definition ---
                if perform_hpo and model_type in ["Gradient Boosting Regressor", "Random Forest Regressor"]:
                    with st.spinner('Searching for optimal hyperparameters (10 iterations, 3-fold CV)...'):
                        param_dist = {
                            'regressor__n_estimators': [50, 100, 200, 300],
                            'regressor__max_depth': [5, 10, 15, 20],
                            'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2] if model_type == "Gradient Boosting Regressor" else [None],
                        }
                        
                        base_model = RandomForestRegressor(random_state=42) if model_type == "Random Forest Regressor" else GradientBoostingRegressor(random_state=42)
                        
                        hpo_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', base_model)])
                        
                        random_search = RandomizedSearchCV(
                            hpo_pipeline, param_distributions=param_dist, n_iter=10, 
                            cv=3, scoring='neg_mean_absolute_error', random_state=42, verbose=0, n_jobs=-1
                        )
                        random_search.fit(X_train, y_train)
                        best_params = {k.replace('regressor__', ''): v for k, v in random_search.best_params_.items()}
                        st.success(f"Best HPO Params Found: {best_params}")
                        params = best_params

                
                # --- Quantile Regression Logic (Trains 3 Models) ---
                if prediction_mode == "Prediction Interval (90% Quantile)" and model_type == "Gradient Boosting Regressor":
                    st.session_state['mode'] = 'quantile'
                    
                    models = {}
                    quantiles = {'Q05': 0.05, 'Q50': 0.50, 'Q95': 0.95}
                    
                    with st.spinner(f"Training 3 {model_type} models..."):
                        for name, alpha in quantiles.items():
                            gbr = GradientBoostingRegressor(loss='quantile', alpha=alpha, **params)
                            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', gbr)])
                            pipeline.fit(X_train, y_train)
                            models[name] = pipeline
                            
                        pipeline_median = models['Q50']
                        preds_median = pipeline_median.predict(X_test)
                        
                        mae = mean_absolute_error(y_test, preds_median)
                        r2 = r2_score(y_test, preds_median)
                        
                        preds_q05 = models['Q05'].predict(X_test)
                        preds_q95 = models['Q95'].predict(X_test)
                        coverage = np.mean((y_test >= preds_q05) & (y_test <= preds_q95))
                        
                        # Display metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Median Accuracy (R²)", f"{r2:.3f}", delta="Target: >0.70")
                        m2.metric("MAE (Q50 Error)", f"{mae:.1f} Hours", delta_color="inverse")
                        m3.metric("PICP (Coverage)", f"{coverage:.1%}", delta=f"Target: 90.0%")
                        
                        st.session_state['adv_model'] = models
                        st.session_state['model_trained'] = True
                        st.session_state['pipeline_median'] = pipeline_median
                        st.success("✅ 90% Quantile Prediction Intervals (Q05, Q50, Q95) Trained Successfully!")
                    
                # --- Point Estimate Logic (Trains 1 Model/Stacking) ---
                else: 
                    st.session_state['mode'] = 'point'
                    
                    with st.spinner(f"Training 1 {model_type} model..."):
                        if model_type == "Gradient Boosting Regressor":
                            model = GradientBoostingRegressor(**params)
                        elif model_type == "Random Forest Regressor":
                            model = RandomForestRegressor(**params)
                        elif model_type == "Stacked Ensemble Regressor":
                            # Define the sophisticated base estimators
                            estimators = [
                                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
                                ('gbr', GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)),
                                ('lr', LinearRegression())
                            ]
                            # Use StackingRegressor with a Ridge meta-estimator for robustness
                            model = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0))
                            st.info("🧠 **Advanced Stacking:** Combining Random Forest, Gradient Boosting, and Linear Regression with a Ridge final estimator.")
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
                        st.session_state['pipeline_median'] = pipeline
                        st.success(f"✅ {model_type} Point Estimate Model Trained Successfully!")
                        
                    plot_preds = preds # Define plot_preds for the next block

                # --- Residual Plot (Actual vs. Predicted) ---
                if st.session_state.get('model_trained', False):
                    st.subheader("Residual Analysis (Actual vs. Predicted)")
                    
                    # Ensure plot_preds is correctly set from the Quantile or Point logic
                    if st.session_state['mode'] == 'quantile':
                        # Use preds_median defined in the quantile block
                        plot_preds = locals().get('preds_median')
                    
                    if plot_preds is not None:
                        fig_res = go.Figure()
                        fig_res.add_trace(go.Scatter(x=y_test, y=plot_preds, mode='markers', name='Predictions', marker=dict(color='#049449', opacity=0.6)))
                        fig_res.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', name='Perfect Prediction', line=dict(color='black', dash='dash')))
                        fig_res.update_layout(xaxis_title="Actual Hours", yaxis_title="Predicted Hours")
                        st.plotly_chart(fig_res, use_container_width=True)

                        # --- SUPER SMART IMPROVEMENT 2: Error Distribution Plot ---
                        st.subheader("Error Distribution (Residuals)")
                        residuals = y_test - plot_preds
                        
                        fig_res_hist = px.histogram(
                            x=residuals,
                            nbins=50,
                            title='Distribution of Prediction Errors (Residuals)',
                            labels={'x': 'Residual (Actual - Predicted Hours)'},
                            color_discrete_sequence=['#049449']
                        )
                        fig_res_hist.update_layout(showlegend=False)
                        st.plotly_chart(fig_res_hist, use_container_width=True)
                        st.info(f"""
                        **Error Interpretation:** The distribution should be centered around zero.
                        * **Mean Residual (Bias):** **{residuals.mean():.2f} hours** (Ideally close to zero).
                        * **Standard Deviation:** **{residuals.std():.2f} hours** (The typical magnitude of error).
                        """)
                    else:
                        st.warning("Cannot generate Residual Plot: Prediction values are missing.")

            else:
                 st.info("Click 'Train Model(s) on Current Data' to run the pipeline and evaluate performance.")

        # --------------------------------------------------------------------------
        # --- SHAP Implementation for GBR/RF/Stacking ---
        # --------------------------------------------------------------------------
        if st.session_state.get('model_trained', False) and model_type in ["Gradient Boosting Regressor", "Random Forest Regressor", "Stacked Ensemble Regressor"]:
            st.divider()
            st.markdown(f"### 💡 Model Explainability (XAI) using SHAP")
            
            with st.spinner('Calculating SHAP values for explainability...'):
                pipeline_to_explain = st.session_state['pipeline_median'] 
                
                # 1. Get the processed test data (Needed for the background and feature names)
                X_test_processed = pipeline_to_explain.named_steps['preprocessor'].transform(X_test)

                # 2. Get the processed background data (small subset of training data for Kernel Explainer)
                X_train_processed = pipeline_to_explain.named_steps['preprocessor'].transform(X_train)
                # Use k-means sampling for a representative background (faster than passing full training set)
                background_data = shap.maskers.Independent(X_train_processed, max_samples=100)
                
                # 3. Get the feature names after one-hot encoding
                preprocessor_fitted = pipeline_to_explain.named_steps['preprocessor']
                try:
                    ohe_cols = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
                except Exception:
                    ohe_cols = preprocessor_fitted.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
                
                all_feats = numeric_features + list(ohe_cols)
                
                # 4. Instantiate Explainer based on Model Type
                if model_type in ["Gradient Boosting Regressor", "Random Forest Regressor"]:
                    # Tree Explainer works directly on the regressor and processed data
                    explainer = shap.TreeExplainer(pipeline_to_explain.named_steps['regressor'])
                    shap_values = explainer.shap_values(X_test_processed)
                
                elif model_type == "Stacked Ensemble Regressor":
                    # FIX: Use KernelExplainer with the processed background and the regressor's predict function.
                    # This ensures only numerical data is used in the SHAP kernel logic.
                    model_regressor_predict = pipeline_to_explain.named_steps['regressor'].predict
                    
                    explainer = shap.KernelExplainer(model_regressor_predict, background_data)
                    
                    # Compute SHAP values using PROCESSED test data
                    shap_values = explainer.shap_values(X_test_processed) 
                    
                    st.warning("⏳ SHAP for Stacked Regressor uses a slower, model-agnostic approach (**KernelExplainer**) with a processed background to correctly interpret the ensemble. This may take a significant amount of time.")
                
                st.subheader(f"SHAP Summary Plot (Global Feature Impact on {model_type} Estimate)")
                
                fig, ax = plt.subplots(figsize=(10, 6)) 
                
                # SHAP Summary Plot uses the processed data and computed shap values
                shap.summary_plot(
                    shap_values, 
                    X_test_processed, 
                    feature_names=all_feats, 
                    show=False, 
                    max_display=15, 
                    color=plt.get_cmap("viridis")
                )
                
                ax.set_title(f"SHAP Summary Plot - {len(X_test)} Test Samples")
                
                st.pyplot(fig)
                plt.close(fig)
                
                st.info(f"""
                **SHAP Interpretation:** This plot shows the magnitude and direction of each feature's effect on the model's prediction. 
                * The **'complexity_index'** should be a top driver, reflecting the success of the feature engineering phase.
                """)


# ----------------------------------------------------------------------
# --- TAB 4: DECISION SUPPORT (Business Value - Prediction Interval) ---
# ----------------------------------------------------------------------
with tabs[3]:
    st.markdown("### Workforce Planning & Quotation (Business Objective)")
    
    # Initialize session state for prediction results if they don't exist
    if 'ss_pred_hours' not in st.session_state:
        st.session_state['ss_pred_hours'] = None
    if 'ss_risk_margin' not in st.session_state:
        st.session_state['ss_risk_margin'] = None
    
    if not st.session_state.get('model_trained', False):
        st.warning("⚠️ Please train the model in the 'AI Model Engine' tab first.")
    else:
        col_in, col_out = st.columns([1, 2])
        
        # --- COLUMN 1: INPUTS ---
        with col_in:
            st.markdown("#### New Project Parameters")
            i_area = st.number_input("Surface Area (m²)", 1, 10000, 650, key='i_area')
            i_levels = st.number_input("Levels", 1, 50, 3, key='i_levels')
            i_type = st.selectbox("Type", df['project_type'].unique(), key='i_type')
            i_mat = st.selectbox("Material", df['material_type'].unique(), key='i_mat')
            i_winter = st.checkbox("Winter Schedule (Dec-Mar)?", key='i_winter')
            
            # --- Complex Feature Calculation ---
            material_factor = 1.0
            if i_mat == 'Mixed': material_factor = 2.0
            elif i_mat == 'Steel': material_factor = 1.5
            elif i_mat == 'Concrete': material_factor = 1.2
            
            i_complexity_index = (i_levels ** 1.2) * (i_area / i_levels) / 500 * material_factor
            i_complexity_index = max(0.5, i_complexity_index)
            
            # Prepare Input DF
            input_df = pd.DataFrame({
                'project_type': [i_type],
                'material_type': [i_mat],
                'surface_area_m2': [i_area],
                'num_levels': [i_levels],
                'is_winter': [1 if i_winter else 0],
                'floor_area_ratio': [i_area/i_levels],
                'complexity_index': [i_complexity_index]
            }, index=[0])

        # --- COLUMN 2: OUTPUT & BUTTON ---
        with col_out:
            if st.button("Calculate Prediction for Input Parameters"):
                
                # --- Prediction Logic ---
                if st.session_state['mode'] == 'quantile':
                    models = st.session_state['adv_model']
                    pred_q05 = models['Q05'].predict(input_df)[0]
                    pred_q50 = models['Q50'].predict(input_df)[0]
                    pred_q95 = models['Q95'].predict(input_df)[0]
                    
                    pred_hours = pred_q50 
                    risk_margin = pred_q95 - pred_q50
                    
                    # Store results in session state for the reactive planner
                    st.session_state['ss_pred_hours'] = pred_hours
                    st.session_state['ss_risk_margin'] = risk_margin
                    st.session_state['ss_mode'] = 'quantile'
                    st.session_state['ss_pred_q05'] = pred_q05
                    st.session_state['ss_pred_q95'] = pred_q95

                    # Display the main card and risk metrics immediately on click
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="margin:0; color:#1A1A1A;">{pred_q50:.1f} Hours</h2> <p style="margin:0;">Median Effort Estimate (50th Percentile)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("### 🎯 Risk-Managed Quotation (RQ4)")
                    c1, c2 = st.columns(2)
                    c1.metric("90% Prediction Interval", f"{pred_q05:.0f} - {pred_q95:.0f} Hours")
                    c2.metric("Worst-Case Risk Margin", f"{risk_margin:.1f} Hours", delta="Buffer needed above median", delta_color='off')

                else: # Point Estimate Mode (including Stacking)
                    pipeline = st.session_state['adv_model']
                    pred_hours = pipeline.predict(input_df)[0]
                    
                    # Store results in session state for the reactive planner
                    st.session_state['ss_pred_hours'] = pred_hours
                    st.session_state['ss_risk_margin'] = 0
                    st.session_state['ss_mode'] = 'point'
                    
                    # Display the main card immediately on click
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="margin:0; color:#1A1A1A;">{pred_hours:.1f} Hours</h2> <p style="margin:0;">Point Effort Estimate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success("Prediction calculated. Adjust team size below for scheduling.")

        # --- REACTIVE WORKFORCE PLANNER (OUTSIDE the button block) ---
        if st.session_state['ss_pred_hours'] is not None:
            # Retrieve stored values
            pred_hours = st.session_state['ss_pred_hours']
            risk_margin = st.session_state['ss_risk_margin']
            ss_mode = st.session_state['ss_mode']
            
            # Re-display Quotation metrics for context (optional, but helpful when adjusting slider)
            if ss_mode == 'quantile':
                # Re-display the main cards
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style="margin:0; color:#1A1A1A;">{pred_hours:.1f} Hours</h2> <p style="margin:0;">Median Effort Estimate (50th Percentile)</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("### 🎯 Risk-Managed Quotation (RQ4)")
                c1_q, c2_q = st.columns(2)
                c1_q.metric("90% Prediction Interval", f"{st.session_state['ss_pred_q05']:.0f} - {st.session_state['ss_pred_q95']:.0f} Hours")
                c2_q.metric("Worst-Case Risk Margin", f"{risk_margin:.1f} Hours", delta="Buffer needed above median", delta_color='off')
            elif ss_mode == 'point':
                 # Re-display the main card
                 st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="margin:0; color:#1A1A1A;">{pred_hours:.1f} Hours</h2> <p style="margin:0;">Point Effort Estimate</p>
                    </div>
                    """, unsafe_allow_html=True)


            st.subheader("📅 Workforce Planner")
            
            # Shared business logic constants
            avg_hourly_rate = 120 # CAD
            hours_per_week = 40
            
            # --- SLIDER: Max value 100 ---
            team_size = st.slider("Team Size (Engineers)", 1, 100, 2, key="team_size_slider_calc_reactive")
            
            estimated_cost = pred_hours * avg_hourly_rate
            weeks_needed_1_eng = pred_hours / hours_per_week
            
            # Dynamic calculation based on team_size
            real_duration = weeks_needed_1_eng / team_size
            
            c1_p, c2_p, c3_p = st.columns(3)
            
            c1_p.metric("Estimated Cost (CAD)", f"${estimated_cost:,.0f}")
            c2_p.metric("Time (1 Engineer)", f"{weeks_needed_1_eng:.1f} Weeks")
            c3_p.metric(f"Time ({team_size} Engineers)", f"{real_duration:.1f} Weeks", delta="Project Duration")
            
            # Risk Flag
            current_i_winter = st.session_state['i_winter'] 
            
            if real_duration > 10 or (ss_mode == 'quantile' and risk_margin > 50):
                st.error("⚠️ **Risk Alert:** Long duration or high uncertainty (Risk Margin). Review resource allocation.")
            elif current_i_winter:
                st.info("❄️ **Seasonal Note:** Estimation includes a factor for winter efficiency/delays.")
            else:
                st.success("✅ **Standard Project:** Project fits routine procedures.")

        else:
            st.info("Set input parameters above and click 'Calculate Prediction for Input Parameters' to generate the prediction and unlock the reactive planner.")