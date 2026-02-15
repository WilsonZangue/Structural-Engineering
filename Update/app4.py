import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
from datetime import datetime, timedelta

# ML Libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor 
from sklearn.linear_model import Ridge 
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import randint, uniform

# --- 1. CONFIGURATION & CONSTANTS ---
# Daskan Green Accent
DASKAN_GREEN = "#049449" 
DASKAN_COLOR_PALETTE = ['#049449', '#004c29', '#1abc9c', '#3498db', '#9b59b6']
AVG_HOURLY_RATE_CAD = 115
HOURS_PER_WEEK = 30 # For a single engineer

st.set_page_config(
    page_title="Daskan Intelligence | AI Project Estimator - PRO",
    page_icon="cropped-DaskanLogo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS ---
st.markdown(f"""
<style>
    /* Metric Card Border (Daskan Green Accent) */
    .metric-card {{
        background-color: #f0f2f6;
        border-left: 5px solid {DASKAN_GREEN};
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 55px;
        white-space: pre-wrap;
        background-color: #EAEAEA;
        border-radius: 6px 6px 0px 0px;
        gap: 1px;
        padding-top: 15px;
        padding-bottom: 15px;
        font-weight: 600;
        color: #333333;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #FFFFFF;
        border-bottom: 4px solid {DASKAN_GREEN};
        color: {DASKAN_GREEN};
        font-weight: 800;
    }}
    .stSpinner > div {{
        color: {DASKAN_GREEN};
    }}
</style>
""", unsafe_allow_html=True)

# --- 3. CORE FUNCTIONS ---

@st.cache_data
def generate_granular_synthetic_data(num_projects=100, seed=42):
    """Generates synthetic granular timesheet data."""
    np.random.seed(seed)
    
    project_types = ['Residential', 'Commercial', 'Institutional', 'Industrial']
    materials = ['Wood', 'Steel', 'Concrete', 'Mixed']
    projects = []
    project_effort_map = {} 
    
    for i in range(num_projects):
        p_id = f"P-2022-{str(i+1).zfill(3)}"
        p_type = np.random.choice(project_types, p=[0.25, 0.35, 0.15, 0.25])
        
        if p_type == 'Residential': area = np.random.randint(100, 600); levels = np.random.randint(1, 4)
        elif p_type == 'Commercial': area = np.random.randint(500, 2000); levels = np.random.randint(2, 8)
        else: area = np.random.randint(1000, 5000); levels = np.random.randint(1, 15)
            
        start_month = np.random.randint(1, 12)
        start_date = datetime(2022, start_month, 1) + timedelta(days=np.random.randint(0, 28))
        
        # Calc Effort (Hidden Truth)
        base_effort = (area * 0.1) + (levels * 20)
        if p_type in ['Institutional', 'Industrial']: base_effort *= 1.5
        total_hours_est = int(np.random.normal(base_effort, base_effort * 0.15))
        total_hours_est = max(10, total_hours_est)
        project_effort_map[p_id] = total_hours_est
        
        estimated_weeks = max(1, total_hours_est / 30) 
        end_date = start_date + timedelta(days=int(estimated_weeks * 7) + np.random.randint(5, 20))

        projects.append({
            'project_id': p_id,
            'project_type': p_type,
            'material_type': np.random.choice(materials),
            'surface_area_m2': area,
            'num_levels': levels,
            'start_date': start_date,
            'end_date': end_date
        })
        
    df_projects = pd.DataFrame(projects)
    
    # Timesheets
    timesheet_entries = []
    log_id_counter = 1
    
    for _, proj in df_projects.iterrows():
        p_id = proj['project_id']
        total_hours = project_effort_map[p_id]
        num_entries = np.random.randint(5, 60)
        avg_entry_hours = total_hours / num_entries
        date_range_days = (proj['end_date'] - proj['start_date']).days
        
        for _ in range(num_entries):
            hours = round(np.random.normal(avg_entry_hours, 1.5), 2)
            if hours <= 0.25: hours = 0.5
            random_days = np.random.randint(0, max(1, date_range_days))
            log_date = proj['start_date'] + timedelta(days=random_days)
            
            timesheet_entries.append({
                'log_id': log_id_counter,
                'project_id': p_id,
                'employee_id': f"EMP-{np.random.randint(1, 12)}",
                'date_logged': log_date,
                'task_category': np.random.choice(['Design', 'Calculation', 'Drafting', 'Meeting', 'Site Visit'], p=[0.25, 0.25, 0.3, 0.1, 0.1]),
                'hours_worked': hours
            })
            log_id_counter += 1
            
    df_timesheets = pd.DataFrame(timesheet_entries)
    
    # Merge
    df_master = pd.merge(df_timesheets, df_projects, on='project_id', how='left')
    df_master['date_logged'] = pd.to_datetime(df_master['date_logged'])
    return df_master

def feature_engineer_data(df_analytics):
    """Aggregates granular data to project level and creates complex features."""
    if 'log_id' not in df_analytics.columns:
        # Already summary data (though this case is rare with the generator)
        df_modeling = df_analytics.copy()
        
    else:
        # Aggregate granular logs to get total effort
        df_modeling = df_analytics.groupby('project_id').agg({
            'surface_area_m2': 'first',
            'num_levels': 'first',
            'project_type': 'first',
            'material_type': 'first',
            'start_date': 'min',
            'end_date': 'max',
            'hours_worked': 'sum' # The Target Variable
        }).rename(columns={'hours_worked': 'total_project_effort'}).reset_index()

        # Calculate Winter Flag: Logs in Dec, Jan, Feb, Mar (Quebec winter is long)
        winter_projs = df_analytics[df_analytics['date_logged'].dt.month.isin([12,1,2,3])]['project_id'].unique()
        df_modeling['is_winter'] = df_modeling['project_id'].isin(winter_projs).astype(int)

    # Derived Features (applies to both aggregated and summary data)
    df_modeling['floor_area_ratio'] = df_modeling['surface_area_m2'] / df_modeling['num_levels']
    
    # Calculate Advanced Complexity Index
    def get_complexity(row):
        mat_factor = {'Mixed':2.0, 'Steel':1.5, 'Concrete':1.2}.get(row['material_type'], 1.0)
        # Formula: Non-linear penalty for height * Area/Level * Material factor
        c_idx = (row['num_levels']**1.2) * (row['surface_area_m2']/row['num_levels']) / 500 * mat_factor
        return max(0.5, c_idx)
    
    df_modeling['complexity_index'] = df_modeling.apply(get_complexity, axis=1)
    
    return df_modeling

@st.cache_resource(show_spinner=False)
def train_and_tune_model(X_train, y_train, model_choice, use_quantile):
    """Trains a tuned model using Randomized Search CV or a Stacking Regressor."""
    
    num_features = ['surface_area_m2', 'num_levels', 'floor_area_ratio', 'complexity_index', 'is_winter']
    cat_features = ['project_type', 'material_type']
    
    # Preprocessor (Standardization for numerical, One-Hot for categorical)
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])

    if use_quantile and model_choice == "Gradient Boosting":
        st.session_state['mode'] = 'quantile'
        models = {}
        common_params = {'n_estimators': 150, 'max_depth': 5, 'random_state': 42}
        
        for name, alpha in [('Q05', 0.05), ('Q50', 0.5), ('Q95', 0.95)]:
            gbr = GradientBoostingRegressor(loss='quantile', alpha=alpha, **common_params)
            pipe = Pipeline([('prep', preprocessor), ('model', gbr)])
            pipe.fit(X_train, y_train)
            models[name] = pipe
        
        tuned_model = models
        r2 = r2_score(y_train, models['Q50'].predict(X_train)) # Evaluate on Q50 (Median)
        
    elif model_choice == "Stacking Regressor":
        st.session_state['mode'] = 'stacking'
        
        # 1. Define the base models (estimators)
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
            ('gbr', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42))
        ]
        
        # 2. Define the Stacking Regressor
        base_model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0), 
            cv=3,
            n_jobs=-1
        )
        
        pipe = Pipeline([('prep', preprocessor), ('model', base_model)])
        pipe.fit(X_train, y_train)
        tuned_model = pipe # The fitted Stacking Pipeline
        
        # Evaluation on training set
        r2 = r2_score(y_train, tuned_model.predict(X_train))
        
    else: # Random Forest or Standard Gradient Boosting (point mode)
        st.session_state['mode'] = 'point'
        
        # 1. Define Base Model and Parameter Grid for Random Search
        if model_choice == "Random Forest":
            base_model = RandomForestRegressor(random_state=42)
            param_dist = {
                'model__n_estimators': randint(100, 300),
                'model__max_depth': randint(5, 15),
                'model__min_samples_split': uniform(0.01, 0.1),
            }
        elif model_choice == "Gradient Boosting":
            base_model = GradientBoostingRegressor(random_state=42)
            param_dist = {
                'model__n_estimators': randint(100, 300),
                'model__max_depth': randint(3, 7),
                'model__learning_rate': uniform(0.01, 0.2),
            }
        
        pipe = Pipeline([('prep', preprocessor), ('model', base_model)])
        
        # 2. Perform Randomized Search CV
        random_search = RandomizedSearchCV(
            pipe, 
            param_distributions=param_dist, 
            n_iter=10, 
            cv=3, 
            scoring='neg_mean_absolute_error', 
            random_state=42, 
            n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        tuned_model = random_search.best_estimator_
        
        # 3. Evaluation
        r2 = r2_score(y_train, tuned_model.predict(X_train)) # Using R2 on training set for display
        
    return tuned_model, r2, st.session_state['mode']


@st.cache_data(show_spinner="Calculating SHAP values...")
def get_shap_data(_models, X_train):
    """Caches the expensive SHAP calculation."""
    
    # Get the trained model (either RF, GBR, or the final estimator from Stacking)
    if st.session_state['mode'] == 'quantile':
        model_pipe = _models['Q50'] 
    elif st.session_state['mode'] == 'stacking':
        # For Stacking, SHAP is often run on the final estimator (Meta-Model)
        # Note: This is an approximation since the input is preprocessed data.
        model_pipe = _models 
    else:
        model_pipe = _models

    # Isolate the preprocessor and the final estimator
    preprocessor = model_pipe.named_steps['prep']
    final_estimator = model_pipe.named_steps['model']
    
    # Get preprocessed training data
    X_train_transformed = preprocessor.transform(X_train)
    
    # Get feature names after one-hot encoding
    feature_names = preprocessor.transformers_[0][2] + list(preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(input_features=['project_type', 'material_type']))

    # Create an independent background data for SHAP
    explainer = shap.TreeExplainer(final_estimator)
    shap_values = explainer.shap_values(X_train_transformed)
    
    return explainer, shap_values, X_train_transformed, feature_names


# --- 4. STREAMLIT APPLICATION LAYOUT ---

# Initialize Session State
if 'models' not in st.session_state:
    st.session_state['models'] = None
    st.session_state['mode'] = 'point'
    st.session_state['train_r2'] = 0.0
    st.session_state['i_winter'] = 0
    st.session_state['quote_generated'] = False
    st.session_state['pred_val'] = 0.0
    st.session_state['y_test'] = None
    st.session_state['y_preds'] = None
    st.session_state['approved_model_version'] = None
    st.session_state['approved_r2'] = 0.0
    st.session_state['train_stats'] = None


# Sidebar for Data Control
with st.sidebar:
    st.image("cropped-DaskanLogo.png", width=500) 
    st.markdown(f"### Daskan Intelligence")
    st.caption("v1.0 - Powered by Machine Learning")
    st.divider()
    
    st.subheader("Data Source")
    uploaded_file = st.file_uploader("Upload CSV data file (Granular)", type=['csv'])
    
    st.divider()
    st.subheader(" Data Generation")
    random_seed = st.slider("Seed", 1, 100, 42)
    sample_size = st.slider("Projects to Generate", 50, 500, 200)

# Data Loading & Processing
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        if 'date_logged' in raw_df.columns:
            # FIX: Explicitly set dayfirst=True to handle DD/MM/YYYY and silence the UserWarning
            raw_df['date_logged'] = pd.to_datetime(raw_df['date_logged'], dayfirst=True)
        df_analytics = raw_df.copy()
        data_source = "Uploaded Granular Data"
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        df_analytics = generate_granular_synthetic_data(sample_size, random_seed)
        data_source = "Synthetic Data (Fallback)"
else:
    df_analytics = generate_granular_synthetic_data(sample_size, random_seed)
    data_source = f"Synthetic Data ({sample_size} Projects)"

# Feature Engineering
df_modeling = feature_engineer_data(df_analytics)

# Calculate Training Data Statistics for Drift Monitoring
num_cols = ['surface_area_m2', 'num_levels', 'floor_area_ratio', 'complexity_index']
# Ensure we store the stats after feature engineering
st.session_state['train_stats'] = df_modeling[num_cols].agg(['mean', 'std']).T


# --- MAIN APP TITLE ---
st.title(" Structural Project Intelligence Dashboard")
st.markdown(f"**Current Data Source:** `{data_source}` | **Total Projects:** `{len(df_modeling)}`")
st.divider()

tabs = st.tabs([" Deep Dive Analytics", " AI Model Engine", " Model Explainability (XAI)", " Smart Quotation"])

# ----------------------------------------------------------------------
# TAB 1: ANALYTICS
# ----------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Historical Project Insights")
    
    col_kpis_1, col_kpis_2, col_kpis_3 = st.columns(3)
    col_kpis_1.metric("Average Project Effort", f"{df_modeling['total_project_effort'].mean():.0f} Hours")
    col_kpis_2.metric("Most Common Type", df_modeling['project_type'].mode()[0])
    col_kpis_3.metric("Max Complexity Index", f"{df_modeling['complexity_index'].max():.2f}")

    c_m1, c_m2 = st.columns([2, 1])
    
    with c_m1:
        st.subheader("Macro View: Effort vs. Scale")
        fig = px.scatter(df_modeling, x='surface_area_m2', y='total_project_effort', 
                         color='project_type', size='complexity_index',
                         hover_data=['num_levels', 'material_type'], 
                         title="Project Effort (Hours) by Area and Complexity",
                         color_discrete_sequence=DASKAN_COLOR_PALETTE)
        st.plotly_chart(fig, use_container_width=True)
        
    with c_m2:
        st.subheader("Distribution")
        # Effort Distribution
        fig_hist = px.histogram(df_modeling, x='total_project_effort', color_discrete_sequence=[DASKAN_GREEN], 
                                title="Total Effort Distribution")
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    
    # Feature Correlation Heatmap
    st.markdown("###  Feature Correlation Analysis")
    st.caption("Understand linear relationships between numerical features and project effort.")
    
    numerical_features = ['surface_area_m2', 'num_levels', 'floor_area_ratio', 'complexity_index', 'total_project_effort']
    
    # Calculate correlation matrix
    corr_matrix = df_modeling[numerical_features].corr()
    
    # Plot heatmap
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto",
        color_continuous_scale="RdBu_r", # Diverging color scale for correlation
        title="Correlation Heatmap of Numerical Features",
        x=corr_matrix.columns,
        y=corr_matrix.columns
    )
    fig_corr.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # --- CAPTION ADDED HERE ---
    st.caption(
        "**Understanding the Heatmap:** This chart displays the **linear correlation** between all numerical features. "
        "Values range from **-1.0 to +1.0**."
        "\n\n"
        "**🔵 Negative Correlation (Blue, closer to -1.0):** As one feature **increases**, the other tends to **decrease** (e.g., lower `floor_area_ratio` is weakly associated with higher `complexity_index`)."
        "\n\n"
        "**🔴 Positive Correlation (Red, closer to +1.0):** As one feature **increases**, the other tends to **increase** (e.g., higher `complexity_index` strongly drives higher `total_project_effort`). The goal is to identify features that strongly correlate with the target variable (`total_project_effort`)."
    )
    # --- END CAPTION ADDED ---

    st.divider()
    
    # Granular Deep Dive (S-Curve & Task)
    st.markdown("###  Project Timeline Analysis (S-Curve)")
    col_g1, col_g2 = st.columns([2, 1])
    
    with col_g1:
        selected_proj = st.selectbox("Select Project ID for Deep Dive", df_modeling['project_id'].unique(), key='anal_proj_select')
        
        proj_data = df_analytics[df_analytics['project_id'] == selected_proj].copy()
        daily_df = proj_data.groupby('date_logged')['hours_worked'].sum().reset_index().sort_values('date_logged')
        daily_df['cumulative'] = daily_df['hours_worked'].cumsum()
        
        fig_burn = go.Figure()
        fig_burn.add_trace(go.Bar(x=daily_df['date_logged'], y=daily_df['hours_worked'], name='Daily Hours', marker_color='#f39c12', opacity=0.6))
        fig_burn.add_trace(go.Scatter(x=daily_df['date_logged'], y=daily_df['cumulative'], name='Cumulative (S-Curve)', line=dict(color=DASKAN_GREEN, width=3)))
        
        fig_burn.update_layout(title=f"Progress Timeline: {selected_proj}", xaxis_title="Date", yaxis_title="Hours", hovermode="x unified")
        st.plotly_chart(fig_burn, use_container_width=True)
        
    with col_g2:
        st.subheader("Task Breakdown")
        
        fig_task = px.pie(proj_data, names='task_category', values='hours_worked', hole=0.4, 
                         title=f"Tasks: {selected_proj}", color_discrete_sequence=DASKAN_COLOR_PALETTE)
        st.plotly_chart(fig_task, use_container_width=True)
        
        st.metric("Total Effort Logged", f"{daily_df['cumulative'].max():.0f} Hours")
        st.metric("Unique Employees", proj_data['employee_id'].nunique())

# ----------------------------------------------------------------------
# TAB 2: AI ENGINE
# ----------------------------------------------------------------------
with tabs[1]:
    st.header(" Model Training & Tuning")
    
    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    
    X = df_modeling[['surface_area_m2', 'num_levels', 'floor_area_ratio', 'complexity_index', 'is_winter', 'project_type', 'material_type']]
    y = df_modeling['total_project_effort']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with col_m1:
        st.subheader("Configuration")
        # ADDED Stacking Regressor
        model_choice = st.selectbox("Algorithm", ["Gradient Boosting", "Random Forest", "Stacking Regressor"], key='model_select')
        use_quantile = st.checkbox("Use Quantile Regression (Risk Intervals)", value=True, key='use_quantile')
        
    with col_m2:
        st.subheader("Hyperparameters")
        # Disabled for Stacking
        is_random_search_disabled = use_quantile or model_choice == "Stacking Regressor"
        num_iters = st.number_input("Random Search Iterations (n_iter)", 5, 20, 10, disabled=is_random_search_disabled, key='num_iters')
        
        if model_choice == "Random Forest":
            st.info("Tuning: n_estimators (100-300), max_depth (5-15).")
        elif model_choice == "Stacking Regressor":
            st.info("Fixed base estimators (RF, GBR) with Ridge meta-model.")
        else: # Gradient Boosting
            if use_quantile:
                st.info("Using fixed GBR parameters for Quantile Regression (loss='quantile').")
            else:
                st.info("Tuning: n_estimators (100-300), max_depth (3-7), learning_rate.")
            
    with col_m3:
        st.subheader("Action")
        st.markdown("---")
        if st.button(" Train & Tune Model", type="primary", use_container_width=True):
            st.session_state['quote_generated'] = False # Reset quote when training new model
            with st.spinner("Executing Training & Hyperparameter Search..."):
                try:
                    tuned_model, performance_metric, mode = train_and_tune_model(
                        X_train, y_train, model_choice, use_quantile
                    )
                    st.session_state['models'] = tuned_model
                    st.session_state['mode'] = mode
                    
                    if mode == 'quantile':
                        # Evaluate Median (Q50) on Test Set
                        preds = tuned_model['Q50'].predict(X_test)
                        r2_test = r2_score(y_test, preds)
                        mae_test = mean_absolute_error(y_test, preds)
                        st.session_state['train_r2'] = r2_test # Store R2 for display
                        st.metric("Test R² (Median Model)", f"{r2_test:.3f}", delta_color="normal")
                        st.metric("Test MAE (Median Model)", f"{mae_test:.0f} Hours")
                        
                    elif mode == 'stacking':
                        # Evaluate Stacking Model on Test Set
                        preds = tuned_model.predict(X_test)
                        r2_test = r2_score(y_test, preds)
                        mae_test = mean_absolute_error(y_test, preds)
                        st.session_state['train_r2'] = r2_test # Store R2 for display
                        st.metric("Test R² (Stacking Model)", f"{r2_test:.3f}")
                        st.metric("Test MAE (Stacking Model)", f"{mae_test:.0f} Hours")

                    else:
                        # Evaluate Tuned Model on Test Set (RF or GBR Point)
                        preds = tuned_model.predict(X_test)
                        r2_test = r2_score(y_test, preds)
                        mae_test = mean_absolute_error(y_test, preds)
                        st.session_state['train_r2'] = r2_test # Store R2 for display
                        st.metric("Test R² (Tuned Model)", f"{r2_test:.3f}")
                        st.metric("Test MAE (Tuned Model)", f"{mae_test:.0f} Hours")
                    
                    # Store predictions for visualization and MLOps simulation
                    st.session_state['y_test'] = y_test
                    st.session_state['y_preds'] = preds
                    
                    st.success(" Model Training & Tuning Complete!")
                except Exception as e:
                    st.error(f"Training Error: {e}")
        
    st.markdown("---")
    st.info(f"Current Model: **{model_choice}** | Mode: **{st.session_state['mode'].upper()}** | Test R²: **{st.session_state['train_r2']:.3f}**")
    
    # --- Advanced Error Analysis Visualization ---
    if st.session_state['models'] is not None and st.session_state.get('y_preds') is not None:
        
        st.subheader("Test Set Error Analysis: Predicted vs. Actual")
        
        # Create a DataFrame for plotting
        pred_df = pd.DataFrame({
            'Actual Effort (h)': st.session_state['y_test'],
            'Predicted Effort (h)': st.session_state['y_preds'],
        })
        
        # Determine plot limits for a square plot with the 45-degree line
        max_val = max(pred_df['Actual Effort (h)'].max(), pred_df['Predicted Effort (h)'].max()) * 1.05
        
        fig = go.Figure()

        # 1. Add Scatter points
        fig.add_trace(go.Scatter(
            x=pred_df['Actual Effort (h)'], 
            y=pred_df['Predicted Effort (h)'], 
            mode='markers', 
            name='Test Projects', 
            marker=dict(color=DASKAN_GREEN, opacity=0.7)
        ))

        # 2. Add the ideal 45-degree line (Perfect Prediction)
        fig.add_trace(go.Scatter(
            x=[0, max_val], 
            y=[0, max_val], 
            mode='lines', 
            name='Perfect Prediction', 
            line=dict(color='black', dash='dash')
        ))

        fig.update_layout(
            xaxis_title="Actual Effort (Hours)",
            yaxis_title="Predicted Effort (Hours)",
            title="Model Accuracy on Unseen Test Data",
            hovermode="closest",
            width=800, 
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("Points close to the diagonal line indicate accurate predictions. Points below the line are **underestimates**, and points above are **overestimates**.")
        
        # --- MLOps Simulation: Model Approval & Versioning ---
        st.markdown("---")
        st.subheader("Model Deployment Management (MLOps Simulation)")
        
        current_r2 = st.session_state['train_r2']
        r2_threshold = 0.8 # Enterprise-level threshold
        
        if current_r2 > r2_threshold:
            model_status = "Ready for Production"
            button_label = f" Approve Model for Production Use (R² > {r2_threshold:.3f})"
            is_safe = True
        else:
            model_status = "Requires Further Tuning"
            button_label = f" Review Model Performance (R² < {r2_threshold:.3f})"
            is_safe = False
            
        col_d1, col_d2 = st.columns([2, 1])
        col_d1.metric("Production Quality Threshold", f"R² > {r2_threshold:.3f}", model_status)
        
        if col_d2.button(button_label, type="primary" if is_safe else "secondary", use_container_width=True, key='approve_model_btn'):
            st.session_state['approved_model_version'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state['approved_r2'] = current_r2
            st.success(f"Model version {st.session_state['approved_model_version']} **APPROVED** for Quoting.")

        if 'approved_model_version' in st.session_state and st.session_state['approved_model_version'] is not None:
            st.info(f"**Live Production Model:** Version {st.session_state['approved_model_version']} (Trained on R²: {st.session_state['approved_r2']:.3f})")
        else:
            st.warning("No model is currently approved for production use.")

# ----------------------------------------------------------------------
# TAB 3: MODEL EXPLAINABILITY (XAI)
# ----------------------------------------------------------------------
with tabs[2]:
    st.header(" Model Explainability (SHAP)")

    if st.session_state['models'] is None:
        st.warning("Please train a model in the 'AI Model Engine' tab first.")
    else:
        # 1. Setup Data and Explainer
        X = df_modeling[['surface_area_m2', 'num_levels', 'floor_area_ratio', 'complexity_index', 'is_winter', 'project_type', 'material_type']]
        # Use a fixed split for the display
        X_train, X_test, y_train, y_test = train_test_split(X, df_modeling['total_project_effort'], test_size=0.2, random_state=42)

        # Retrieve cached SHAP data (using fixed function call with underscore)
        explainer, shap_values, X_train_transformed, feature_names = get_shap_data(st.session_state['models'], X_train)
        
        # Get the preprocessor for transforming the test set
        model_pipe_for_prep = (st.session_state['models']['Q50'] if st.session_state['mode'] == 'quantile' else st.session_state['models'])
        preprocessor = model_pipe_for_prep.named_steps['prep']
        X_test_transformed = preprocessor.transform(X_test)


        st.markdown("### 1. Global Feature Impact (Why the Model Works)")
        st.caption("SHAP values provide model-agnostic insights. The magnitude of the SHAP value indicates the feature's influence on the model's output.")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Global Feature Summary (Beeswarm)")
            # SHAP Beeswarm Plot (shows distribution and impact)
            fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(10, 6))
            # Use fixed feature names from cached SHAP data
            shap.summary_plot(shap_values, X_train_transformed, feature_names=feature_names, show=False, plot_type="dot") 
            st.pyplot(fig_beeswarm)
            plt.close(fig_beeswarm) # Prevent memory leak
            st.caption("Each point is a project. Red indicates a high feature value, blue indicates a low feature value. E.g., high complexity (red) tends to drive the predicted effort to the right (higher).")

        with col_g2:
            st.subheader("Aggregated Feature Importance")
            # SHAP Bar Plot (cleaner feature ranking)
            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_train_transformed, feature_names=feature_names, plot_type="bar", show=False)
            st.pyplot(fig_bar)
            plt.close(fig_bar) # Prevent memory leak
            st.caption("The mean absolute SHAP value for each feature, ranking them by their overall importance across the dataset.")

        st.divider()

        st.markdown("### 2. Feature Dependence Analysis")
        st.caption("Analyze how a feature's value non-linearly affects the predicted effort and how features interact.")
        
        # Select features to plot
        all_features = feature_names
        
        col_dep1, col_dep2 = st.columns(2)
        
        with col_dep1:
            dependence_feature = st.selectbox(
                "Select Primary Feature (X-Axis)", 
                all_features,
                key='dependence_feature_select',
                # Default to complexity_index as it's often the most interesting
                index=all_features.index('complexity_index') if 'complexity_index' in all_features else 0
            )
        
        with col_dep2:
            # Create a list including 'Auto-Detect' and all features
            interaction_options = ['Auto-Detect (Strongest Interaction)'] + all_features
            
            interaction_selection = st.selectbox(
                "Select Interaction Feature (Color)", 
                interaction_options,
                key='interaction_feature_select',
                # Default to the 'Auto-Detect' option
                index=0
            )

        # Determine the interaction_index for shap.dependence_plot
        if interaction_selection == 'Auto-Detect (Strongest Interaction)':
            interaction_index_to_use = "auto"
            interaction_feature_name = "the most strongly interacting feature (auto-detected)"
        else:
            interaction_index_to_use = interaction_selection
            interaction_feature_name = interaction_selection
        
        # Plot Dependence Plot
        fig_dep, ax_dep = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            dependence_feature, 
            shap_values, 
            X_train_transformed, 
            feature_names=feature_names, 
            show=False,
            alpha=0.6,
            interaction_index=interaction_index_to_use 
        )
        st.pyplot(fig_dep)
        plt.close(fig_dep) # Prevent memory leak
        
        # Update caption for clarity
        cap_text = f"This plot shows the effect of **{dependence_feature}** on the model's output (predicted effort). The color indicates the value of the selected interaction feature: **{interaction_feature_name}**. This helps visualize how the effect of the primary feature changes based on the value of the interaction feature."

        st.caption(cap_text)
        
        st.divider()
        
# ----------------------------------------------------------------------
# TAB 4: SMART QUOTATION 
# ----------------------------------------------------------------------
with tabs[3]:
    st.header(" Project Quotation & Resource Planner")
    
    # Check for an APPROVED model version for a more sophisticated check
    if st.session_state['approved_model_version'] is None and st.session_state['models'] is None:
        st.warning("Train and/or **Approve** a model in the 'AI Model Engine' tab first.")
    else:
        # --- INPUTS ---
        st.subheader("1. Project Specification")
        c1, c2, c3, c4 = st.columns(4)
        i_area = c1.number_input("Surface Area (m²)", 100, 5000, 800, key='i_area')
        i_levels = c2.number_input("Levels", 1, 20, 2, key='i_levels')
        i_type = c3.selectbox("Type", df_modeling['project_type'].unique(), key='i_type')
        i_mat = c4.selectbox("Material", df_modeling['material_type'].unique(), key='i_mat')
        
        st.markdown("---")
        st.subheader("2. Financial & Risk Settings")
        col_res_1, col_res_2, col_res_3, col_res_4 = st.columns(4)
        avg_hourly_rate = col_res_1.number_input("Avg. Cost Rate ($/h)", 80, 200, AVG_HOURLY_RATE_CAD, key='avg_hourly_rate')
        # NEW INPUT: Profit Margin
        profit_markup = col_res_2.slider("Profit/Markup (%)", 5, 50, 25) / 100
        
        hours_per_week_slider = col_res_3.number_input("Hours per Engineer/Week", 20, 40, HOURS_PER_WEEK, key='hours_per_week')
        i_winter = col_res_4.selectbox("Winter Project Factor", [0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No", key='i_winter_q')
        st.session_state['i_winter'] = i_winter
        
        # Calc Features (Needs to be done here for the prediction button)
        floor_area_ratio = i_area / i_levels
        mat_factor = {'Mixed':2.0, 'Steel':1.5, 'Concrete':1.2}.get(i_mat, 1.0)
        c_idx = (i_levels**1.2) * (floor_area_ratio) / 500 * mat_factor
        
        input_data = pd.DataFrame({
            'surface_area_m2':[i_area], 'num_levels':[i_levels], 'floor_area_ratio':[floor_area_ratio],
            'complexity_index':[max(0.5, c_idx)], 'is_winter':[i_winter], 'project_type':[i_type], 'material_type':[i_mat]
        })
        
        # Prediction Button (This updates the session state)
        if st.button(" RUN AI ESTIMATION", type="primary", use_container_width=True):
            if st.session_state['models'] is None:
                st.error("No model is trained yet. Please go to the 'AI Model Engine' tab to train and approve a model.")
            else:
                with st.spinner("Running AI Prediction..."):
                    # 1. Prediction
                    ss_mode = st.session_state['mode']
                    if ss_mode == 'quantile':
                        q05 = st.session_state['models']['Q05'].predict(input_data)[0]
                        q50 = st.session_state['models']['Q50'].predict(input_data)[0]
                        q95 = st.session_state['models']['Q95'].predict(input_data)[0]
                        pred_val = q50
                        risk_margin = (q95 - q05) / 2
                    else:
                        # Use the non-quantile model (RF, GBR, or Stacking)
                        pred_val = st.session_state['models'].predict(input_data)[0]
                        risk_margin = 0 # Cannot calculate risk margin without quantile model
                    
                    # Store results in session state
                    st.session_state['pred_val'] = pred_val
                    st.session_state['risk_margin'] = risk_margin
                    st.session_state['input_data'] = input_data # Store the data used for prediction
                    st.session_state['c_idx'] = c_idx
                    st.session_state['profit_markup'] = profit_markup
                    st.session_state['quote_generated'] = True
                    st.success("Prediction Generated! Scroll down for plan.")
                
        
        st.markdown("---")

        # --- DYNAMIC OUTPUTS (Reacting to Session State) ---
        if st.session_state.get('quote_generated', False):
            
            pred_val = st.session_state['pred_val']
            risk_margin = st.session_state['risk_margin']
            c_idx = st.session_state['c_idx']
            profit_markup = st.session_state['profit_markup']
            
            # Recalculate financial/resource metrics based on current inputs
            estimated_cost_base = pred_val * avg_hourly_rate
            final_quote = estimated_cost_base * (1 + profit_markup)
            
            # --- NEW: Data Drift Analysis ---
            st.markdown("---")
            st.subheader(" Input Data Drift & Reliability Check")
            
            train_stats = st.session_state['train_stats']
            is_drifting = False
            drift_message = ""
            
            # Check numerical features against training distribution
            for feature in ['surface_area_m2', 'num_levels', 'floor_area_ratio', 'complexity_index']:
                input_val = input_data[feature].iloc[0]
                mean_val = train_stats.loc[feature, 'mean']
                std_val = train_stats.loc[feature, 'std']
                
                # Simple drift detection: > 2 standard deviations away from the mean
                if abs(input_val - mean_val) > 2 * std_val:
                    is_drifting = True
                    drift_message += f"- **{feature}** value ({input_val:.2f}) is significantly outside the historical range (Mean: {mean_val:.2f}, Std Dev: {std_val:.2f}).\n"
            
            if is_drifting:
                st.error(" **MODEL RELIABILITY WARNING:** The input project significantly deviates from the training data distribution in the following areas. Predictions may be **unreliable**.")
                st.markdown(drift_message)
            else:
                st.success(" **Data Quality Check:** Input parameters are well within the historical data distribution. Model predictions are expected to be reliable.")
            st.markdown("---")

            
            # 1. Summary & Financials
            st.markdown("####  Project Summary & Financials")
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            
            col_p1.metric("Estimated Effort (h)", f"{pred_val:.0f} Hours", f"Risk: +/- {risk_margin:.0f}h" if risk_margin > 0 else "Point Estimate")
            col_p2.metric("Base Cost (CAD)", f"${estimated_cost_base:,.0f}")
            col_p3.metric("Profit Markup", f"{int(profit_markup*100)}%", f"${final_quote - estimated_cost_base:,.0f} Profit")
            col_p4.metric("FINAL QUOTE PRICE", f"${final_quote:,.0f}", "Client Price", delta_color="normal")
            
            
            # 2. Resource Planning (Reactive to Slider)
            st.markdown("####  Resource & Timeline Planning")
            
            team_size = st.slider("Select Planned Team Size (Engineers) for Schedule", 1, 10, 2, key='team_size_live')
            
            weeks_needed_1_eng = pred_val / hours_per_week_slider
            real_duration = weeks_needed_1_eng / team_size
            
            c_res_1, c_res_2, c_res_3 = st.columns(3)
            
            c_res_1.metric("Duration (1 Engineer)", f"{weeks_needed_1_eng:.1f} Weeks")
            c_res_2.metric(f"Duration ({team_size} Engineers)", f"{real_duration:.1f} Weeks", delta="Project Duration")
            c_res_3.metric("Project Complexity Index", f"{c_idx:.2f}")

            # Risk Flag
            if real_duration > 15 or (st.session_state['mode'] == 'quantile' and risk_margin > 50) or is_drifting:
                st.error(" **High Risk Alert:** Long duration, high uncertainty, or data drift detected. Increase team size or pad the quote.")
            elif i_winter == 1:
                st.info(" **Seasonal Note:** The 'is_winter' factor was included in the estimation.")
            else:
                st.success(" **Standard Project:** Project fits typical historical parameters.")


            st.markdown("---")
            
            # 3. Task Breakdown & Gantt
            st.markdown("####  Recommended Task Breakdown & Schedule")
            
            type_logs = df_analytics[df_analytics['project_type'] == i_type]
            if not type_logs.empty:
                dist = type_logs['task_category'].value_counts(normalize=True).sort_index()
                
                breakdown_df = pd.DataFrame({'Task': dist.index, 'Ratio': dist.values})
                breakdown_df['Hours'] = breakdown_df['Ratio'] * pred_val
                # Use the live slider value
                breakdown_df['Weeks'] = breakdown_df['Hours'] / (hours_per_week_slider * team_size) 
                
                # --- Breakdown Table ---
                st.dataframe(breakdown_df[['Task', 'Ratio', 'Hours', 'Weeks']].style.format({'Ratio': '{:.1%}', 'Hours': '{:.0f}', 'Weeks': '{:.1f}'}), use_container_width=True, hide_index=True)
                
                # --- Gantt Chart Concept (Advanced Visualization) ---
                st.subheader("Conceptual Project Timeline")
          
                # Prepare data for Plotly Gantt (requires Start, Finish, Task)
                breakdown_df['End_Date'] = (datetime.now() + pd.to_timedelta(breakdown_df['Weeks'].cumsum(), unit='W')).dt.strftime('%Y-%m-%d')
                breakdown_df['Start_Date'] = breakdown_df['End_Date'].shift(1).fillna(datetime.now().strftime('%Y-%m-%d'))
                
                fig_gantt = px.timeline(
                    breakdown_df, 
                    x_start="Start_Date", 
                    x_end="End_Date", 
                    y="Task", 
                    color="Task",
                    title=f"Resource-Loaded Schedule (Team Size: {team_size})",
                    color_discrete_sequence=DASKAN_COLOR_PALETTE
                )
                fig_gantt.update_yaxes(autorange="reversed") 
                st.plotly_chart(fig_gantt, use_container_width=True)

            else:
                st.warning("No historical granular data for this project type to generate a task breakdown.")
