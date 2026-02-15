import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
from datetime import datetime, timedelta

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score, accuracy_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import randint, uniform

# --- 1. CONFIGURATION & CONSTANTS ---
# Daskan Green Accent
DASKAN_GREEN = "#049449" 
DASKAN_COLOR_PALETTE = ['#049449', '#004c29', '#1abc9c', '#3498db', '#9b59b6']
AVG_HOURLY_RATE_CAD = 115
HOURS_PER_WEEK = 30 # For a single engineer

# Model feature configuration (extended metadata)
# Final modeling feature set (metadata-focused)
NUM_FEATURES = [
    'surface_area_m2',
    'building_height_m',
    'num_levels',
    'floor_area_ratio',
    'height_to_area_ratio',
    'area_per_level',
    'expected_duration_days',
    'num_revisions',
    'is_winter',
    'design_hours_total',
    'avg_floor_height'
]
LEAKY_NUM_FEATURES = []
CAT_FEATURES = [
    'material_type'
]

st.set_page_config(
    page_title="Daskan Intelligence | Project Estimator",
    page_icon="DaskanLogo.png",
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

def _infer_dayfirst(series):
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    has_slash = sample.str.contains("/").any()
    has_iso = sample.str.contains(r"^\d{4}-\d{2}-\d{2}").any()
    if has_iso and not has_slash:
        return False
    if has_slash:
        return True
    return False

def _parse_date_column(df, col):
    if col not in df.columns:
        return df
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return df
    dayfirst = _infer_dayfirst(df[col])
    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=dayfirst)
    return df

def _detect_data_profile(df):
    if {'date_logged', 'hours_worked'}.issubset(df.columns):
        return "granular"
    if 'total_project_effort' in df.columns or 'total_project_hours' in df.columns:
        return "metadata"
    return "unknown"

def get_selected_num_features(is_metadata, include_leaky=False):
    features = list(NUM_FEATURES)
    if is_metadata and 'design_hours_total' in features:
        features.remove('design_hours_total')
    if include_leaky:
        features += LEAKY_NUM_FEATURES
    return features

def feature_engineer_data(df_analytics):
    """Aggregates granular data to project level and creates complex features."""
    df_analytics = df_analytics.copy()
    
    # Normalize common column aliases from summary datasets
    if 'is_winter' not in df_analytics.columns and 'is_winter_day' in df_analytics.columns:
        df_analytics = df_analytics.rename(columns={'is_winter_day': 'is_winter'})
    if 'total_project_effort' not in df_analytics.columns and 'total_project_hours' in df_analytics.columns:
        df_analytics = df_analytics.rename(columns={'total_project_hours': 'total_project_effort'})

    # Safety Check: Ensure Date Column is datetime (only if present)
    df_analytics = _parse_date_column(df_analytics, 'date_logged')

    if 'log_id' not in df_analytics.columns:
        # Already summary data (though this case is rare with the generator)
        df_modeling = df_analytics.copy()
        
    else:
        # Aggregate granular logs to get total effort and retain metadata columns when present
        meta_candidates = [
            'project_type',
            'material_type',
            'scope_category',
            'surface_area_m2',
            'num_levels',
            'num_units',
            'building_height_m',
            'floor_area_ratio',
            'planned_start_date',
            'planned_end_date',
            'expected_duration_days',
            'corrected_start_date',
            'corrected_end_date',
            'project_duration_days',
            'actual_duration_days',
            'month_started',
            'quarter',
            'season_flag',
            'holiday_period_flag',
            'total_project_hours',
            'design_hours_total',
            'avg_hours_per_employee',
            'num_revisions',
            'is_winter',
            'is_winter_day'
        ]
        agg_map = {'hours_worked': 'sum'}
        for col in meta_candidates:
            if col in df_analytics.columns:
                agg_map[col] = 'first'
        if 'start_date' in df_analytics.columns:
            agg_map['start_date'] = 'min'
        if 'end_date' in df_analytics.columns:
            agg_map['end_date'] = 'max'

        df_modeling = df_analytics.groupby('project_id').agg(agg_map).reset_index()
        df_modeling = df_modeling.rename(columns={'hours_worked': 'total_project_effort'})

        # Calculate Winter Flag: Logs in Dec, Jan, Feb, Mar (Quebec winter is long)
        winter_projs = df_analytics[df_analytics['date_logged'].dt.month.isin([12,1,2,3])]['project_id'].unique()
        df_modeling['is_winter'] = df_modeling['project_id'].isin(winter_projs).astype(int)
    if 'is_winter' not in df_modeling.columns:
        df_modeling['is_winter'] = 0

    # (No winter concrete interaction flags)

    # Derived Features (applies to both aggregated and summary data)
    if 'surface_area_m2' in df_modeling.columns and 'num_levels' in df_modeling.columns:
        df_modeling['floor_area_ratio'] = np.where(
            df_modeling['num_levels'] > 0,
            df_modeling['surface_area_m2'] / df_modeling['num_levels'],
            np.nan
        )
    elif 'floor_area_ratio' not in df_modeling.columns:
        df_modeling['floor_area_ratio'] = np.nan

    if 'building_height_m' in df_modeling.columns and 'surface_area_m2' in df_modeling.columns:
        df_modeling['height_to_area_ratio'] = np.where(
            df_modeling['surface_area_m2'] > 0,
            df_modeling['building_height_m'] / df_modeling['surface_area_m2'],
            np.nan
        )
    elif 'height_to_area_ratio' not in df_modeling.columns:
        df_modeling['height_to_area_ratio'] = np.nan

    if 'surface_area_m2' in df_modeling.columns and 'num_levels' in df_modeling.columns:
        df_modeling['area_per_level'] = np.where(
            df_modeling['num_levels'] > 0,
            df_modeling['surface_area_m2'] / df_modeling['num_levels'],
            np.nan
        )
    elif 'area_per_level' not in df_modeling.columns:
        df_modeling['area_per_level'] = np.nan

    if 'building_height_m' in df_modeling.columns and 'num_levels' in df_modeling.columns:
        df_modeling['avg_floor_height'] = np.where(
            df_modeling['num_levels'] > 0,
            df_modeling['building_height_m'] / df_modeling['num_levels'],
            np.nan
        )
        try:
            corr = df_modeling[['building_height_m', 'num_levels']].corr().iloc[0, 1]
            if pd.isna(corr) or corr <= 0.9:
                df_modeling['avg_floor_height'] = np.nan
        except Exception:
            pass
    elif 'avg_floor_height' not in df_modeling.columns:
        df_modeling['avg_floor_height'] = np.nan
    
    # Calculate Advanced Complexity Index
    def get_complexity(row):
        mat_factor = {'Mixed':2.0, 'Steel':1.5, 'Concrete':1.2}.get(row['material_type'], 1.0)
        # Formula: Non-linear penalty for height * Area/Level * Material factor
        try:
            levels = row.get('num_levels', np.nan)
            area = row.get('surface_area_m2', np.nan)
            if pd.isna(levels) or pd.isna(area) or levels <= 0:
                return np.nan
            c_idx = (levels**1.2) * (area / levels) / 500 * mat_factor
            return max(0.5, c_idx)
        except Exception:
            return np.nan
    
    df_modeling['complexity_index'] = df_modeling.apply(get_complexity, axis=1)

    # Derived duration features when date columns exist
    date_candidates = [
        ('planned_start_date', 'planned_end_date', 'expected_duration_days'),
        ('corrected_start_date', 'corrected_end_date', 'actual_duration_days'),
        ('start_date', 'end_date', 'project_duration_days')
    ]
    for start_col, end_col, dur_col in date_candidates:
        if start_col in df_modeling.columns and end_col in df_modeling.columns:
            try:
                df_modeling = _parse_date_column(df_modeling, start_col)
                df_modeling = _parse_date_column(df_modeling, end_col)
                if dur_col not in df_modeling.columns:
                    df_modeling[dur_col] = (df_modeling[end_col] - df_modeling[start_col]).dt.days
            except Exception:
                pass

    # If expected_duration_days is missing but planned dates exist, compute it
    if 'expected_duration_days' not in df_modeling.columns and 'planned_start_date' in df_modeling.columns and 'planned_end_date' in df_modeling.columns:
        try:
            df_modeling = _parse_date_column(df_modeling, 'planned_start_date')
            df_modeling = _parse_date_column(df_modeling, 'planned_end_date')
            df_modeling['expected_duration_days'] = (df_modeling['planned_end_date'] - df_modeling['planned_start_date']).dt.days
        except Exception:
            pass

    # If project_duration_days is missing but expected_duration_days exists, use expected as a safe fallback
    if 'project_duration_days' not in df_modeling.columns and 'expected_duration_days' in df_modeling.columns:
        df_modeling['project_duration_days'] = df_modeling['expected_duration_days']

    # Coerce known numeric metadata fields to numeric types
    numeric_candidates = [
        'surface_area_m2',
        'num_levels',
        'building_height_m',
        'floor_area_ratio',
        'height_to_area_ratio',
        'area_per_level',
        'avg_floor_height',
        'num_revisions',
        'is_winter',
        'design_hours_total',
        'expected_duration_days',
        'actual_duration_days',
        'project_duration_days',
        'total_project_effort'
    ]
    for col in numeric_candidates:
        if col in df_modeling.columns:
            df_modeling[col] = pd.to_numeric(df_modeling[col], errors='coerce')
    # If num_revisions is missing in metadata, assume 0 (no revisions logged yet)
    if 'num_revisions' not in df_modeling.columns:
        df_modeling['num_revisions'] = 0

    # Derive a classification target from rule-based complexity logic (no effort leakage)
    if 'project_complexity_class' not in df_modeling.columns:
        # Choose duration column for thresholds (planned/expected preferred)
        duration_col = None
        for col in ['expected_duration_days', 'project_duration_days']:
            if col in df_modeling.columns:
                duration_col = col
                break

        # Quantile thresholds from current dataset
        area_q1 = df_modeling['surface_area_m2'].dropna().quantile(0.25) if 'surface_area_m2' in df_modeling.columns else np.nan
        area_q3 = df_modeling['surface_area_m2'].dropna().quantile(0.75) if 'surface_area_m2' in df_modeling.columns else np.nan
        dur_q1 = df_modeling[duration_col].dropna().quantile(0.25) if duration_col else np.nan
        dur_q3 = df_modeling[duration_col].dropna().quantile(0.75) if duration_col else np.nan

        def _score_scale(row):
            score = 0
            area = row.get('surface_area_m2', np.nan)
            levels = row.get('num_levels', np.nan)
            height = row.get('building_height_m', np.nan)
            if pd.notna(area) and pd.notna(area_q1) and pd.notna(area_q3):
                if area >= area_q3:
                    score = max(score, 2)
                elif area >= area_q1:
                    score = max(score, 1)
            if pd.notna(levels):
                if levels >= 7:
                    score = max(score, 2)
                elif levels >= 4:
                    score = max(score, 1)
            if pd.notna(height):
                if height >= 30:
                    score = max(score, 2)
                elif height >= 15:
                    score = max(score, 1)
            return score

        def _score_duration(row):
            if duration_col is None:
                return 0
            val = row.get(duration_col, np.nan)
            if pd.isna(val) or pd.isna(dur_q1) or pd.isna(dur_q3):
                return 0
            if val >= dur_q3:
                return 2
            if val >= dur_q1:
                return 1
            return 0

        def _score_revisions(row):
            val = row.get('num_revisions', np.nan)
            if pd.isna(val):
                return 0
            if val >= 4:
                return 2
            if val >= 2:
                return 1
            return 0

        def _context_bump(row):
            scope = str(row.get('scope_category', '')).lower()
            material = str(row.get('material_type', '')).lower()
            bump = 0
            if any(k in scope for k in ['institutional', 'infrastructure']):
                bump = 1
            if ('mixed' in material) or ('concrete' in material):
                bump = max(bump, 1)
            return bump

        def _classify(row):
            base = _score_scale(row) + _score_duration(row) + _score_revisions(row)
            if base >= 3:
                base += _context_bump(row)
            if base >= 6:
                return 'High'
            if base >= 3:
                return 'Medium'
            return 'Low'

        df_modeling['project_complexity_class'] = df_modeling.apply(_classify, axis=1)

    # Derive month_started and quarter if missing and any date exists
    if 'month_started' not in df_modeling.columns or 'quarter' not in df_modeling.columns:
        date_source = None
        for col in ['planned_start_date', 'corrected_start_date', 'start_date']:
            if col in df_modeling.columns:
                date_source = col
                break
        if date_source is not None:
            df_modeling = _parse_date_column(df_modeling, date_source)
            if 'month_started' not in df_modeling.columns:
                df_modeling['month_started'] = df_modeling[date_source].dt.month
            if 'quarter' not in df_modeling.columns:
                df_modeling['quarter'] = df_modeling[date_source].dt.quarter

    # Derive ordinal date features for training
    date_cols = [
        ('planned_start_date', 'planned_start_ordinal'),
        ('planned_end_date', 'planned_end_ordinal'),
        ('corrected_start_date', 'corrected_start_ordinal'),
        ('corrected_end_date', 'corrected_end_ordinal')
    ]
    for src, dst in date_cols:
        if src in df_modeling.columns:
            df_modeling = _parse_date_column(df_modeling, src)
            df_modeling[dst] = df_modeling[src].map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

    # Ensure all configured model feature columns exist
    for col in NUM_FEATURES + LEAKY_NUM_FEATURES + CAT_FEATURES:
        if col not in df_modeling.columns:
            df_modeling[col] = np.nan

    return df_modeling

def build_model(task, model_choice, num_features, cat_features, use_log_target=True):
    """Builds regression or classification pipeline with domain-ready preprocessing."""
    robust_cols = [c for c in num_features if c in ['num_revisions', 'design_hours_total']]
    standard_cols = [c for c in num_features if c not in robust_cols]
    transformers = []
    if standard_cols:
        transformers.append(('num_std', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), standard_cols))
    if robust_cols:
        transformers.append(('num_robust', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), robust_cols))
    if cat_features:
        transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_features))
    preprocessor = ColumnTransformer(transformers=transformers)

    if task == "regression":
        if model_choice == "Linear Regression":
            base_model = LinearRegression()
        elif model_choice == "Random Forest Regressor":
            base_model = RandomForestRegressor(
                n_estimators=400,
                max_depth=16,
                min_samples_split=3,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            )
        else:
            base_model = GradientBoostingRegressor(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                max_features='sqrt',
                random_state=42
            )
        pipe = Pipeline([('prep', preprocessor), ('model', base_model)])
        if use_log_target:
            return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
        return pipe

    if task == "classification":
        if model_choice == "Logistic Regression":
            base_model = LogisticRegression(max_iter=300, class_weight="balanced")
        else:
            base_model = GradientBoostingClassifier(random_state=42)
        return Pipeline([('prep', preprocessor), ('model', base_model)])

    raise ValueError(f"Unknown task: {task}")

@st.cache_resource(show_spinner=False)
def train_and_tune_model(X_train, y_train, model_choice, num_features, cat_features, use_log_target, sample_weight=None):
    """Trains a tuned model using Linear Regression or Gradient Boosting Regressor."""

    def _sample_weight_params():
        if sample_weight is None:
            return None
        # Always route to the pipeline's final estimator
        return {"model__sample_weight": sample_weight}

    st.session_state['mode'] = 'point'
    reg = build_model("regression", model_choice, num_features, cat_features, use_log_target=use_log_target)
    sw_params = _sample_weight_params()
    if sw_params:
        reg.fit(X_train, y_train, **sw_params)
    else:
        reg.fit(X_train, y_train)
    r2 = r2_score(y_train, reg.predict(X_train))
    return reg, r2, st.session_state['mode']


@st.cache_data(show_spinner="Calculating SHAP values...")
def get_shap_data(_models, X_train):
    """Caches the expensive SHAP calculation."""
    
    # Get the trained model (Linear Regression or Gradient Boosting Regressor)
    model_pipe = _models
    if isinstance(model_pipe, TransformedTargetRegressor):
        model_pipe = model_pipe.regressor_

    # Isolate the preprocessor and the final estimator
    preprocessor = model_pipe.named_steps['prep']
    final_estimator = model_pipe.named_steps['model']
    
    # Get preprocessed training data
    X_train_transformed = preprocessor.transform(X_train)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    
    # Get feature names after one-hot encoding
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        num_features = list(preprocessor.transformers_[0][2])
        cat_inputs = list(preprocessor.transformers_[1][2])
        cat_feature_names = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(
            input_features=cat_inputs
        )
        feature_names = num_features + list(cat_feature_names)

    # Create an independent background data for SHAP
    if isinstance(final_estimator, LinearRegression):
        explainer = shap.LinearExplainer(final_estimator, X_train_transformed)
    else:
        explainer = shap.TreeExplainer(final_estimator)
    shap_values = explainer.shap_values(X_train_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    return explainer, shap_values, X_train_transformed, feature_names

@st.cache_data(show_spinner="Calculating SHAP interaction values...")
def get_shap_interaction_data(_models, X_train):
    """Caches SHAP interaction values for tree-based models."""
    model_pipe = _models
    if isinstance(model_pipe, TransformedTargetRegressor):
        model_pipe = model_pipe.regressor_

    preprocessor = model_pipe.named_steps['prep']
    final_estimator = model_pipe.named_steps['model']

    X_train_transformed = preprocessor.transform(X_train)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        num_features = list(preprocessor.transformers_[0][2])
        cat_inputs = list(preprocessor.transformers_[1][2])
        cat_feature_names = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(
            input_features=cat_inputs
        )
        feature_names = num_features + list(cat_feature_names)

    if isinstance(final_estimator, LinearRegression):
        return None, feature_names

    explainer = shap.TreeExplainer(final_estimator)
    interaction_values = explainer.shap_interaction_values(X_train_transformed)
    if isinstance(interaction_values, list):
        interaction_values = interaction_values[0]

    return interaction_values, feature_names

def get_ale_data(model_pipe, X_train_raw, feature_name):
    """Computes a simple 1D ALE for a numeric feature without external deps."""
    if feature_name not in X_train_raw.columns:
        return None

    x = X_train_raw[feature_name].dropna()
    if x.empty:
        return None

    # Use quantile bins for stability
    num_bins = 20
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.unique(np.quantile(x, quantiles))
    if len(bin_edges) < 3:
        return None

    # Assign each row to a bin
    bin_ids = np.digitize(X_train_raw[feature_name], bin_edges[1:-1], right=True)

    effects = []
    centers = []

    for b in range(len(bin_edges) - 1):
        in_bin = bin_ids == b
        if not np.any(in_bin):
            continue

        X_low = X_train_raw.loc[in_bin].copy()
        X_high = X_train_raw.loc[in_bin].copy()
        X_low[feature_name] = bin_edges[b]
        X_high[feature_name] = bin_edges[b + 1]

        pred_low = model_pipe.predict(X_low)
        pred_high = model_pipe.predict(X_high)
        local_eff = np.mean(pred_high - pred_low)

        effects.append(local_eff)
        centers.append((bin_edges[b] + bin_edges[b + 1]) / 2.0)

    if not effects:
        return None

    ale_vals = np.cumsum(effects)
    ale_vals = ale_vals - np.mean(ale_vals)

    return pd.DataFrame({feature_name: centers, "eff": ale_vals})


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
if 'target_col' in st.session_state:
    del st.session_state['target_col']


# Sidebar for Data Control
with st.sidebar:
    st.image("cropped-DaskanLogo.png", width=500) 
    st.markdown(f"### Daskan Intelligence")
    st.caption("v1.0 - Powered by Machine Learning")
    st.divider()
    
    st.subheader("Data Source")
    uploaded_file = st.file_uploader("Upload CSV data file (Granular or Metadata)", type=['csv'])
    
    st.divider()
    include_leaky_features = True
    st.divider()
    st.subheader(" Data Generation")
    random_seed = st.slider("Seed", 1, 100, 42)
    sample_size = st.slider("Projects to Generate", 50, 500, 200)

# Data Loading & Processing
profile = "synthetic"
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        # Normalize column names (trim whitespace)
        raw_df.columns = [c.strip() for c in raw_df.columns]
        # Normalize common aliases early for schema clarity
        alias_map = {
            'total_project_hours': 'total_project_effort',
            'is_winter_day': 'is_winter'
        }
        raw_df = raw_df.rename(columns={k: v for k, v in alias_map.items() if k in raw_df.columns})
        # Normalize common date columns across granular/full-exploration/metadata files
        for col in [
            'date_logged',
            'planned_start_date', 'planned_end_date',
            'corrected_start_date', 'corrected_end_date',
            'start_date', 'end_date'
        ]:
            raw_df = _parse_date_column(raw_df, col)
        df_analytics = raw_df.copy()
        profile = _detect_data_profile(raw_df)
        if profile == "granular":
            if 'scope_category' in raw_df.columns:
                data_source = "Uploaded Full Exploration Data (Granular + Metadata)"
            else:
                data_source = "Uploaded Granular Data"
        elif profile == "metadata":
            data_source = "Uploaded Project Metadata"
        else:
            data_source = "Uploaded Data (Unknown Schema)"
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        df_analytics = generate_granular_synthetic_data(sample_size, random_seed)
        data_source = "Synthetic Data (Fallback)"
else:
    df_analytics = generate_granular_synthetic_data(sample_size, random_seed)
    data_source = f"Synthetic Data ({sample_size} Projects)"

# Feature Engineering
df_modeling = feature_engineer_data(df_analytics)

# Dataset profile flags
is_granular = profile == "granular"
is_metadata = profile == "metadata"

# Calculate Training Data Statistics for Drift Monitoring
num_cols = NUM_FEATURES + LEAKY_NUM_FEATURES
# Ensure we store the stats after feature engineering
st.session_state['train_stats'] = df_modeling[num_cols].agg(['mean', 'std']).T

# Complexity threshold diagnostics (for auditability)
def _get_duration_col(df):
    for col in ['expected_duration_days', 'project_duration_days']:
        if col in df.columns:
            return col
    return None

_dur_col = _get_duration_col(df_modeling)
_area_q1 = df_modeling['surface_area_m2'].dropna().quantile(0.25) if 'surface_area_m2' in df_modeling.columns else np.nan
_area_q3 = df_modeling['surface_area_m2'].dropna().quantile(0.75) if 'surface_area_m2' in df_modeling.columns else np.nan
_dur_q1 = df_modeling[_dur_col].dropna().quantile(0.25) if _dur_col else np.nan
_dur_q3 = df_modeling[_dur_col].dropna().quantile(0.75) if _dur_col else np.nan

def run_data_quality_checks(df_raw, df_model, selected_features):
    """Basic data quality and metadata checks for reporting."""
    required_cols = ['project_id'] + selected_features + [
        'total_project_effort',
        'project_complexity_class'
    ]

    missing = {}
    for col in required_cols:
        if col not in df_model.columns:
            missing[col] = 1.0
        else:
            missing[col] = float(df_model[col].isna().mean())

    missing_df = pd.DataFrame(
        [{"Column": k, "Missing Rate": v} for k, v in missing.items()]
    ).sort_values("Missing Rate", ascending=False)

    issues = []

    # Range checks
    range_checks = {
        'surface_area_m2': (0, None),
        'num_levels': (1, None),
        'num_units': (0, None),
        'building_height_m': (0, None),
        'project_duration_days': (0, None),
        'num_revisions': (0, None),
        'total_project_effort': (0, None)
    }

    for col, (min_v, max_v) in range_checks.items():
        if col in df_model.columns:
            series = df_model[col]
            if min_v is not None:
                bad = series < min_v
                if bad.any():
                    issues.append(f"{col}: {bad.sum()} values below {min_v}")
            if max_v is not None:
                bad = series > max_v
                if bad.any():
                    issues.append(f"{col}: {bad.sum()} values above {max_v}")

    # Date order checks
    date_pairs = [
        ('planned_start_date', 'planned_end_date'),
        ('corrected_start_date', 'corrected_end_date'),
        ('start_date', 'end_date')
    ]
    for start_col, end_col in date_pairs:
        if start_col in df_model.columns and end_col in df_model.columns:
            start = pd.to_datetime(df_model[start_col], errors='coerce')
            end = pd.to_datetime(df_model[end_col], errors='coerce')
            bad = (start.notna() & end.notna() & (end < start))
            if bad.any():
                issues.append(f"{start_col} > {end_col}: {bad.sum()} rows")

    # Duplicates
    if 'project_id' in df_model.columns:
        dup = df_model['project_id'].duplicated().sum()
        if dup > 0:
            issues.append(f"project_id duplicates: {dup}")

    # Training metadata checks
    meta_issues = []
    if 'total_project_effort' not in df_model.columns:
        meta_issues.append("Target missing: total_project_effort")
    else:
        if df_model['total_project_effort'].isna().mean() > 0.2:
            meta_issues.append("Target missing rate > 20%")
    if 'project_complexity_class' not in df_model.columns:
        meta_issues.append("Target missing: project_complexity_class")

    feature_missing = missing_df[missing_df["Missing Rate"] > 0.2]
    if not feature_missing.empty:
        meta_issues.append("Model features with missing rate > 20%")

    return missing_df, issues, meta_issues


# --- MAIN APP TITLE ---
st.title(" Structural Project Intelligence Dashboard")
st.markdown(f"**Current Data Source:** `{data_source}` | **Total Projects:** `{len(df_modeling)}`")
st.divider()

tabs = st.tabs([
    " Deep Dive Analytics",
    " AI Model Engine",
    " Model Explainability (XAI)",
    " Classification",
    " Smart Quotation",
    " Clustering & Personas"
])

# ----------------------------------------------------------------------
# TAB 1: ANALYTICS
# ----------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Historical Project Insights")
    st.caption(f"Dataset profile: {profile}")
    
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
    st.markdown("### Data Quality & Metadata Checks")
    selected_features_for_checks = NUM_FEATURES + (LEAKY_NUM_FEATURES if include_leaky_features else []) + CAT_FEATURES
    missing_df, issues, meta_issues = run_data_quality_checks(df_analytics, df_modeling, selected_features_for_checks)

    with st.expander("Overall Data Quality Check", expanded=False):
        st.caption("Shows missing values and basic sanity checks across key fields.")
        st.dataframe(
            missing_df.style.format({"Missing Rate": "{:.1%}"}),
            use_container_width=True,
            hide_index=True
        )
        if issues:
            st.warning("Potential data issues detected:")
            st.write("\n".join([f"- {i}" for i in issues]))
        else:
            st.success("No major data quality issues detected.")

    with st.expander("Model Training Metadata Check", expanded=False):
        st.caption("Checks that model features and target are usable for training.")
        if meta_issues:
            st.warning("Training metadata issues detected:")
            st.write("\n".join([f"- {i}" for i in meta_issues]))
        else:
            st.success("Training metadata looks good.")

    with st.expander("Metadata Schema Health", expanded=False):
        st.caption("Quick summary of core training schema availability for the uploaded data.")
        core_cols = selected_features_for_checks + ['total_project_effort']
        present = [c for c in core_cols if c in df_modeling.columns]
        missing = [c for c in core_cols if c not in df_modeling.columns]
        missing_rates = {}
        for c in present:
            missing_rates[c] = float(df_modeling[c].isna().mean())

        st.write(f"Rows: {len(df_modeling)}")
        st.write(f"Core columns present: {len(present)} / {len(core_cols)}")
        if missing:
            st.warning("Missing core columns:")
            st.write(", ".join(missing))
        high_missing = [c for c, r in missing_rates.items() if r > 0.2]
        if high_missing:
            st.warning("Core columns with >20% missing values:")
            st.write(", ".join(high_missing))
        else:
            st.success("Core schema looks healthy.")

        extra_cols = [c for c in df_modeling.columns if c not in core_cols]
        if extra_cols:
            st.caption("Extra columns found (not used in training):")
            st.write(", ".join(sorted(extra_cols)))

    with st.expander("Complexity Thresholds (Audit)", expanded=False):
        st.caption("Computed from current dataset using quantiles and fixed engineering cutoffs.")
        rows = [
            {"Dimension": "Surface area Q1/Q3 (m2)", "Thresholds": f"{_area_q1:.1f} / {_area_q3:.1f}" if pd.notna(_area_q1) and pd.notna(_area_q3) else "N/A"},
            {"Dimension": f"Duration Q1/Q3 ({_dur_col})", "Thresholds": f"{_dur_q1:.1f} / {_dur_q3:.1f}" if _dur_col and pd.notna(_dur_q1) and pd.notna(_dur_q3) else "N/A"},
            {"Dimension": "Duration source used", "Thresholds": _dur_col or "N/A"},
            {"Dimension": "Levels cutoffs", "Thresholds": ">= 4 (Medium), >= 7 (High)"},
            {"Dimension": "Height cutoffs", "Thresholds": ">= 15m (Medium), >= 30m (High)"},
            {"Dimension": "Revisions cutoffs", "Thresholds": "2-3 (Medium), >= 4 (High)"},
            {"Dimension": "Context bump", "Thresholds": "Institutional/Infrastructure or Mixed/Concrete if base >= 3"}
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Project Complexity By Project", expanded=False):
        st.caption("Computed complexity class per project.")
        if df_modeling['project_id'].isna().any():
            st.warning("Some rows have missing project_id; those rows will appear at the end.")
        display_cols = [
            'project_id',
            'project_complexity_class',
            'project_type',
            'scope_category',
            'material_type',
            'surface_area_m2',
            'num_levels',
            'building_height_m',
            'expected_duration_days',
            'project_duration_days',
            'num_revisions'
        ]
        display_cols = [c for c in display_cols if c in df_modeling.columns]
        st.dataframe(
            df_modeling[display_cols].sort_values('project_id'),
            use_container_width=True,
            hide_index=True
        )
    
    # Feature Correlation Heatmap
    st.markdown("###  Feature Correlation Analysis")
    st.caption("Understand linear relationships between numerical features and project effort.")
    
    numerical_features = NUM_FEATURES + (LEAKY_NUM_FEATURES if include_leaky_features else []) + ['total_project_effort']
    corr_df = df_modeling[numerical_features].copy()
    # Coerce to numeric to avoid string/object columns producing NaNs
    for col in corr_df.columns:
        corr_df[col] = pd.to_numeric(corr_df[col], errors='coerce')
    # Drop columns with all-NaN or zero variance (corr undefined)
    valid_cols = []
    excluded_cols = []
    for col in corr_df.columns:
        series = corr_df[col]
        if series.notna().sum() == 0:
            excluded_cols.append(col)
            continue
        if series.nunique(dropna=True) <= 1:
            excluded_cols.append(col)
            continue
        valid_cols.append(col)
    corr_df = corr_df[valid_cols]
    
    # Calculate correlation matrix
    corr_matrix = corr_df.corr()
    
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

    if excluded_cols:
        st.caption(
            "Excluded from correlation (all-missing or constant): "
            + ", ".join(excluded_cols)
        )
    
    # --- CAPTION ADDED HERE ---
    st.caption(
        "**Understanding the Heatmap:** This chart displays the **linear correlation** between all numerical features. "
        "Values range from **-1.0 to +1.0**."
        "\n\n"
        "**Negative Correlation (Blue, closer to -1.0):** As one feature **increases**, the other tends to **decrease** (e.g., shorter `project_duration_days` can be associated with higher `num_revisions`)."
        "\n\n"
        "**Positive Correlation (Red, closer to +1.0):** As one feature **increases**, the other tends to **increase** (e.g., higher `num_levels` often drives higher `total_project_effort`). The goal is to identify features that strongly correlate with the target variable (`total_project_effort`)."
    )
    # --- END CAPTION ADDED ---

    st.divider()
    
    # Granular Deep Dive (S-Curve & Task)
    st.markdown("###  Project Timeline Analysis (S-Curve)")
    col_g1, col_g2 = st.columns([2, 1])
    
    has_granular_cols = all(col in df_analytics.columns for col in ['date_logged', 'hours_worked', 'task_category'])
    if not has_granular_cols:
        st.info("Timeline and task breakdown require granular timesheet data (date_logged, hours_worked, task_category). Upload granular data to enable this section.")
    else:
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
    
    # Training readiness checks
    train_issues = []
    use_leaky = include_leaky_features
    selected_num_features = get_selected_num_features(is_metadata, include_leaky=use_leaky)
    selected_features = selected_num_features + CAT_FEATURES
    missing_features = [c for c in selected_features if c not in df_modeling.columns]
    if missing_features:
        train_issues.append(f"Missing training features: {', '.join(missing_features)}")

    available_features = [c for c in selected_features if c in df_modeling.columns]
    # Fixed target (regression): total_project_effort (hours)
    target_col = "total_project_effort"
    st.session_state['target_col'] = target_col

    if target_col not in df_modeling.columns:
        train_issues.append(f"Missing target: {target_col}")
        X = df_modeling[available_features]
        y = pd.Series(dtype=float)
    else:
        X = df_modeling[available_features]
        y = df_modeling[target_col]
        if y.dropna().shape[0] < 10:
            train_issues.append("Not enough non-null target values (need at least 10).")
        if X.shape[0] < 10:
            train_issues.append("Not enough rows to train a model (need at least 10).")

    train_blocked = len(train_issues) > 0
    if train_blocked:
        st.warning("Training is disabled until data issues are resolved:")
        st.write("\n".join([f"- {i}" for i in train_issues]))
        X_train = X_test = y_train = y_test = None
    else:
        split_mode = st.session_state.get("split_mode", "Random Split")
        sample_weight = None
        if target_col in df_modeling.columns:
            hi_thresh = df_modeling[target_col].quantile(0.75)
            sample_weight = np.where(df_modeling[target_col] > hi_thresh, 1.5, 1.0)
        if split_mode.startswith("Time"):
            if 'planned_start_date' in df_modeling.columns:
                df_modeling = _parse_date_column(df_modeling, 'planned_start_date')
                order = df_modeling['planned_start_date'].sort_values().index
            elif 'corrected_start_date' in df_modeling.columns:
                df_modeling = _parse_date_column(df_modeling, 'corrected_start_date')
                order = df_modeling['corrected_start_date'].sort_values().index
            else:
                order = df_modeling.index
            df_ord = df_modeling.loc[order]
            X = df_ord[available_features]
            y = df_ord[target_col]
            split_idx = int(len(df_ord) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            if sample_weight is not None:
                w_ord = sample_weight[order]
                w_train, w_test = w_ord[:split_idx], w_ord[split_idx:]
        else:
            if sample_weight is not None:
                X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
                    X, y, sample_weight, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with col_m1:
        st.subheader("Configuration")
        model_choice = st.selectbox(
            "Algorithm (Regression)",
            [
                "Linear Regression",
                "Random Forest Regressor",
                "Gradient Boosting Regressor"
            ],
            key='model_select'
        )
        use_log_target = st.checkbox(
            "Use log-target (skew handling via TransformedTargetRegressor)",
            value=True,
            key='use_log_target',
            disabled=True
        )
        
    with col_m2:
        st.subheader("Hyperparameters")
        split_mode = st.selectbox(
            "Validation Split",
            ["Random Split", "Time Split (if dates available)"],
            key="split_mode"
        )
        cv_folds = st.slider("CV Folds (for model summary)", 3, 8, 5, key="cv_folds")
        
        if model_choice == "Linear Regression":
            st.info("Using Linear Regression for transparent linear coefficients.")
        elif model_choice == "Random Forest Regressor":
            st.info("Robust non-linear baseline; good for mixed feature effects.")
        elif model_choice == "Gradient Boosting Regressor":
            st.info("Tuning: n_estimators (100-300), max_depth (3-7), learning_rate.")
            
    with col_m3:
        st.subheader("Action")
        st.markdown("---")
        if st.button("Train & Tune Model", type="primary", use_container_width=True, disabled=train_blocked):
            st.session_state['quote_generated'] = False # Reset quote when training new model
            with st.spinner("Executing Training & Hyperparameter Search..."):
                try:
                    tuned_model, performance_metric, mode = train_and_tune_model(
                        X_train, y_train, model_choice,
                        selected_num_features, CAT_FEATURES, use_log_target,
                        sample_weight=w_train if 'w_train' in locals() else None
                    )
                    st.session_state['models'] = tuned_model
                    st.session_state['mode'] = mode
                    
                    preds = tuned_model.predict(X_test)
                    mae_test = mean_absolute_error(y_test, preds)
                    r2_test = r2_score(y_test, preds)
                    st.session_state['train_r2'] = r2_test # Store R2 for display
                    st.metric("Test MAE (Tuned Model)", f"{mae_test:.1f} Hours")
                    st.metric("Test R2 (Tuned Model)", f"{r2_test:.3f}")
                    st.caption(f"Approx. prediction interval: +/-{mae_test * 1.5:.1f} hours (1.5 x MAE)")
                    
                    # Store predictions for visualization and MLOps simulation
                    st.session_state['y_test'] = y_test
                    st.session_state['y_preds'] = preds
                    
                    # CV summary for selected model
                    try:
                        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(
                            tuned_model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1
                        )
                        st.caption(f"CV MAE: {(-cv_scores.mean()):.2f} +/- {cv_scores.std():.2f} hours")
                    except Exception:
                        pass

                    st.success("Model training complete.")
                except Exception as e:
                    st.error(f"Training Error: {e}")

    st.markdown("---")
    st.info(f"Current Model: **{model_choice}** | Mode: **{st.session_state['mode'].upper()}** | Test R2: **{st.session_state['train_r2']:.3f}**")
    st.caption("Scaling: RobustScaler for num_revisions/design_hours_total; StandardScaler for other numeric features.")
    
    st.markdown("---")
    st.subheader("Baseline Model Comparison")
    st.caption("Quick, consistent comparison of common regressors on the same train/test split.")
    
    if st.button("Run Baseline Comparison", key="run_model_comparison", disabled=train_blocked):
        with st.spinner("Training baseline models for comparison..."):
            robust_cols = [c for c in selected_num_features if c in ['num_revisions', 'design_hours_total']]
            standard_cols = [c for c in selected_num_features if c not in robust_cols]
            transformers = []
            if standard_cols:
                transformers.append(('num_std', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), standard_cols))
            if robust_cols:
                transformers.append(('num_robust', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), robust_cols))
            transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CAT_FEATURES))
            preprocessor = ColumnTransformer(transformers=transformers)
            
            model_map = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(
                    n_estimators=400,
                    max_depth=16,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42
                ),
                "Gradient Boosting Regressor": GradientBoostingRegressor(
                    n_estimators=400,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    max_features='sqrt',
                    random_state=42
                )
            }
            
            results = []
            comparison_models = {}
            for name, model in model_map.items():
                pipe = Pipeline([('prep', preprocessor), ('model', model)])
                reg = pipe
                if use_log_target:
                    reg = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
                reg.fit(X_train, y_train)
                preds = reg.predict(X_test)
                
                results.append({
                    "Model": name,
                    "R2": r2_score(y_test, preds),
                    "MAE": mean_absolute_error(y_test, preds),
                    "RMSE": np.sqrt(mean_squared_error(y_test, preds))
                })
                comparison_models[name] = reg
            
            st.session_state['model_comparison'] = pd.DataFrame(results).sort_values("MAE", ascending=True)
            st.session_state['comparison_models'] = comparison_models
    
    if 'model_comparison' in st.session_state:
        st.dataframe(
            st.session_state['model_comparison'].style.format({"R2": "{:.3f}", "MAE": "{:.1f}", "RMSE": "{:.1f}"}),
            use_container_width=True,
            hide_index=True
        )
        with st.expander("Baseline Linear Feature Importance (Permutation + SHAP)", expanded=False):
            base_model = st.session_state['comparison_models'].get("Linear Regression")
            if base_model is None:
                st.info("Run baseline comparison to compute baseline importance.")
            else:
                try:
                    if isinstance(base_model, TransformedTargetRegressor):
                        base_model = base_model.regressor_
                    perm = permutation_importance(
                        base_model,
                        X_test,
                        y_test,
                        n_repeats=10,
                        random_state=42,
                        scoring='neg_mean_absolute_error'
                    )
                    imp_df = pd.DataFrame({
                        "Feature": X_test.columns,
                        "Importance": perm.importances_mean
                    }).sort_values("Importance", ascending=False)
                    fig_imp = px.bar(
                        imp_df.head(10),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Baseline Linear: Top 10 Features (Permutation Importance)"
                    )
                    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_imp, use_container_width=True)

                    # SHAP summary for baseline (linear)
                    X_tr = base_model.named_steps['prep'].transform(X_train)
                    if hasattr(X_tr, "toarray"):
                        X_tr = X_tr.toarray()
                    explainer = shap.LinearExplainer(base_model.named_steps['model'], X_tr)
                    shap_vals = explainer.shap_values(X_tr[:200])
                    fig_sum, ax_sum = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_vals, X_tr[:200], show=False)
                    st.pyplot(fig_sum)
                    plt.close(fig_sum)
                except Exception as e:
                    st.warning(f"Baseline importance unavailable: {e}")
    
    st.markdown("---")
    st.subheader("Permutation Feature Importance (Current Model)")
    if st.session_state['models'] is None:
        st.info("Train a model to view feature importance.")
    else:
        model_pipe_for_imp = st.session_state['models']
        
        perm = permutation_importance(
            model_pipe_for_imp,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring='neg_mean_absolute_error'
        )
        imp_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": perm.importances_mean
        }).sort_values("Importance", ascending=False)
        
        fig_imp = px.bar(
            imp_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Features (Permutation Importance)"
        )
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)
    
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
        r2_threshold = 0.7 # Production quality threshold
        
        if current_r2 > r2_threshold:
            model_status = "Ready for Production"
            button_label = f" Approve Model for Production Use (R2 > {r2_threshold:.3f})"
            is_safe = True
        else:
            model_status = "Requires Further Tuning"
            button_label = f" Review Model Performance (R2 < {r2_threshold:.3f})"
            is_safe = False
            
        col_d1, col_d2 = st.columns([2, 1])
        col_d1.metric("Production Quality Threshold", f"R2 > {r2_threshold:.3f}", model_status)
        
        if col_d2.button(button_label, type="primary" if is_safe else "secondary", use_container_width=True, key='approve_model_btn'):
            st.session_state['approved_model_version'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state['approved_r2'] = current_r2
            st.success(f"Model version {st.session_state['approved_model_version']} **APPROVED** for Quoting.")

        if 'approved_model_version' in st.session_state and st.session_state['approved_model_version'] is not None:
            st.info(f"**Live Production Model:** Version {st.session_state['approved_model_version']} (Trained on R2: {st.session_state['approved_r2']:.3f})")
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
        selected_num_features = get_selected_num_features(is_metadata, include_leaky=include_leaky_features)
        selected_features = selected_num_features + CAT_FEATURES
        missing_features = [c for c in selected_features if c not in df_modeling.columns]
        if missing_features:
            st.warning("Explainability unavailable due to missing training features:")
            st.write(", ".join(missing_features))
            st.stop()
        X = df_modeling[selected_features]
        # Use a fixed split for the display
        target_col = st.session_state.get('target_col', 'total_project_effort')
        X_train, X_test, y_train, y_test = train_test_split(X, df_modeling[target_col], test_size=0.2, random_state=42)

        # Retrieve cached SHAP data (using fixed function call with underscore)
        explainer, shap_values, X_train_transformed, feature_names = get_shap_data(st.session_state['models'], X_train)
        
        # Get the preprocessor for transforming the test set
        model_pipe_for_prep = st.session_state['models']
        if isinstance(model_pipe_for_prep, TransformedTargetRegressor):
            model_pipe_for_prep = model_pipe_for_prep.regressor_
            
        preprocessor = model_pipe_for_prep.named_steps['prep']
        X_test_transformed = preprocessor.transform(X_test)
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()


        st.markdown("### 1. Global Feature Impact (Why the Model Works)")
        st.caption("SHAP values indicate how each feature pushes predictions up or down.")
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.subheader("Global Feature Summary (Beeswarm)")
            # Build a Plotly-based beeswarm for consistent styling
            try:
                sv = np.array(shap_values)
                if sv.ndim == 1:
                    sv = sv.reshape(-1, 1)
                if sv.shape[1] != len(feature_names):
                    st.warning("SHAP feature names do not match SHAP values. Beeswarm is unavailable.")
                else:
                    # Use top features to keep plot readable
                    mean_abs = np.mean(np.abs(sv), axis=0)
                    top_idx = np.argsort(mean_abs)[-15:][::-1]
                    top_features = [feature_names[i] for i in top_idx]

                    # Sample rows for performance
                    max_points = 2000
                    n_rows = sv.shape[0]
                    if n_rows > max_points:
                        sample_idx = np.random.choice(n_rows, size=max_points, replace=False)
                    else:
                        sample_idx = np.arange(n_rows)

                    rows = []
                    for i in top_idx:
                        vals = X_train_transformed[sample_idx, i]
                        shap_vals = sv[sample_idx, i]
                        for v, s in zip(vals, shap_vals):
                            rows.append({"feature": feature_names[i], "shap": s, "value": v})
                    bees_df = pd.DataFrame(rows)

                    fig_bees = px.scatter(
                        bees_df,
                        x="shap",
                        y="feature",
                        color="value",
                        color_continuous_scale="RdBu",
                        title="SHAP Beeswarm (Top Features)"
                    )
                    fig_bees.update_traces(marker=dict(opacity=0.5, size=6))
                    fig_bees.update_layout(xaxis_title="SHAP Value", yaxis_title="")
                    st.plotly_chart(fig_bees, use_container_width=True)
                    st.caption("Points show SHAP impact per project. Color indicates feature value.")
            except Exception as e:
                st.warning(f"Beeswarm plot unavailable: {e}")

        with col_g2:
            st.subheader("Aggregated Feature Importance")
            try:
                sv = np.array(shap_values)
                if sv.ndim == 1:
                    sv = sv.reshape(-1, 1)
                if sv.shape[1] != len(feature_names):
                    st.warning("SHAP feature names do not match SHAP values. Importance plot is unavailable.")
                else:
                    mean_abs = np.mean(np.abs(sv), axis=0)
                    imp_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Mean |SHAP|": mean_abs
                    }).sort_values("Mean |SHAP|", ascending=False).head(20)

                    fig_bar = px.bar(
                        imp_df,
                        x="Mean |SHAP|",
                        y="Feature",
                        orientation="h",
                        color="Mean |SHAP|",
                        color_continuous_scale="Viridis",
                        title="Top Feature Importance (Mean |SHAP|)"
                    )
                    fig_bar.update_layout(yaxis_title="")
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.caption("Mean absolute SHAP value per feature (top 20).")
            except Exception as e:
                st.warning(f"Importance plot unavailable: {e}")

        st.divider()

        st.markdown("### 2. Feature Dependence Analysis")
        st.caption("See how a feature changes the predicted effort, and how features interact.")
        
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

        view = st.radio("Effect View", ["SHAP", "ALE (correlation-safe)"], horizontal=True, key="effect_view")

        # --- Feature Interaction Analysis (TreeSHAP) ---
        interaction_values, interaction_feature_names = get_shap_interaction_data(st.session_state['models'], X_train)
        selected_pair = None
        if interaction_values is not None:
            interaction_strength = np.abs(interaction_values).mean(axis=0)
            np.fill_diagonal(interaction_strength, 0.0)
            top_n = 10
            top_pairs = np.dstack(np.unravel_index(np.argsort(interaction_strength.ravel())[::-1], interaction_strength.shape))[0]

            rows = []
            for idx in top_pairs[:top_n]:
                i, j = int(idx[0]), int(idx[1])
                if i == j:
                    continue
                rows.append({
                    "Feature A": interaction_feature_names[i],
                    "Feature B": interaction_feature_names[j],
                    "Interaction Strength": interaction_strength[i, j]
                })

            if rows:
                inter_df = pd.DataFrame(rows).drop_duplicates(subset=["Feature A", "Feature B"])
                st.subheader("Top Feature Interactions")
                st.caption("Pairs with stronger values interact more in the model.")
                # Plot top interactions for quick visual scan
                inter_df["Pair"] = inter_df["Feature A"] + " x " + inter_df["Feature B"]
                fig_inter = px.bar(
                    inter_df.sort_values("Interaction Strength", ascending=True),
                    x="Interaction Strength",
                    y="Pair",
                    orientation="h",
                    color="Interaction Strength",
                    color_continuous_scale="RdBu_r",
                    title="Top Interaction Strengths"
                )
                fig_inter.update_layout(yaxis_title="", xaxis_title="Interaction Strength")
                st.plotly_chart(fig_inter, use_container_width=True)
                st.dataframe(
                    inter_df.style.format({"Interaction Strength": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True
                )

                pair_labels = [f"{r['Feature A']} x {r['Feature B']}" for _, r in inter_df.iterrows()]
                pair_choice = st.selectbox(
                    "Select an interaction pair to highlight",
                    ["Auto-Detect (Strongest Interaction)"] + pair_labels,
                    key="interaction_pair_select"
                )
                if pair_choice != "Auto-Detect (Strongest Interaction)":
                    selected_pair = pair_choice.split(" x ")

        # Determine the interaction_index for shap.dependence_plot
        if selected_pair:
            interaction_index_to_use = selected_pair[1]
            interaction_feature_name = selected_pair[1]
        elif interaction_selection == 'Auto-Detect (Strongest Interaction)':
            interaction_index_to_use = "auto"
            interaction_feature_name = "the most strongly interacting feature (auto-detected)"
        else:
            interaction_index_to_use = interaction_selection
            interaction_feature_name = interaction_selection

        if view == "SHAP":
            st.caption("SHAP values show how much a feature increases or decreases the predicted effort.")
            try:
                sv = np.array(shap_values)
                if sv.ndim == 1:
                    sv = sv.reshape(-1, 1)
                if dependence_feature not in feature_names:
                    st.warning("Selected feature is not available in SHAP feature list.")
                else:
                    f_idx = feature_names.index(dependence_feature)
                    x_vals = X_train_transformed[:, f_idx]
                    y_vals = sv[:, f_idx]

                    # Color by interaction feature if available
                    color_vals = None
                    color_label = None
                    if interaction_index_to_use != "auto" and interaction_index_to_use in feature_names:
                        i_idx = feature_names.index(interaction_index_to_use)
                        color_vals = X_train_transformed[:, i_idx]
                        color_label = interaction_index_to_use
                    elif interaction_index_to_use == "auto":
                        color_vals = None
                        color_label = None

                    # Sample for performance
                    max_points = 3000
                    n_rows = len(x_vals)
                    if n_rows > max_points:
                        sample_idx = np.random.choice(n_rows, size=max_points, replace=False)
                    else:
                        sample_idx = np.arange(n_rows)

                    dep_df = pd.DataFrame({
                        dependence_feature: x_vals[sample_idx],
                        "SHAP": y_vals[sample_idx]
                    })
                    if color_vals is not None:
                        dep_df[color_label] = color_vals[sample_idx]

                    fig_dep = px.scatter(
                        dep_df,
                        x=dependence_feature,
                        y="SHAP",
                        color=color_label if color_vals is not None else None,
                        color_continuous_scale="RdBu",
                        title=f"SHAP Dependence: {dependence_feature}"
                    )
                    fig_dep.update_traces(marker=dict(opacity=0.6, size=6))
                    fig_dep.update_layout(yaxis_title="SHAP Value")
                    st.plotly_chart(fig_dep, use_container_width=True)
            except Exception as e:
                st.warning(f"Dependence plot unavailable: {e}")
        else:
            st.caption("ALE plots avoid unrealistic feature combinations when features are correlated.")
            # ALE requires raw feature names (not one-hot)
            if dependence_feature in ['project_type', 'material_type'] or dependence_feature.startswith('project_type_') or dependence_feature.startswith('material_type_'):
                st.info("ALE is not available for one-hot encoded features. Select a numeric feature.")
            else:
                model_pipe_for_ale = st.session_state['models']

                ale_df = get_ale_data(model_pipe_for_ale, X_train, dependence_feature)
                if ale_df is None:
                    st.warning("ALE requires the pyALE package. Please install pyALE to enable this view.")
                else:
                    fig_ale = px.line(
                        ale_df,
                        x=dependence_feature,
                        y="eff",
                        title=f"ALE Plot: {dependence_feature}",
                        color_discrete_sequence=[DASKAN_GREEN]
                    )
                    fig_ale.update_layout(xaxis_title=dependence_feature, yaxis_title="ALE Effect")
                    st.plotly_chart(fig_ale, use_container_width=True)

        # Update caption for clarity
        cap_text = f"This view shows the effect of **{dependence_feature}** on the predicted target. Color indicates **{interaction_feature_name}**."
        st.caption(cap_text)
        
        st.divider()
        st.markdown("### 3. Classic SHAP Plots (Structural Drivers)")
        with st.expander("Show SHAP Summary + Waterfall", expanded=False):
            st.caption("Matplotlib-based SHAP views for detailed driver inspection.")
            try:
                # Summary plot
                fig_sum, ax_sum = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_train_transformed, feature_names=feature_names, show=False)
                st.pyplot(fig_sum)
                plt.close(fig_sum)

                # Waterfall plot for a single instance
                max_idx = min(200, X_train_transformed.shape[0]) - 1
                idx = st.slider("Select Example Index", 0, max_idx, 0, key="shap_waterfall_idx")
                base_val = explainer.expected_value
                if isinstance(base_val, (list, np.ndarray)):
                    base_val = base_val[0]
                exp = shap.Explanation(
                    values=shap_values[idx],
                    base_values=base_val,
                    data=X_train_transformed[idx],
                    feature_names=feature_names
                )
                fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(exp, show=False)
                st.pyplot(fig_wf)
                plt.close(fig_wf)
            except Exception as e:
                st.warning(f"SHAP plots unavailable: {e}")
        
# ----------------------------------------------------------------------
# TAB 4: CLASSIFICATION
# ----------------------------------------------------------------------
with tabs[3]:
    st.header(" Structural Complexity Classification")
    if 'project_complexity_class' not in df_modeling.columns:
        st.warning("project_complexity_class is missing. Run feature engineering or check data.")
    else:
        Xc = df_modeling[available_features]
        yc = df_modeling['project_complexity_class']
        if yc.dropna().shape[0] < 10:
            st.warning("Not enough labeled rows to train classification.")
        else:
            Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)
            clf_choice = st.selectbox(
                "Classifier",
                ["Logistic Regression", "Gradient Boosting Classifier"],
                key="clf_choice"
            )
            if st.button("Train Classifier", key="train_clf"):
                try:
                    clf_pipe = build_model("classification", clf_choice, selected_num_features, CAT_FEATURES)
                    high_weight = 2.0
                    class_weights = np.where(yc_train == "High", high_weight, 1.0)
                    clf_pipe.fit(Xc_train, yc_train, model__sample_weight=class_weights)
                    preds = clf_pipe.predict(Xc_test)
                    acc = accuracy_score(yc_test, preds)
                    recall_high = recall_score(yc_test, preds, labels=["High"], average="macro", zero_division=0)
                    f1 = f1_score(yc_test, preds, average="macro")
                    st.metric("Accuracy", f"{acc:.3f}")
                    st.metric("Recall (High)", f"{recall_high:.3f}")
                    st.metric("F1 (Macro)", f"{f1:.3f}")
                    st.success("Classification training complete.")

                    st.session_state['clf_model'] = clf_pipe
                    st.session_state['clf_choice_model'] = clf_choice
                except Exception as e:
                    st.error(f"Classification Error: {e}")

            if st.session_state.get('clf_model') is not None and st.session_state.get('clf_choice_model') == "Gradient Boosting Classifier":
                with st.expander("SHAP: Structural Drivers (Classifier)", expanded=False):
                    try:
                        model_pipe = st.session_state['clf_model']
                        preprocessor = model_pipe.named_steps['prep']
                        model = model_pipe.named_steps['model']
                        Xc_train_tx = preprocessor.transform(Xc_train)
                        if hasattr(Xc_train_tx, "toarray"):
                            Xc_train_tx = Xc_train_tx.toarray()
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(Xc_train_tx)
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                        try:
                            feature_names = list(preprocessor.get_feature_names_out())
                        except Exception:
                            feature_names = selected_num_features
                        fig_sum, ax_sum = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, Xc_train_tx, feature_names=feature_names, show=False)
                        st.pyplot(fig_sum)
                        plt.close(fig_sum)
                    except Exception as e:
                        st.warning(f"SHAP classifier plot unavailable: {e}")

# ----------------------------------------------------------------------
# TAB 5: SMART QUOTATION 
# ----------------------------------------------------------------------
with tabs[4]:
    st.header(" Project Quotation & Resource Planner")
    
    # Check for an APPROVED model version for a more sophisticated check
    if st.session_state['approved_model_version'] is None and st.session_state['models'] is None:
        st.warning("Train and/or **Approve** a model in the 'AI Model Engine' tab first.")
    else:
        # --- INPUTS ---
        st.subheader("1. Project Specification")
        c1, c2, c3, c4 = st.columns(4)
        i_area = c1.number_input("Surface Area (m^2)", 100, 50000, 800, key='i_area')
        i_levels = c2.number_input("Levels", 1, 50, 2, key='i_levels')
        i_height = c3.number_input("Building Height (m)", 3, 300, 12, key='i_height')
        i_units = c4.number_input("Number of Units", 0, 10000, 20, key='i_units')
        
        c5, c6, c7 = st.columns(3)
        i_type = c5.selectbox("Type", df_modeling['project_type'].dropna().unique(), key='i_type')
        i_scope = c6.selectbox("Scope Category", df_modeling['scope_category'].dropna().unique(), key='i_scope')
        i_mat = c7.selectbox("Material", df_modeling['material_type'].dropna().unique(), key='i_mat')

        c8, c9, c10 = st.columns(3)
        use_planned_dates = c8.checkbox("Provide Planned Dates", value=False, key='use_planned_dates')
        use_corrected_dates = c9.checkbox("Provide Corrected Dates", value=False, key='use_corrected_dates')
        i_season = c10.selectbox(
            "Season Flag",
            df_modeling['season_flag'].dropna().unique() if 'season_flag' in df_modeling.columns else ['Unknown'],
            key='i_season'
        )
        c11, c12, c13 = st.columns(3)
        i_holiday = c11.selectbox(
            "Holiday Period Flag",
            df_modeling['holiday_period_flag'].dropna().unique() if 'holiday_period_flag' in df_modeling.columns else ['Unknown'],
            key='i_holiday'
        )
        i_planned_start = c12.date_input("Planned Start Date", value=datetime.today(), key='i_planned_start', disabled=not use_planned_dates)
        i_planned_end = c13.date_input("Planned End Date", value=datetime.today(), key='i_planned_end', disabled=not use_planned_dates)
        c14, c15 = st.columns(2)
        i_corrected_start = c14.date_input("Corrected Start Date", value=datetime.today(), key='i_corrected_start', disabled=not use_corrected_dates)
        i_corrected_end = c15.date_input("Corrected End Date", value=datetime.today(), key='i_corrected_end', disabled=not use_corrected_dates)
        
        st.markdown("---")
        st.subheader("2. Financial & Risk Settings")
        col_res_1, col_res_2, col_res_3, col_res_4 = st.columns(4)
        avg_hourly_rate = col_res_1.number_input("Avg. Cost Rate ($/h)", 80, 200, AVG_HOURLY_RATE_CAD, key='avg_hourly_rate')
        profit_markup = col_res_2.slider("Profit/Markup (%)", 5, 50, 25) / 100
        hours_per_week_slider = col_res_3.number_input("Hours per Engineer/Week", 20, 40, HOURS_PER_WEEK, key='hours_per_week')
        i_duration = col_res_4.number_input("Expected Duration (days)", 1, 3650, 120, key='i_duration')
        
        col_res_5, col_res_6 = st.columns(2)
        i_revisions = col_res_5.number_input("Revisions (count)", 0, 50, 2, key='i_revisions')
        _ = col_res_6.empty()

        st.markdown("---")
        st.subheader("3. Post-Outcome Fields (Used in Quotation Only)")
        c16, c17, c18 = st.columns(3)
        i_actual_duration = c16.number_input("Actual Duration (days)", 0, 3650, 0, key='i_actual_duration')
        i_avg_hours_emp = c17.number_input("Avg. Hours per Employee", 0.0, 1000.0, 0.0, key='i_avg_hours_emp')
        i_design_hours_total = c18.number_input("Design Hours Total", 0.0, 100000.0, 0.0, key='i_design_hours_total')

        st.markdown("---")
        st.subheader("Civil Engineering Reasoning")
        st.caption(
            "Primary drivers: scope category, material type, surface area, levels/height, expected duration, and revisions. "
            "These reflect structural scale, complexity of load paths/material behavior, and coordination effort."
        )

        # Derived features
        floor_area_ratio = (i_area / i_levels) if i_levels > 0 else np.nan
        planned_start_ordinal = i_planned_start.toordinal() if use_planned_dates else np.nan
        planned_end_ordinal = i_planned_end.toordinal() if use_planned_dates else np.nan
        corrected_start_ordinal = i_corrected_start.toordinal() if use_corrected_dates else np.nan
        corrected_end_ordinal = i_corrected_end.toordinal() if use_corrected_dates else np.nan
        month_started = i_planned_start.month if use_planned_dates else np.nan
        quarter = ((i_planned_start.month - 1) // 3 + 1) if use_planned_dates else np.nan
        is_winter = int(i_planned_start.month in [12, 1, 2, 3]) if use_planned_dates else np.nan
        project_duration_days = i_actual_duration if i_actual_duration > 0 else i_duration

        input_payload = {
            'surface_area_m2': i_area,
            'building_height_m': i_height,
            'num_levels': i_levels,
            'floor_area_ratio': floor_area_ratio,
            'height_to_area_ratio': (i_height / i_area) if i_area > 0 else np.nan,
            'area_per_level': (i_area / i_levels) if i_levels > 0 else np.nan,
            'expected_duration_days': i_duration,
            'num_revisions': i_revisions,
            'is_winter': is_winter,
            'design_hours_total': i_design_hours_total if i_design_hours_total > 0 else np.nan,
            'avg_floor_height': (i_height / i_levels) if i_levels > 0 else np.nan,
            'material_type': i_mat
        }
        selected_num_features = get_selected_num_features(is_metadata, include_leaky=True)
        selected_features = selected_num_features + CAT_FEATURES
        input_data = pd.DataFrame([{col: input_payload.get(col, np.nan) for col in selected_features}])
        
        # Prediction Button (This updates the session state)
        if st.button(" RUN AI ESTIMATION", type="primary", use_container_width=True):
            if st.session_state['models'] is None:
                st.error("No model is trained yet. Please go to the 'AI Model Engine' tab to train and approve a model.")
            else:
                with st.spinner("Running AI Prediction..."):
                    # 1. Prediction
                    pred_val = st.session_state['models'].predict(input_data)[0]
                    # Heuristic risk margin based on civil-engineering drivers
                    complexity_flags = 0
                    if i_area >= 2500:
                        complexity_flags += 1
                    if i_levels >= 6 or i_height >= 25:
                        complexity_flags += 1
                    if i_duration >= 180:
                        complexity_flags += 1
                    if i_revisions >= 4:
                        complexity_flags += 1
                    base_risk = 0.10
                    risk_margin = pred_val * (base_risk + 0.05 * complexity_flags)
                    
                    # Store results in session state
                    st.session_state['pred_val'] = pred_val
                    st.session_state['risk_margin'] = risk_margin
                    st.session_state['input_data'] = input_data # Store the data used for prediction
                    st.session_state['profit_markup'] = profit_markup
                    st.session_state['quote_generated'] = True
                    st.success("Prediction Generated! Scroll down for plan.")
                
        
        st.markdown("---")

        # --- DYNAMIC OUTPUTS (Reacting to Session State) ---
        if st.session_state.get('quote_generated', False):
            
            pred_val = st.session_state['pred_val']
            risk_margin = st.session_state['risk_margin']
            profit_markup = st.session_state['profit_markup']
            
            # Recalculate financial/resource metrics based on current inputs (prediction is hours)
            est_hours = pred_val
            estimated_cost_base = est_hours * avg_hourly_rate
            competitive_quote = estimated_cost_base * (1 + profit_markup)
            conservative_quote = (estimated_cost_base + risk_margin * avg_hourly_rate) * (1 + profit_markup)
            
            # --- NEW: Data Drift Analysis ---
            st.markdown("---")
            st.subheader(" Input Data Drift & Reliability Check")
            
            train_stats = st.session_state['train_stats']
            is_drifting = False
            drift_message = ""
            
            # Check numerical features against training distribution
            for feature in selected_num_features:
                input_val = input_data[feature].iloc[0]
                mean_val = train_stats.loc[feature, 'mean']
                std_val = train_stats.loc[feature, 'std']
                
                # Simple drift detection: > 2 standard deviations away from the mean
                if pd.notna(input_val) and std_val > 0 and abs(input_val - mean_val) > 2 * std_val:
                    is_drifting = True
                    drift_message += f"- **{feature}** value ({input_val:.2f}) is significantly outside the historical range (Mean: {mean_val:.2f}, Std Dev: {std_val:.2f}).\n"
            
            if is_drifting:
                st.error(" **MODEL RELIABILITY WARNING:** The input project significantly deviates from the training data distribution in the following areas. Predictions may be **unreliable**.")
                st.markdown(drift_message)
            else:
                st.success(" **Data Quality Check:** Input parameters are well within the historical data distribution. Model predictions are expected to be reliable.")
            st.markdown("---")

            
            # 1. Summary & Financials
            st.markdown("#### Project Summary & Financials")
            # Structural red-flag check
            if 'total_project_effort' in df_modeling.columns:
                high_hours_threshold = df_modeling['total_project_effort'].quantile(0.8)
                if est_hours > high_hours_threshold and i_revisions < 3:
                    st.warning("Warning: High structural complexity with low revision history—check for incomplete design data.")
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            
            col_p1.metric("Estimated Effort (h)", f"{pred_val:.0f} Hours", f"Risk: +/- {risk_margin:.0f}h" if risk_margin > 0 else "Point Estimate")
            col_p2.metric("Base Cost (CAD)", f"${estimated_cost_base:,.0f}")
            col_p3.metric("Competitive Quote", f"${competitive_quote:,.0f}", f"{int(profit_markup*100)}% markup")
            col_p4.metric("Conservative Quote", f"${conservative_quote:,.0f}", "Risk + Profit", delta_color="normal")
            
            
            # 2. Resource Planning (Reactive to Slider)
            st.markdown("####  Resource & Timeline Planning")
            
            team_size = st.slider("Select Planned Team Size (Engineers) for Schedule", 1, 10, 2, key='team_size_live')
            
            weeks_needed_1_eng = pred_val / hours_per_week_slider
            real_duration = weeks_needed_1_eng / team_size
            
            c_res_1, c_res_2, c_res_3 = st.columns(3)
            
            c_res_1.metric("Duration (1 Engineer)", f"{weeks_needed_1_eng:.1f} Weeks")
            c_res_2.metric(f"Duration ({team_size} Engineers)", f"{real_duration:.1f} Weeks", delta="Project Duration")

            # Risk Flag
            high_risk = (i_revisions > 5)
            if real_duration > 15 or risk_margin > 50 or is_drifting or high_risk:
                st.error(" **High Risk Alert:** Long duration, high uncertainty, data drift, or high revisions detected. Increase team size or pad the quote.")
            else:
                st.success(" **Standard Project:** Project fits typical historical parameters.")


            st.markdown("---")
            
            # 3. Task Breakdown & Gantt
            st.markdown("####  Recommended Task Breakdown & Schedule")
            
            has_granular_cols = all(col in df_analytics.columns for col in ['date_logged', 'hours_worked', 'task_category'])
            if not has_granular_cols:
                st.info("Task breakdown and Gantt require granular timesheet data. Upload full exploration data to enable.")
            else:
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

# ----------------------------------------------------------------------
# TAB 5: CLUSTERING & PERSONAS
# ----------------------------------------------------------------------
with tabs[5]:
    st.header(" Clustering & Personas")
    st.caption("K-Means clustering on numeric inputs to surface project personas.")
    
    if df_modeling[NUM_FEATURES].dropna(how='all').empty:
        st.warning("Not enough numeric data available to run clustering.")
    else:
        k = st.slider("Number of Clusters (k)", 2, 8, 3, key="kmeans_k")
        
        # Prep data
        num_df = df_modeling[NUM_FEATURES].copy()
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_num = imputer.fit_transform(num_df)
        X_scaled = scaler.fit_transform(X_num)
        if len(num_df) < k:
            st.warning("Not enough rows to form the selected number of clusters.")
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Attach cluster labels
            cluster_df = num_df.copy()
            cluster_df['cluster'] = clusters
            if 'material_type' in df_modeling.columns:
                cluster_df['material_type'] = df_modeling.loc[cluster_df.index, 'material_type'].astype(str)

            def label_cluster(sub_df):
                levels = sub_df['num_levels'].mean() if 'num_levels' in sub_df.columns else np.nan
                mat = sub_df['material_type'].mode().iloc[0] if 'material_type' in sub_df.columns and not sub_df['material_type'].mode().empty else "Mixed"
                mat_l = str(mat).lower()
                if pd.notna(levels) and levels <= 3 and ('wood' in mat_l or 'timber' in mat_l):
                    return "Light Timber Residential"
                if pd.notna(levels) and levels >= 8 and 'concrete' in mat_l:
                    return "Complex Concrete High-Rise"
                if pd.notna(levels) and levels >= 6 and 'steel' in mat_l:
                    return "Steel Mid/High-Rise"
                if pd.notna(levels) and levels >= 5:
                    return "Dense Mid-Rise Mixed"
                return "Low-Rise Mixed"

            cluster_labels = cluster_df.groupby('cluster').apply(label_cluster)
            cluster_df['cluster_label'] = cluster_df['cluster'].map(cluster_labels)
            
            # Silhouette score
            sil_score = None
            if k > 1 and len(cluster_df) > k:
                try:
                    sil_score = silhouette_score(X_scaled, clusters)
                except Exception:
                    sil_score = None
            
            col_c1, col_c2 = st.columns(2)
            col_c1.metric("Clusters", f"{k}")
            if sil_score is not None:
                col_c2.metric("Silhouette Score", f"{sil_score:.3f}")
            else:
                col_c2.metric("Silhouette Score", "n/a")
            
            st.markdown("### Cluster Size")
            st.dataframe(
                cluster_df['cluster_label'].value_counts().rename_axis('cluster_label').reset_index(name='count'),
                use_container_width=True,
                hide_index=True
            )
            
            # Cluster profiles
            st.markdown("### Cluster Profiles (Mean)")
            profile_df = cluster_df.groupby('cluster')[NUM_FEATURES].mean().reset_index()
            profile_df['cluster_label'] = profile_df['cluster'].map(cluster_labels)
            st.dataframe(profile_df.style.format(precision=2), use_container_width=True, hide_index=True)
            
            # 2D visualization using PCA
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            pca_df['cluster'] = [cluster_labels[c] for c in clusters]
            
            fig_cluster = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='cluster',
                title="Cluster Map (PCA Projection)",
                color_discrete_sequence=DASKAN_COLOR_PALETTE
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

