import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
from datetime import datetime, timedelta, timezone
import io
import zipfile

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score, accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Dataset handlers
from data_handlers.common import _parse_date_column, get_selected_num_features
from modules.data import load_uploaded_csv, compute_train_stats
from modules.modeling import build_model, train_and_tune_model, train_classifier, get_regression_registry
from modules.explainability import get_shap_data, get_shap_interaction_data, get_ale_data
from modules.diagnostics import detect_leakage, select_features, stability_report, residual_outliers

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
    'num_levels',
    'num_units',
    'building_height_m',
    'floor_area_ratio',
    'actual_duration_days',
    'num_revisions',
    'area_per_unit',
    'height_per_level',
    'complexity_interaction_index',
    'revision_intensity'
]
LEAKY_NUM_FEATURES = []
CAT_FEATURES = [
    'project_type',
    'material_type',
    'scope_category',
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
    st.session_state['last_train_metrics'] = None
    st.session_state['last_train_features'] = None
    st.session_state['last_cv_metrics'] = None
    st.session_state['last_baseline_mae'] = None
    st.session_state['clf_metrics'] = None
    st.session_state['clf_confusion_matrix'] = None
    st.session_state['clf_last_error'] = None
    st.session_state['tuning_history'] = []
    st.session_state['tuning_attempt_counter'] = 0
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
    st.subheader("Training Controls")
    enable_leakage_guard = st.checkbox(
        "Leakage Guard",
        value=False,
        help="Warns and removes post-outcome features that could leak target information."
    )
    split_mode_ui = st.selectbox(
        "Split Strategy",
        ["Auto (Stratified Random)", "Group-Aware (project_id)", "Time-Aware (Chronological)"],
        index=0,
        help="Auto uses stratified random split on binned target; use Group for repeated project IDs; use Time for chronological holdout."
    )
    include_leaky_features = False

# Data Loading & Processing
profile = "unknown"
if uploaded_file is not None:
    try:
        df_modeling, df_analytics, profile, data_source = load_uploaded_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        st.stop()
else:
    st.warning("Please upload a CSV file to begin.")
    st.stop()

# Dataset profile flags
is_granular = profile == "granular"
is_metadata = profile == "metadata"

def run_data_quality_checks(df_raw, df_model, selected_features):
    """Basic data quality and metadata checks for reporting."""
    required_cols = ['project_id'] + selected_features + [
        'total_project_hours'
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
        'actual_duration_days': (0, None),
        'num_revisions': (0, None),
        'total_project_hours': (0, None)
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

    # Duplicates
    if 'project_id' in df_model.columns:
        dup = df_model['project_id'].duplicated().sum()
        if dup > 0:
            issues.append(f"project_id duplicates: {dup}")

    # Training metadata checks
    meta_issues = []
    if 'total_project_hours' not in df_model.columns:
        meta_issues.append("Target missing: total_project_hours")
    else:
        if df_model['total_project_hours'].isna().mean() > 0.2:
            meta_issues.append("Target missing rate > 20%")
    # project_complexity_class is optional for metadata-only datasets

    feature_missing = missing_df[missing_df["Missing Rate"] > 0.2]
    if not feature_missing.empty:
        meta_issues.append("Model features with missing rate > 20%")

    return missing_df, issues, meta_issues


def add_interaction_features(df):
    """Add domain interaction features to improve small-data signal capture."""
    df = df.copy()
    for col in ['surface_area_m2', 'num_units', 'building_height_m', 'num_levels', 'num_revisions', 'actual_duration_days']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if {'surface_area_m2', 'num_units'}.issubset(df.columns):
        df['area_per_unit'] = df['surface_area_m2'] / (df['num_units'].fillna(0) + 1.0)

    if {'building_height_m', 'num_levels'}.issubset(df.columns):
        df['height_per_level'] = df['building_height_m'] / (df['num_levels'].fillna(0) + 1.0)

    if {'surface_area_m2', 'num_levels'}.issubset(df.columns):
        df['complexity_interaction_index'] = df['surface_area_m2'] * df['num_levels']

    if {'num_revisions', 'actual_duration_days'}.issubset(df.columns):
        df['revision_intensity'] = df['num_revisions'] / (df['actual_duration_days'].fillna(0) + 1.0)

    return df


df_modeling = add_interaction_features(df_modeling)

# Calculate training data statistics for drift monitoring
# (after feature engineering so derived features are included)
num_cols = NUM_FEATURES + LEAKY_NUM_FEATURES
st.session_state['train_stats'] = compute_train_stats(df_modeling, num_cols)


def _make_regression_strata(y, max_bins=8):
    """Create quantile bins for stratified regression splits; fallback to None."""
    y = pd.Series(y)
    if y.dropna().shape[0] < 20:
        return None
    n_bins = min(max_bins, max(2, y.dropna().shape[0] // 25))
    try:
        bins = pd.qcut(y, q=n_bins, duplicates='drop')
    except Exception:
        return None
    if bins.nunique(dropna=True) < 2:
        return None
    counts = bins.value_counts()
    if counts.empty or counts.min() < 2:
        return None
    return bins.astype(str)


def split_regression_best_practice(X, y, sample_weight=None, split_mode="auto", test_size=0.2, random_state=42):
    """
    Split data using best-practice strategy:
    - auto: stratified random split on binned target (fallback to random)
    - group: group-aware split on project_id when duplicates exist
    - time: chronological split using available start-date columns
    """
    X = X.copy()
    y = pd.Series(y, index=X.index)
    weights = None if sample_weight is None else pd.Series(sample_weight, index=X.index)

    # Group-aware split when requested and possible
    if split_mode == "group" and 'project_id' in X.columns:
        groups = X['project_id']
        if groups.nunique(dropna=True) > 1 and groups.duplicated(keep=False).any():
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            tr_idx, te_idx = next(splitter.split(X, y, groups=groups))
            idx_train = X.index[tr_idx]
            idx_test = X.index[te_idx]
            X_train = X.loc[idx_train].drop(columns=['project_id'], errors='ignore')
            X_test = X.loc[idx_test].drop(columns=['project_id'], errors='ignore')
            y_train = y.loc[idx_train]
            y_test = y.loc[idx_test]
            if weights is None:
                return X_train, X_test, y_train, y_test, None, None, "group"
            return X_train, X_test, y_train, y_test, weights.loc[idx_train], weights.loc[idx_test], "group"

    # Time-aware split when requested and possible
    if split_mode == "time":
        date_candidates = ['planned_start_date', 'corrected_start_date', 'start_date']
        time_col = next((c for c in date_candidates if c in X.columns), None)
        if time_col is not None:
            date_series = pd.to_datetime(X[time_col], errors='coerce')
            valid_idx = date_series.dropna().sort_values().index
            if len(valid_idx) >= 10:
                n_test = max(1, int(np.ceil(len(valid_idx) * test_size)))
                idx_test = valid_idx[-n_test:]
                idx_train = X.index.difference(idx_test)
                X_train = X.loc[idx_train].drop(columns=['project_id'], errors='ignore')
                X_test = X.loc[idx_test].drop(columns=['project_id'], errors='ignore')
                y_train = y.loc[idx_train]
                y_test = y.loc[idx_test]
                if weights is None:
                    return X_train, X_test, y_train, y_test, None, None, "time"
                return X_train, X_test, y_train, y_test, weights.loc[idx_train], weights.loc[idx_test], "time"

    # Default: stratified random split for regression (binned y), fallback random
    stratify_bins = _make_regression_strata(y) if split_mode in ["auto", "stratified"] else None
    if weights is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X.drop(columns=['project_id'], errors='ignore'),
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_bins
        )
        return X_train, X_test, y_train, y_test, None, None, "stratified_random" if stratify_bins is not None else "random"

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X.drop(columns=['project_id'], errors='ignore'),
        y,
        weights,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_bins
    )
    return X_train, X_test, y_train, y_test, w_train, w_test, "stratified_random" if stratify_bins is not None else "random"


# --- MAIN APP TITLE ---
st.title("Structural Project Intelligence Dashboard")
st.markdown(f"**Current Data Source:** `{data_source}` | **Total Projects:** `{len(df_modeling)}`")
st.divider()

tabs = st.tabs([
    "Deep Dive Analytics",
    "AI Model Engine",
    "Model Explainability (XAI)",
    "Classification",
    "Smart Quotation",
    "Clustering & Personas"
])

# ----------------------------------------------------------------------
# TAB 1: ANALYTICS
# ----------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Historical Project Insights")
    st.caption(f"Dataset profile: {profile}")
    
    col_kpis_1, col_kpis_2, col_kpis_3 = st.columns(3)
    col_kpis_1.metric("Average Project Effort", f"{df_modeling['total_project_hours'].mean():.0f} Hours")
    col_kpis_2.metric("Most Common Type", df_modeling['project_type'].mode()[0])
    if 'complexity_index' in df_modeling.columns:
        col_kpis_3.metric("Max Complexity Index", f"{df_modeling['complexity_index'].max():.2f}")
    else:
        col_kpis_3.metric("Max Complexity Index", "N/A")

    c_m1, c_m2 = st.columns([2, 1])
    
    with c_m1:
        st.subheader("Macro View: Effort vs. Scale")
        fig = px.scatter(df_modeling, x='surface_area_m2', y='total_project_hours', 
                         color='project_type', size='complexity_index' if 'complexity_index' in df_modeling.columns else None,
                         hover_data=['num_levels', 'material_type'], 
                         title="Project Effort (Hours) by Area and Complexity",
                         color_discrete_sequence=DASKAN_COLOR_PALETTE)
        st.plotly_chart(fig, use_container_width=True)
        
    with c_m2:
        st.subheader("Distribution")
        # Effort Distribution
        fig_hist = px.histogram(df_modeling, x='total_project_hours', color_discrete_sequence=[DASKAN_GREEN], 
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
        core_cols = selected_features_for_checks + ['total_project_hours']
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

    with st.expander("Project Complexity By Project", expanded=False):
        st.caption("Computed complexity class per project (rule-based).")

        def _compute_complexity_class(df):
            dfc = df.copy()
            duration_col = None
            for col in ['expected_duration_days', 'actual_duration_days', 'project_duration_days']:
                if col in dfc.columns:
                    duration_col = col
                    break

            area_q1 = dfc['surface_area_m2'].dropna().quantile(0.25) if 'surface_area_m2' in dfc.columns else np.nan
            area_q3 = dfc['surface_area_m2'].dropna().quantile(0.75) if 'surface_area_m2' in dfc.columns else np.nan
            dur_q1 = dfc[duration_col].dropna().quantile(0.25) if duration_col else np.nan
            dur_q3 = dfc[duration_col].dropna().quantile(0.75) if duration_col else np.nan

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

            dfc['project_complexity_class'] = dfc.apply(_classify, axis=1)
            return dfc

        df_complex = _compute_complexity_class(df_modeling)
        if df_complex['project_id'].isna().any():
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
            'actual_duration_days',
            'num_revisions'
        ]
        display_cols = [c for c in display_cols if c in df_complex.columns]
        st.dataframe(
            df_complex[display_cols].sort_values('project_id'),
            use_container_width=True,
            hide_index=True
        )

    # Feature Correlation Heatmap
    st.markdown("### Feature Correlation Analysis")
    st.caption("Understand linear relationships between numerical features and project effort.")
    
    numerical_features = NUM_FEATURES + (LEAKY_NUM_FEATURES if include_leaky_features else []) + ['total_project_hours']
    numerical_features = [c for c in numerical_features if c in df_modeling.columns]
    corr_df = df_modeling[numerical_features].copy()
    # Guard against duplicate column names (can happen with messy CSV headers)
    if corr_df.columns.duplicated().any():
        corr_df = corr_df.loc[:, ~corr_df.columns.duplicated()]
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
        "**Negative Correlation (Blue, closer to -1.0):** As one feature **increases**, the other tends to **decrease** (e.g., shorter `actual_duration_days` can be associated with higher `num_revisions`)."
        "\n\n"
        "**Positive Correlation (Red, closer to +1.0):** As one feature **increases**, the other tends to **increase** (e.g., higher `num_levels` often drives higher `total_project_hours`). The goal is to identify features that strongly correlate with the target variable (`total_project_hours`)."
    )
    # --- END CAPTION ADDED ---

    st.divider()
    
    # Granular Deep Dive (S-Curve & Task)
    st.markdown("### Project Timeline Analysis (S-Curve)")
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
    st.header("Effort Estimation Models")
    r2_threshold = 0.7

    def _append_tuning_history(
        trigger,
        model_name,
        split_used,
        cv_score,
        test_r2_val,
        test_mae_val,
        train_rows_used,
        test_rows_used,
        cv_fold_used,
        outliers_removed=0,
        seed_used=42,
        n_iter_scale_used=1.0
    ):
        attempt = int(st.session_state.get('tuning_attempt_counter', 0)) + 1
        st.session_state['tuning_attempt_counter'] = attempt
        row = {
            "attempt": attempt,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "trigger": trigger,
            "model": model_name,
            "split_mode": split_used,
            "train_rows": int(train_rows_used),
            "test_rows": int(test_rows_used),
            "cv_folds": int(cv_fold_used),
            "cv_r2_best_or_mean": float(cv_score),
            "test_r2": float(test_r2_val),
            "test_mae": float(test_mae_val),
            "outliers_removed": int(outliers_removed),
            "seed": int(seed_used),
            "n_iter_scale": float(n_iter_scale_used),
            "threshold_passed": bool(test_r2_val >= r2_threshold),
        }
        hist = st.session_state.get('tuning_history', [])
        hist.append(row)
        st.session_state['tuning_history'] = hist
    
    # Training readiness checks
    train_issues = []
    use_leaky = include_leaky_features
    selected_num_features = get_selected_num_features(NUM_FEATURES, LEAKY_NUM_FEATURES, is_metadata, include_leaky=use_leaky)
    selected_features = selected_num_features + CAT_FEATURES
    missing_features = [c for c in selected_features if c not in df_modeling.columns]
    if missing_features:
        st.warning("Some training features are missing and will be ignored:")
        st.write(", ".join(missing_features))

    # Fixed target (regression): total_project_hours (hours)
    target_col = "total_project_hours"
    st.session_state['target_col'] = target_col

    # Leakage guardrails (toggle)
    if enable_leakage_guard:
        leakage_cols = detect_leakage(selected_features, target_col, allow_leakage=False)
        if leakage_cols:
            st.warning("Potential leakage features detected and excluded from training:")
            st.write(", ".join(leakage_cols))
            selected_num_features = [c for c in selected_num_features if c not in leakage_cols]
            selected_features = [c for c in selected_features if c not in leakage_cols]

    # Feature selection
    kept_num, kept_cat, dropped_feats = select_features(df_modeling, selected_num_features, CAT_FEATURES)
    if dropped_feats:
        dropped_df = pd.DataFrame(dropped_feats, columns=["feature", "reason"])
        with st.expander("Feature Selection Summary", expanded=False):
            st.caption("Low-variance and highly-correlated features were removed.")
            st.dataframe(dropped_df, use_container_width=True, hide_index=True)
    selected_num_features = kept_num
    selected_features = kept_num + kept_cat

    # Drop high-missing features (helps small datasets)
    high_missing = []
    if selected_features:
        missing_rates = df_modeling[selected_features].isna().mean()
        high_missing = missing_rates[missing_rates > 0.4].index.tolist()
        if high_missing:
            selected_num_features = [c for c in selected_num_features if c not in high_missing]
            selected_features = [c for c in selected_features if c not in high_missing]

    col_m1, col_m2, col_m3 = st.columns([1.2, 1.1, 1])

    with col_m2:
        st.subheader("Data Fit Diagnostics")
        st.caption("Quick checks that commonly explain low R2 on small metadata datasets.")
        st.write(f"Rows: {len(df_modeling)}")
        if target_col in df_modeling.columns:
            tgt = df_modeling[target_col]
            st.write(f"Target missing rate: {tgt.isna().mean():.1%}")
            st.write(f"Target mean / median: {tgt.mean():.1f} / {tgt.median():.1f}")
            st.write(f"Target skew: {tgt.skew():.2f}")
        if high_missing:
            st.warning("Dropped high-missing features (>40% missing):")
            st.write(", ".join(high_missing))
        st.write(f"Active features: {len(selected_features)}")

    # Final available features after leakage + selection + missing filters
    available_features = [c for c in selected_features if c in df_modeling.columns]

    if target_col not in df_modeling.columns:
        train_issues.append(f"Missing target: {target_col}")
        X = df_modeling[available_features].copy()
        if 'project_id' in df_modeling.columns:
            X['project_id'] = df_modeling['project_id']
        y = pd.Series(dtype=float)
    else:
        X = df_modeling[available_features].copy()
        if 'project_id' in df_modeling.columns:
            X['project_id'] = df_modeling['project_id']
        y = df_modeling[target_col].copy()
        non_null_target = y.notna()
        dropped_target_rows = int((~non_null_target).sum())
        if dropped_target_rows > 0:
            X = X.loc[non_null_target]
            y = y.loc[non_null_target]
            st.caption(f"Dropped {dropped_target_rows} rows with missing target before splitting.")
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
        sample_weight = None
        hi_thresh = y.quantile(0.75)
        sample_weight = np.where(y > hi_thresh, 1.5, 1.0)
        split_mode_map = {
            "Auto (Stratified Random)": "auto",
            "Group-Aware (project_id)": "group",
            "Time-Aware (Chronological)": "time"
        }
        requested_split_mode = split_mode_map.get(split_mode_ui, "auto")
        X_train, X_test, y_train, y_test, w_train, w_test, split_mode_used = split_regression_best_practice(
            X, y, sample_weight=sample_weight, split_mode=requested_split_mode, test_size=0.2, random_state=42
        )
        if requested_split_mode != split_mode_used:
            st.caption(f"Requested split mode `{requested_split_mode}` fell back to `{split_mode_used}` due to data constraints.")
        else:
            st.caption(f"Using `{split_mode_used}` split strategy.")
        # Auto-select CV folds for training diagnostics
        cv_folds = 5 if len(df_modeling) < 500 else 7
        total_split_rows = max(1, len(X_train) + len(X_test))
        train_ratio = len(X_train) / total_split_rows
        test_ratio = len(X_test) / total_split_rows

        st.markdown("#### Split & Validation Setup")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Train Split", f"{train_ratio:.0%}", f"{len(X_train)} rows")
        s2.metric("Test Split", f"{test_ratio:.0%}", f"{len(X_test)} rows")
        s3.metric("Validation Method", "Cross-Validation", f"{cv_folds} folds")
        s4.metric("Split Strategy", split_mode_used.replace("_", " ").title())
        st.caption("Validation is performed using cross-validation on the training split; no fixed validation holdout is used.")
    
    with col_m1:
        st.subheader("Configuration")
        reg_registry = get_regression_registry()
        reg_keys = [k for k, v in reg_registry.items() if v["available"]]
        reg_labels = {k: reg_registry[k]["label"] for k in reg_keys}
        model_choice = st.selectbox(
            "Algorithm (Regression)",
            reg_keys,
            format_func=lambda k: reg_labels.get(k, k),
            key='model_select'
        )
        use_log_target = False
        remove_outliers = st.checkbox(
            "Exclude Top Residual Outliers",
            value=False,
            help="Removes the highest-error training rows before final training."
        )
        outlier_count = st.slider(
            "Outliers to Remove",
            min_value=1,
            max_value=10,
            value=3,
            disabled=not remove_outliers,
            help="Number of highest-residual training samples to exclude."
        )
        
    with col_m3:
        st.subheader("Action")
        st.markdown("---")
        if st.button("Train & Tune Model", type="primary", use_container_width=True, disabled=train_blocked):
            st.session_state['quote_generated'] = False # Reset quote when training new model
            with st.spinner("Executing Training & Hyperparameter Search..."):
                try:
                    # Optional outlier filtering (train-only) to test impact
                    X_train_use, y_train_use = X_train, y_train
                    baseline_metrics = None
                    if remove_outliers:
                        base_model, _ = train_and_tune_model(
                            X_train, y_train, model_choice,
                            selected_num_features, CAT_FEATURES, use_log_target,
                            sample_weight=w_train if 'w_train' in locals() else None
                        )
                        train_preds = base_model.predict(X_train)
                        resid = (y_train - train_preds).abs()
                        drop_idx = resid.sort_values(ascending=False).head(outlier_count).index
                        X_train_use = X_train.drop(index=drop_idx)
                        y_train_use = y_train.drop(index=drop_idx)
                        st.caption(f"Outliers removed from training: {len(drop_idx)}")

                        base_preds = base_model.predict(X_test)
                        baseline_metrics = {
                            "mae": float(mean_absolute_error(y_test, base_preds)),
                            "r2": float(r2_score(y_test, base_preds))
                        }

                    tuned_model, performance_metric = train_and_tune_model(
                        X_train_use, y_train_use, model_choice,
                        selected_num_features, CAT_FEATURES, use_log_target,
                        sample_weight=None,
                        random_state=42,
                        n_iter_scale=1.0
                    )
                    st.session_state['models'] = tuned_model
                    st.session_state['mode'] = 'point'
                    
                    preds = tuned_model.predict(X_test)
                    mae_test = mean_absolute_error(y_test, preds)
                    r2_test = r2_score(y_test, preds)
                    st.session_state['train_r2'] = r2_test # Store R2 for display
                    st.metric("Test MAE (Tuned Model)", f"{mae_test:.1f} Hours")
                    st.metric("Test R2 (Tuned Model)", f"{r2_test:.3f}")
                    st.caption(f"Best CV R2 during tuning: {performance_metric:.3f}")
                    if baseline_metrics is not None:
                        delta_r2 = r2_test - baseline_metrics["r2"]
                        delta_mae = mae_test - baseline_metrics["mae"]
                        impact_df = pd.DataFrame([{
                            "Baseline R2": baseline_metrics["r2"],
                            "Baseline MAE": baseline_metrics["mae"],
                            "With Outliers Removed R2": r2_test,
                            "With Outliers Removed MAE": mae_test,
                            "ΔR2": delta_r2,
                            "ΔMAE": delta_mae
                        }])
                        st.dataframe(
                            impact_df.style.format({
                                "Baseline R2": "{:.3f}",
                                "Baseline MAE": "{:.1f}",
                                "With Outliers Removed R2": "{:.3f}",
                                "With Outliers Removed MAE": "{:.1f}",
                                "ΔR2": "{:+.3f}",
                                "ΔMAE": "{:+.1f}"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                    st.caption(f"Approx. prediction interval: +/-{mae_test * 1.5:.1f} hours (1.5 x MAE)")

                    # Store predictions for visualization and MLOps simulation
                    st.session_state['y_test'] = y_test
                    st.session_state['y_preds'] = preds
                    # Baseline comparison (mean predictor)
                    try:
                        baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
                        baseline_mae = mean_absolute_error(y_test, baseline_pred)
                        st.session_state['last_baseline_mae'] = float(baseline_mae)
                        st.caption(f"Baseline MAE (mean predictor): {baseline_mae:.1f} hours")
                    except Exception:
                        pass

                    # R2 threshold gate (engineering estimation reliability)
                    if r2_test < r2_threshold:
                        st.warning("R2 below 0.70. Model may be unreliable for engineering estimation.")
                    st.session_state['last_train_metrics'] = {
                        "model": model_choice,
                        "mae": float(mae_test),
                        "r2": float(r2_test),
                        "rows": int(len(X_train) + len(X_test)),
                        "features": ", ".join(selected_features)
                    }
                    st.session_state['last_train_features'] = selected_features
                    _append_tuning_history(
                        trigger="initial_train",
                        model_name=model_choice,
                        split_used=split_mode_used,
                        cv_score=performance_metric,
                        test_r2_val=r2_test,
                        test_mae_val=mae_test,
                        train_rows_used=len(X_train_use),
                        test_rows_used=len(X_test),
                        cv_fold_used=cv_folds,
                        outliers_removed=(len(X_train) - len(X_train_use)),
                        seed_used=42,
                        n_iter_scale_used=1.0
                    )
                    
                    # CV summary for selected model
                    try:
                        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        cv_mae_scores = cross_val_score(
                            tuned_model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1
                        )
                        cv_r2_scores = cross_val_score(
                            tuned_model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1
                        )
                        st.session_state['last_cv_metrics'] = {
                            "cv_mae_mean": float(-cv_mae_scores.mean()),
                            "cv_mae_std": float(cv_mae_scores.std()),
                            "cv_r2_mean": float(cv_r2_scores.mean()),
                            "cv_r2_std": float(cv_r2_scores.std()),
                            "cv_folds": int(cv_folds)
                        }
                        st.caption(f"CV MAE: {(-cv_mae_scores.mean()):.2f} +/- {cv_mae_scores.std():.2f} hours")
                        st.caption(f"CV R2: {cv_r2_scores.mean():.3f} +/- {cv_r2_scores.std():.3f}")
                    except Exception:
                        pass

                    st.success("Model training complete.")
                except Exception as e:
                    st.error(f"Training Error: {e}")

    st.markdown("---")
    st.info(f"Current Model: **{model_choice}** | Mode: **{st.session_state['mode'].upper()}** | Test R2: **{st.session_state['train_r2']:.3f}**")
    st.caption("Scaling: RobustScaler for num_revisions; StandardScaler for other numeric features.")

    st.markdown("---")
    st.subheader("Model Diagnostics")
    if st.session_state.get('y_test') is None or st.session_state.get('y_preds') is None:
        st.info("Train a model to view diagnostics.")
    else:
        y_test = st.session_state['y_test']
        y_preds = st.session_state['y_preds']
        diag_df = df_modeling.loc[y_test.index].copy()

        st.caption("Stability Report (MAE by group)")
        group_cols = ['project_type', 'material_type', 'scope_category']
        stab = stability_report(y_test, y_preds, diag_df, group_cols)
        st.dataframe(stab, use_container_width=True, hide_index=True)

        st.caption("Top Residual Outliers")
        outliers = residual_outliers(y_test, y_preds, diag_df, top_n=10)
        show_cols = ['project_id', 'project_type', 'material_type', 'scope_category', 'abs_error']
        show_cols = [c for c in show_cols if c in outliers.columns]
        st.dataframe(outliers[show_cols], use_container_width=True, hide_index=True)

        if not outliers.empty:
            st.markdown("#### Outlier Distribution by Project")
            st.caption("Distribution of `total_project_hours` for each top residual outlier project context.")

            top_outliers = outliers.head(10)
            if 'total_project_hours' not in df_modeling.columns:
                st.info("Column `total_project_hours` is required to render this chart.")
            else:
                top_outliers = top_outliers.sort_values("abs_error", ascending=False)
                palette = plt.cm.tab10(np.linspace(0, 1, 10))
                pid_labels = []
                box_data = []
                abs_err_list = []

                for i, (idx, row) in enumerate(top_outliers.iterrows()):
                    pid = str(row.get("project_id", idx))
                    abs_err = float(row.get("abs_error", np.nan))

                    # Build per-project reference distribution from similar projects.
                    peer_mask = pd.Series(True, index=df_modeling.index)
                    if 'project_type' in df_modeling.columns and 'project_type' in row.index and pd.notna(row.get('project_type', np.nan)):
                        peer_mask = peer_mask & (df_modeling['project_type'] == row['project_type'])
                    if 'scope_category' in df_modeling.columns and 'scope_category' in row.index and pd.notna(row.get('scope_category', np.nan)):
                        peer_mask = peer_mask & (df_modeling['scope_category'] == row['scope_category'])

                    peer_vals = pd.to_numeric(df_modeling.loc[peer_mask, 'total_project_hours'], errors='coerce').dropna()
                    if len(peer_vals) < 8:
                        peer_vals = pd.to_numeric(df_modeling['total_project_hours'], errors='coerce').dropna()
                    if peer_vals.empty:
                        continue

                    pid_labels.append(pid)
                    box_data.append(peer_vals.values.astype(float))
                    abs_err_list.append(abs_err)

                if not box_data:
                    st.info("Unable to build box plot for outliers with current data.")
                else:
                    err_series = pd.to_numeric(top_outliers['abs_error'], errors='coerce').dropna()
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Outlier Projects Shown", f"{len(pid_labels)}")
                    if not err_series.empty:
                        k2.metric("Mean Abs Error", f"{err_series.mean():.1f} h")
                        k3.metric("Max Abs Error", f"{err_series.max():.1f} h")

                    # Plotly-based boxplot to match app graph theme behavior
                    fig_out = go.Figure()
                    box_colors = [
                        "#D81B60",  # magenta
                        "#1E88E5",  # blue
                        "#FFC107",  # amber
                        "#004D40",  # teal-dark
                        "#8E24AA",  # purple
                        "#43A047",  # green
                        "#F4511E",  # orange
                        "#3949AB",  # indigo
                        "#00ACC1",  # cyan
                        "#6D4C41",  # brown
                    ]

                    for i, (pid, vals) in enumerate(zip(pid_labels, box_data)):
                        color = box_colors[i % len(box_colors)]
                        fig_out.add_trace(
                            go.Box(
                                x=[pid] * len(vals),
                                y=vals,
                                name=pid,
                                width=0.78,
                                boxpoints="outliers",
                                marker=dict(color=color, size=5, opacity=0.9),
                                line=dict(color=color, width=2.8),
                                fillcolor=color,
                                opacity=0.55,
                                whiskerwidth=0.6,
                                quartilemethod="exclusive",
                                legendgroup=pid,
                                showlegend=True,
                                hovertemplate=(
                                    f"Project ID: {pid}"
                                    + "<br>total_project_hours: %{y:.1f}"
                                    + "<extra></extra>"
                                )
                            )
                        )

                    # Overlay the specific outlier project value as a highlighted red dot
                    for j, (idx, row) in enumerate(top_outliers.iterrows()):
                        if j >= len(pid_labels):
                            break
                        pid = pid_labels[j]
                        outlier_val = pd.to_numeric(pd.Series([row.get('total_project_hours', np.nan)]), errors='coerce').iloc[0]
                        if pd.notna(outlier_val):
                            fig_out.add_trace(
                                go.Scatter(
                                    x=[pid],
                                    y=[float(outlier_val)],
                                    mode="markers",
                                    marker=dict(symbol="diamond", size=10, color="#d62728", line=dict(color="white", width=1.1)),
                                    name="Selected top-residual project value",
                                    showlegend=(j == 0),
                                    legendgroup="selected_outlier",
                                    hovertemplate=(
                                        f"Project ID: {pid}"
                                        + "<br>Selected outlier total_project_hours: %{y:.1f}"
                                        + "<extra></extra>"
                                    )
                                )
                            )

                    fig_out.update_layout(
                        title={
                            "text": "Top 10 Outlier Projects - Box Plot",
                            "x": 0.01,
                            "xanchor": "left",
                            "y": 0.99,
                            "yanchor": "top"
                        },
                        font=dict(size=13),
                        xaxis_title="Project ID",
                        yaxis_title="total_project_hours",
                        xaxis_tickangle=-35,
                        width=1350,
                        height=900,
                        boxmode="group",
                        boxgap=0.02,
                        boxgroupgap=0.0,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.01,
                            xanchor="left",
                            x=0.0,
                            title="Legend"
                        ),
                        margin=dict(l=70, r=30, t=130, b=95),
                    )
                    fig_out.update_yaxes(tickformat=",")
                    st.caption("Key: hollow/isolated points on each box are statistical outliers (outside 1.5 x IQR whiskers). Red diamond marks the selected top-residual project value.")
                    st.plotly_chart(fig_out, use_container_width=False, key="outlier_boxplot_top10_plotly")

    st.markdown("---")
    st.subheader("Model Card Export")
    if st.session_state.get('last_train_metrics'):
        metrics = st.session_state['last_train_metrics']
        card_df = pd.DataFrame([{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_profile": profile,
            "model": metrics["model"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "rows": metrics["rows"],
            "features": metrics["features"]
        }])
        csv_buf = io.StringIO()
        card_df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Model Card (CSV)",
            data=csv_buf.getvalue(),
            file_name="model_card.csv",
            mime="text/csv"
        )
    else:
        st.info("Train a model to generate a model card.")

    st.markdown("---")
    st.subheader("Comprehensive Results Export")
    st.caption("Download model performance, explainability summaries, and related diagnostics as a ZIP package (high-quality PNG charts).")

    export_ready = (
        st.session_state.get('models') is not None
        and st.session_state.get('y_test') is not None
        and st.session_state.get('y_preds') is not None
    )

    if not export_ready:
        st.info("Train a model to enable full results export.")
    else:
        if st.button("Build Export Package (ZIP)", key="build_export_package"):
            try:
                zip_buffer = io.BytesIO()
                model_pipe_for_export = st.session_state['models']
                y_test_exp = pd.Series(st.session_state['y_test'])
                y_pred_exp = pd.Series(st.session_state['y_preds'], index=y_test_exp.index)
                X_test_exp = X_test.copy()
                if hasattr(model_pipe_for_export, "feature_names_in_"):
                    X_test_exp = X_test_exp.reindex(columns=model_pipe_for_export.feature_names_in_, fill_value=np.nan)

                def _write_df(zipf, name, df):
                    csv_buf = io.StringIO()
                    df.to_csv(csv_buf, index=True)
                    zipf.writestr(name, csv_buf.getvalue())

                def _write_fig(zipf, stem, fig):
                    png_buf = io.BytesIO()
                    fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
                    zipf.writestr(f"charts/{stem}.png", png_buf.getvalue())
                    plt.close(fig)

                with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    # Metrics tables
                    if st.session_state.get('last_train_metrics'):
                        _write_df(zf, "metrics/regression_metrics.csv", pd.DataFrame([st.session_state['last_train_metrics']]))
                    if st.session_state.get('last_cv_metrics'):
                        _write_df(zf, "metrics/cv_metrics.csv", pd.DataFrame([st.session_state['last_cv_metrics']]))
                    if st.session_state.get('last_baseline_mae') is not None:
                        _write_df(
                            zf,
                            "metrics/baseline_metrics.csv",
                            pd.DataFrame([{"baseline_mae_mean_predictor": float(st.session_state['last_baseline_mae'])}])
                        )
                    if st.session_state.get('tuning_history'):
                        _write_df(zf, "metrics/tuning_history.csv", pd.DataFrame(st.session_state['tuning_history']))
                    if st.session_state.get('backtest_results') is not None:
                        _write_df(zf, "metrics/backtest_results.csv", st.session_state['backtest_results'])
                    if st.session_state.get('clf_metrics') is not None:
                        _write_df(zf, "metrics/classification_metrics.csv", pd.DataFrame([st.session_state['clf_metrics']]))
                    if st.session_state.get('clf_confusion_matrix') is not None:
                        _write_df(zf, "metrics/classification_confusion_matrix.csv", st.session_state['clf_confusion_matrix'])

                    # Summary notes
                    summary_lines = [
                        "Daskan Results Export",
                        f"Generated: {datetime.now(timezone.utc).isoformat()}",
                        f"Rows in test set: {len(y_test_exp)}",
                        f"Model: {st.session_state.get('last_train_metrics', {}).get('model', 'unknown')}",
                    ]
                    zf.writestr("README_export.txt", "\n".join(summary_lines))

                    # Performance chart: Actual vs Predicted
                    fig1, ax1 = plt.subplots(figsize=(9, 6), facecolor="white")
                    ax1.scatter(y_test_exp.values, y_pred_exp.values, alpha=0.7, color="#1f77b4", edgecolor="none")
                    lim_max = float(max(y_test_exp.max(), y_pred_exp.max()) * 1.05)
                    ax1.plot([0, lim_max], [0, lim_max], linestyle="--", color="black", linewidth=1.2)
                    ax1.set_title("Actual vs Predicted (Test Set)")
                    ax1.set_xlabel("Actual Effort (Hours)")
                    ax1.set_ylabel("Predicted Effort (Hours)")
                    ax1.grid(alpha=0.25)
                    _write_fig(zf, "actual_vs_predicted", fig1)

                    # Residual histogram
                    residuals = (y_test_exp - y_pred_exp).values
                    fig2, ax2 = plt.subplots(figsize=(9, 6), facecolor="white")
                    ax2.hist(residuals, bins=25, color="#2ca02c", alpha=0.85)
                    ax2.set_title("Residual Distribution (Actual - Predicted)")
                    ax2.set_xlabel("Residual (Hours)")
                    ax2.set_ylabel("Count")
                    ax2.grid(alpha=0.25)
                    _write_fig(zf, "residual_distribution", fig2)

                    # Residuals vs predicted
                    fig3, ax3 = plt.subplots(figsize=(9, 6), facecolor="white")
                    ax3.scatter(y_pred_exp.values, residuals, alpha=0.7, color="#d62728", edgecolor="none")
                    ax3.axhline(0, linestyle="--", color="black", linewidth=1.2)
                    ax3.set_title("Residuals vs Predicted")
                    ax3.set_xlabel("Predicted Effort (Hours)")
                    ax3.set_ylabel("Residual (Hours)")
                    ax3.grid(alpha=0.25)
                    _write_fig(zf, "residuals_vs_predicted", fig3)

                    # Top residual outliers table + box plot by outlier project context
                    try:
                        diag_export_df = df_modeling.loc[y_test_exp.index].copy()
                        outliers_export = residual_outliers(y_test_exp, y_pred_exp, diag_export_df, top_n=10)
                        if not outliers_export.empty:
                            _write_df(zf, "metrics/top10_residual_outliers.csv", outliers_export)

                            if 'total_project_hours' in df_modeling.columns:
                                top_outliers_export = outliers_export.sort_values("abs_error", ascending=False).head(10)
                                pid_labels_exp = []
                                box_data_exp = []
                                selected_vals_exp = []

                                for idx, row in top_outliers_export.iterrows():
                                    pid = str(row.get("project_id", idx))

                                    peer_mask = pd.Series(True, index=df_modeling.index)
                                    if 'project_type' in df_modeling.columns and 'project_type' in row.index and pd.notna(row.get('project_type', np.nan)):
                                        peer_mask = peer_mask & (df_modeling['project_type'] == row['project_type'])
                                    if 'scope_category' in df_modeling.columns and 'scope_category' in row.index and pd.notna(row.get('scope_category', np.nan)):
                                        peer_mask = peer_mask & (df_modeling['scope_category'] == row['scope_category'])

                                    peer_vals = pd.to_numeric(df_modeling.loc[peer_mask, 'total_project_hours'], errors='coerce').dropna()
                                    if len(peer_vals) < 8:
                                        peer_vals = pd.to_numeric(df_modeling['total_project_hours'], errors='coerce').dropna()
                                    if peer_vals.empty:
                                        continue

                                    pid_labels_exp.append(pid)
                                    box_data_exp.append(peer_vals.values.astype(float))
                                    selected_vals_exp.append(float(pd.to_numeric(pd.Series([row.get('total_project_hours', np.nan)]), errors='coerce').iloc[0]))

                                if box_data_exp:
                                    boxplot_long = pd.DataFrame({
                                        "project_id": np.repeat(pid_labels_exp, [len(v) for v in box_data_exp]),
                                        "total_project_hours": np.concatenate(box_data_exp),
                                    })
                                    _write_df(zf, "metrics/outlier_boxplot_source_data.csv", boxplot_long)

                                    fig_out_exp, ax_out_exp = plt.subplots(figsize=(14, 8), facecolor="white")
                                    bp = ax_out_exp.boxplot(
                                        box_data_exp,
                                        tick_labels=pid_labels_exp,
                                        patch_artist=True,
                                        widths=0.72,
                                        showfliers=True,
                                    )
                                    palette_exp = plt.cm.tab10(np.linspace(0, 1, max(10, len(box_data_exp))))
                                    for i, box in enumerate(bp["boxes"]):
                                        box.set_facecolor(palette_exp[i % len(palette_exp)])
                                        box.set_alpha(0.65)
                                        box.set_linewidth(1.6)
                                    for whisk in bp["whiskers"]:
                                        whisk.set_linestyle("--")
                                        whisk.set_linewidth(1.1)
                                        whisk.set_color("#444444")
                                    for cap in bp["caps"]:
                                        cap.set_linewidth(1.2)
                                        cap.set_color("#444444")
                                    for med in bp["medians"]:
                                        med.set_linewidth(2.2)
                                        med.set_color("#1a1a1a")
                                    for fl in bp["fliers"]:
                                        fl.set_marker("o")
                                        fl.set_markersize(4)
                                        fl.set_markerfacecolor("none")
                                        fl.set_markeredgecolor("#2b2b2b")

                                    x_vals = np.arange(1, len(pid_labels_exp) + 1, dtype=float)
                                    ax_out_exp.scatter(
                                        x_vals,
                                        selected_vals_exp,
                                        marker="D",
                                        s=42,
                                        color="#d62728",
                                        edgecolors="white",
                                        linewidths=0.8,
                                        zorder=5,
                                        label="Selected top-residual project value",
                                    )
                                    ax_out_exp.set_title("Top 10 Outlier Projects - Box Plot", fontsize=14, pad=12)
                                    ax_out_exp.set_xlabel("Project ID")
                                    ax_out_exp.set_ylabel("total_project_hours")
                                    ax_out_exp.tick_params(axis='x', rotation=35)
                                    ax_out_exp.grid(axis="y", alpha=0.25)
                                    ax_out_exp.legend(loc="upper right", frameon=True)
                                    _write_fig(zf, "top10_outlier_projects_boxplot", fig_out_exp)
                    except Exception as e:
                        zf.writestr("metrics/outlier_export_error.txt", str(e))

                    # Permutation importance chart
                    try:
                        perm = permutation_importance(
                            model_pipe_for_export,
                            X_test_exp,
                            y_test_exp,
                            n_repeats=10,
                            random_state=42,
                            scoring='neg_mean_absolute_error'
                        )
                        imp_df = pd.DataFrame({
                            "Feature": X_test_exp.columns,
                            "Importance": perm.importances_mean
                        }).sort_values("Importance", ascending=False).head(15)
                        _write_df(zf, "metrics/permutation_importance.csv", imp_df.set_index("Feature"))
                        fig4, ax4 = plt.subplots(figsize=(10, 7), facecolor="white")
                        ax4.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color="#9467bd")
                        ax4.set_title("Top Permutation Importances")
                        ax4.set_xlabel("Importance (Mean)")
                        ax4.set_ylabel("Feature")
                        ax4.grid(axis="x", alpha=0.25)
                        _write_fig(zf, "permutation_importance_top15", fig4)
                    except Exception as e:
                        zf.writestr("metrics/permutation_importance_error.txt", str(e))

                    # SHAP mean absolute importance chart
                    try:
                        selected_num_export = get_selected_num_features(
                            NUM_FEATURES, LEAKY_NUM_FEATURES, is_metadata, include_leaky=include_leaky_features
                        )
                        selected_features_export = selected_num_export + CAT_FEATURES
                        selected_features_export = [c for c in selected_features_export if c in df_modeling.columns]
                        target_export = st.session_state.get('target_col', 'total_project_hours')
                        xai_df = df_modeling[selected_features_export + [target_export]].dropna()
                        if not xai_df.empty:
                            X_xai = xai_df[selected_features_export]
                            y_xai = xai_df[target_export]
                            X_train_xai, _, _, _, _, _, _ = split_regression_best_practice(
                                X_xai, y_xai, sample_weight=None, split_mode="auto", test_size=0.2, random_state=42
                            )
                            _, shap_values_exp, _, feature_names_exp = get_shap_data(model_pipe_for_export, X_train_xai)
                            sv = np.array(shap_values_exp)
                            if sv.ndim == 1:
                                sv = sv.reshape(-1, 1)
                            if sv.shape[1] == len(feature_names_exp):
                                mean_abs = np.mean(np.abs(sv), axis=0)
                                shap_df = pd.DataFrame({
                                    "Feature": feature_names_exp,
                                    "MeanAbsSHAP": mean_abs
                                }).sort_values("MeanAbsSHAP", ascending=False).head(20)
                                _write_df(zf, "metrics/shap_mean_abs.csv", shap_df.set_index("Feature"))
                                fig5, ax5 = plt.subplots(figsize=(10, 8), facecolor="white")
                                ax5.barh(shap_df["Feature"][::-1], shap_df["MeanAbsSHAP"][::-1], color="#17becf")
                                ax5.set_title("Top SHAP Mean Absolute Contributions")
                                ax5.set_xlabel("Mean |SHAP|")
                                ax5.set_ylabel("Feature")
                                ax5.grid(axis="x", alpha=0.25)
                                _write_fig(zf, "shap_mean_abs_top20", fig5)
                    except Exception as e:
                        zf.writestr("metrics/shap_export_error.txt", str(e))

                    # Classification confusion matrix heatmap if available
                    try:
                        cm_export = st.session_state.get('clf_confusion_matrix')
                        if cm_export is not None and not cm_export.empty:
                            fig6, ax6 = plt.subplots(figsize=(7, 6), facecolor="white")
                            im = ax6.imshow(cm_export.values, cmap="Blues")
                            ax6.set_xticks(np.arange(len(cm_export.columns)))
                            ax6.set_xticklabels(cm_export.columns, rotation=45, ha="right")
                            ax6.set_yticks(np.arange(len(cm_export.index)))
                            ax6.set_yticklabels(cm_export.index)
                            ax6.set_title("Classification Confusion Matrix")
                            for i in range(cm_export.shape[0]):
                                for j in range(cm_export.shape[1]):
                                    ax6.text(j, i, int(cm_export.values[i, j]), ha="center", va="center", color="black")
                            fig6.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
                            _write_fig(zf, "classification_confusion_matrix", fig6)
                    except Exception as e:
                        zf.writestr("metrics/classification_export_error.txt", str(e))

                st.download_button(
                    "Download Full Results Package (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"daskan_results_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    key="download_full_results_zip"
                )
                st.success("Export package built. Use the download button above.")
            except Exception as e:
                st.error(f"Export package error: {e}")
    
    st.markdown("---")
    st.subheader("Permutation Feature Importance (Current Model)")
    if st.session_state['models'] is None:
        st.info("Train a model to view feature importance.")
    else:
        model_pipe_for_imp = st.session_state['models']
        # Align feature columns to the trained model to avoid missing-column errors
        if hasattr(model_pipe_for_imp, "feature_names_in_"):
            X_imp = X_test.reindex(columns=model_pipe_for_imp.feature_names_in_, fill_value=np.nan)
        else:
            X_imp = X_test
        
        perm = permutation_importance(
            model_pipe_for_imp,
            X_imp,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring='neg_mean_absolute_error'
        )
        imp_df = pd.DataFrame({
            "Feature": X_imp.columns,
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

    st.markdown("---")
    st.subheader("Backtest & Model Sweep")
    st.caption("Runs a backtest across all models using only the imported metadata columns.")

    def _prepare_xy(df, features, target):
        data = df[features + [target]].dropna().copy()
        X_ = data[features]
        y_ = data[target]
        return X_, y_

    if st.button("Run Backtest & Test All Models", key="run_backtest_models", disabled=train_blocked):
        with st.spinner("Running backtest and model sweep..."):
            reg_registry = get_regression_registry()
            reg_keys = [k for k, v in reg_registry.items() if v["available"]]
            reg_labels = {k: reg_registry[k]["label"] for k in reg_keys}

            df_bt = df_modeling.copy()

            X_bt, y_bt = _prepare_xy(df_bt, available_features, target_col)
            if len(X_bt) < 10:
                st.warning("Not enough rows after filtering to run backtest.")
            else:
                splitter = KFold(n_splits=5, shuffle=True, random_state=42)

                results = []
                for key in reg_keys:
                    try:
                        # Backtest (CV) with untuned model
                        model = build_model(
                            "regression",
                            key,
                            selected_num_features,
                            CAT_FEATURES,
                            use_log_target=use_log_target,
                            n_rows=len(X_bt)
                        )
                        cv_r2 = cross_val_score(model, X_bt, y_bt, cv=splitter, scoring="r2")
                        cv_mae = -cross_val_score(model, X_bt, y_bt, cv=splitter, scoring="neg_mean_absolute_error")

                        # Holdout test with tuned model
                        X_train_bt, X_test_bt, y_train_bt, y_test_bt, _, _, _ = split_regression_best_practice(
                            X_bt, y_bt, sample_weight=None, split_mode="auto", test_size=0.2, random_state=42
                        )

                        tuned_model, _ = train_and_tune_model(
                            X_train_bt, y_train_bt, key,
                            selected_num_features, CAT_FEATURES, use_log_target,
                            sample_weight=None
                        )
                        preds_bt = tuned_model.predict(X_test_bt)
                        hold_r2 = r2_score(y_test_bt, preds_bt)
                        hold_mae = mean_absolute_error(y_test_bt, preds_bt)

                        results.append({
                            "Model": reg_labels.get(key, key),
                            "CV R2 (mean)": float(np.mean(cv_r2)),
                            "CV R2 (std)": float(np.std(cv_r2)),
                            "CV MAE (mean)": float(np.mean(cv_mae)),
                            "Holdout R2": float(hold_r2),
                            "Holdout MAE": float(hold_mae),
                            "Rows Used": int(len(X_bt))
                        })
                    except Exception as e:
                        results.append({
                            "Model": reg_labels.get(key, key),
                            "CV R2 (mean)": np.nan,
                            "CV R2 (std)": np.nan,
                            "CV MAE (mean)": np.nan,
                            "Holdout R2": np.nan,
                            "Holdout MAE": np.nan,
                            "Rows Used": int(len(X_bt)),
                            "Error": str(e)
                        })

                results_df = pd.DataFrame(results)
                st.session_state["backtest_results"] = results_df.sort_values(
                    ["Holdout R2", "CV R2 (mean)"], ascending=[False, False]
                )

    if st.session_state.get("backtest_results") is not None:
        st.dataframe(
            st.session_state["backtest_results"],
            use_container_width=True,
            hide_index=True
        )
    
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
        
        if current_r2 > r2_threshold:
            model_status = "Ready for Production"
            button_label = f"Approve Model for Production Use (R2 > {r2_threshold:.3f})"
            is_safe = True
        else:
            model_status = "Requires Further Tuning"
            button_label = f"Review Model Performance (R2 < {r2_threshold:.3f})"
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

        st.markdown("---")
        st.subheader("Iterative Retuning and History")
        history_df = pd.DataFrame(st.session_state.get('tuning_history', []))
        if history_df.empty:
            st.info("No tuning attempts recorded yet.")
        else:
            display_cols = [
                "attempt", "timestamp_utc", "trigger", "model", "split_mode",
                "train_rows", "test_rows", "cv_folds", "cv_r2_best_or_mean",
                "test_r2", "test_mae", "outliers_removed", "seed", "n_iter_scale",
                "threshold_passed"
            ]
            display_cols = [c for c in display_cols if c in history_df.columns]
            st.dataframe(
                history_df[display_cols].sort_values("attempt", ascending=False),
                use_container_width=True,
                hide_index=True
            )

        col_hist_1, col_hist_2 = st.columns([2, 1])
        if col_hist_2.button("Clear Tuning History", key="clear_tuning_history"):
            st.session_state['tuning_history'] = []
            st.session_state['tuning_attempt_counter'] = 0
            st.success("Tuning history cleared.")

        if current_r2 < r2_threshold and not train_blocked:
            st.caption("Model is below threshold. Run additional tuning attempts and track each result.")
            col_rt_1, col_rt_2, col_rt_3 = st.columns([1.2, 1.2, 1.4])
            retune_iter_scale = col_rt_1.slider(
                "Retune Search Intensity",
                min_value=1.0,
                max_value=3.0,
                value=1.0,
                step=0.25,
                key="retune_iter_scale",
                help="Scales randomized-search iteration budget for tunable models."
            )
            retune_resplit = col_rt_2.checkbox(
                "Reshuffle Train/Test Split",
                value=True,
                key="retune_resplit",
                help="Uses a new split seed for each retune attempt."
            )
            retune_outlier = col_rt_3.checkbox(
                "Apply Outlier Removal in Retune",
                value=remove_outliers,
                key="retune_use_outlier"
            )

            if st.button("Run Additional Tuning Attempt", key="retune_attempt_btn", type="secondary"):
                try:
                    st.session_state['quote_generated'] = False
                    retune_seed = int(st.session_state.get('tuning_attempt_counter', 0)) + 43
                    split_seed = retune_seed if retune_resplit else 42

                    # Rebuild split for retune attempt
                    sample_weight_rt = np.where(y > y.quantile(0.75), 1.5, 1.0)
                    X_train_rt, X_test_rt, y_train_rt, y_test_rt, w_train_rt, _, split_mode_rt = split_regression_best_practice(
                        X, y, sample_weight=sample_weight_rt, split_mode=requested_split_mode, test_size=0.2, random_state=split_seed
                    )

                    X_train_use_rt, y_train_use_rt = X_train_rt, y_train_rt
                    removed_outliers_rt = 0
                    if retune_outlier:
                        base_model_rt, _ = train_and_tune_model(
                            X_train_rt, y_train_rt, model_choice,
                            selected_num_features, CAT_FEATURES, use_log_target,
                            sample_weight=w_train_rt,
                            random_state=retune_seed,
                            n_iter_scale=retune_iter_scale
                        )
                        resid_rt = (y_train_rt - base_model_rt.predict(X_train_rt)).abs()
                        drop_idx_rt = resid_rt.sort_values(ascending=False).head(outlier_count).index
                        X_train_use_rt = X_train_rt.drop(index=drop_idx_rt)
                        y_train_use_rt = y_train_rt.drop(index=drop_idx_rt)
                        removed_outliers_rt = int(len(drop_idx_rt))

                    tuned_model_rt, cv_score_rt = train_and_tune_model(
                        X_train_use_rt, y_train_use_rt, model_choice,
                        selected_num_features, CAT_FEATURES, use_log_target,
                        sample_weight=None,
                        random_state=retune_seed,
                        n_iter_scale=retune_iter_scale
                    )
                    preds_rt = tuned_model_rt.predict(X_test_rt)
                    mae_rt = mean_absolute_error(y_test_rt, preds_rt)
                    r2_rt = r2_score(y_test_rt, preds_rt)

                    # Update active model/session outputs
                    st.session_state['models'] = tuned_model_rt
                    st.session_state['mode'] = 'point'
                    st.session_state['train_r2'] = float(r2_rt)
                    st.session_state['y_test'] = y_test_rt
                    st.session_state['y_preds'] = preds_rt
                    st.session_state['last_train_metrics'] = {
                        "model": model_choice,
                        "mae": float(mae_rt),
                        "r2": float(r2_rt),
                        "rows": int(len(X_train_rt) + len(X_test_rt)),
                        "features": ", ".join(selected_features)
                    }
                    st.session_state['last_train_features'] = selected_features

                    _append_tuning_history(
                        trigger="retune",
                        model_name=model_choice,
                        split_used=split_mode_rt,
                        cv_score=cv_score_rt,
                        test_r2_val=r2_rt,
                        test_mae_val=mae_rt,
                        train_rows_used=len(X_train_use_rt),
                        test_rows_used=len(X_test_rt),
                        cv_fold_used=(5 if len(df_modeling) < 500 else 7),
                        outliers_removed=removed_outliers_rt,
                        seed_used=retune_seed,
                        n_iter_scale_used=retune_iter_scale
                    )

                    if r2_rt >= r2_threshold:
                        st.success(f"Retune succeeded. Test R2 improved to {r2_rt:.3f} and passed threshold {r2_threshold:.3f}.")
                    else:
                        st.warning(f"Retune complete. Test R2 is {r2_rt:.3f}; still below threshold {r2_threshold:.3f}.")
                except Exception as e:
                    st.error(f"Retune error: {e}")

# ----------------------------------------------------------------------
# TAB 3: MODEL EXPLAINABILITY (XAI)
# ----------------------------------------------------------------------
with tabs[2]:
    st.header("Explainability (Permutation + SHAP)")

    if st.session_state['models'] is None:
        st.warning("Please train a model in the 'AI Model Engine' tab first.")
    else:
        # 1. Setup Data and Explainer
        selected_num_features = get_selected_num_features(NUM_FEATURES, LEAKY_NUM_FEATURES, is_metadata, include_leaky=include_leaky_features)
        selected_features = selected_num_features + CAT_FEATURES
        missing_features = [c for c in selected_features if c not in df_modeling.columns]
        if missing_features:
            st.warning("Explainability unavailable due to missing training features:")
            st.write(", ".join(missing_features))
            st.stop()
        X = df_modeling[selected_features].copy()
        target_col = st.session_state.get('target_col', 'total_project_hours')
        y_for_xai = df_modeling[target_col].copy()
        valid_mask_xai = y_for_xai.notna()
        X = X.loc[valid_mask_xai]
        y_for_xai = y_for_xai.loc[valid_mask_xai]
        X_train, X_test, y_train, y_test, _, _, _ = split_regression_best_practice(
            X, y_for_xai, sample_weight=None, split_mode="auto", test_size=0.2, random_state=42
        )

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
                    st.warning("ALE could not be computed for this feature (insufficient variation or data).")
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
    st.header("Complexity Classification Models")
    if 'project_complexity_class' not in df_modeling.columns:
        # Compute on the fly (rule-based) for metadata-only datasets
        def _compute_complexity_class_inline(df):
            dfc = df.copy()
            duration_col = None
            for col in ['expected_duration_days', 'actual_duration_days', 'project_duration_days']:
                if col in dfc.columns:
                    duration_col = col
                    break

            area_q1 = dfc['surface_area_m2'].dropna().quantile(0.25) if 'surface_area_m2' in dfc.columns else np.nan
            area_q3 = dfc['surface_area_m2'].dropna().quantile(0.75) if 'surface_area_m2' in dfc.columns else np.nan
            dur_q1 = dfc[duration_col].dropna().quantile(0.25) if duration_col else np.nan
            dur_q3 = dfc[duration_col].dropna().quantile(0.75) if duration_col else np.nan

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

            dfc['project_complexity_class'] = dfc.apply(_classify, axis=1)
            return dfc

        df_modeling = _compute_complexity_class_inline(df_modeling)

    class_df = df_modeling[available_features + ['project_complexity_class']].dropna(subset=['project_complexity_class']).copy()
    Xc = class_df[available_features]
    yc = class_df['project_complexity_class']
    cls_counts = yc.value_counts(dropna=True)
    if not cls_counts.empty:
        st.caption("Class distribution: " + ", ".join([f"{k}={int(v)}" for k, v in cls_counts.to_dict().items()]))
        if int(cls_counts.min()) < 5:
            st.warning("Some classes have very low counts; Gradient Boosting tuning may be unstable.")
    if yc.dropna().shape[0] < 10:
        st.warning("Not enough labeled rows to train classification.")
    else:
        stratify_cls = yc if yc.nunique(dropna=True) > 1 else None
        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            Xc, yc, test_size=0.2, random_state=42, stratify=stratify_cls
        )
        clf_choice = st.selectbox(
            "Classifier",
            ["Logistic Regression", "Gradient Boosting Classifier"],
            key="clf_choice"
        )
        if st.button("Train Classifier", key="train_clf"):
            try:
                st.session_state['clf_last_error'] = None
                with st.spinner("Training classifier..."):
                    high_weight = 2.0
                    class_weights = np.where(yc_train == "High", high_weight, 1.0)
                    class_cat_features = [c for c in CAT_FEATURES if c in Xc_train.columns]
                    class_num_features = [c for c in Xc_train.columns if c not in class_cat_features]
                    clf_pipe = train_classifier(
                        Xc_train,
                        yc_train,
                        clf_choice,
                        class_num_features,
                        class_cat_features,
                        sample_weight=class_weights
                    )
                    preds = clf_pipe.predict(Xc_test)
                    acc = accuracy_score(yc_test, preds)
                    precision = precision_score(yc_test, preds, average="macro", zero_division=0)
                    recall_high = recall_score(yc_test, preds, labels=["High"], average="macro", zero_division=0)
                    f1 = f1_score(yc_test, preds, average="macro")

                    st.session_state['clf_model'] = clf_pipe
                    st.session_state['clf_choice_model'] = clf_choice

                    # Confusion matrix + false positive rate for High class
                    labels = ["Low", "Medium", "High"]
                    cm = confusion_matrix(yc_test, preds, labels=labels)
                    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

                    high_idx = labels.index("High")
                    fp_high = cm[:, high_idx].sum() - cm[high_idx, high_idx]
                    tn_high = cm.sum() - cm[:, high_idx].sum() - cm[high_idx, :].sum() + cm[high_idx, high_idx]
                    fpr_high = fp_high / max(1, (fp_high + tn_high))

                    st.session_state['clf_metrics'] = {
                        "accuracy": float(acc),
                        "precision_macro": float(precision),
                        "recall_high": float(recall_high),
                        "f1_macro": float(f1),
                        "fpr_high": float(fpr_high)
                    }
                    st.session_state['clf_confusion_matrix'] = cm_df.copy()
                st.success("Classification training complete.")
            except Exception as e:
                st.session_state['clf_last_error'] = str(e)
                st.error(f"Classification Error: {e}")

        # Persisted results display (shows even after reruns)
        clf_metrics = st.session_state.get('clf_metrics')
        clf_cm = st.session_state.get('clf_confusion_matrix')
        if clf_metrics is not None and st.session_state.get('clf_choice_model') == clf_choice:
            st.metric("Accuracy", f"{clf_metrics['accuracy']:.3f}")
            st.metric("Precision (Macro)", f"{clf_metrics['precision_macro']:.3f}")
            st.metric("Recall (High)", f"{clf_metrics['recall_high']:.3f}")
            st.metric("F1 (Macro)", f"{clf_metrics['f1_macro']:.3f}")
            if clf_cm is not None:
                st.caption("Confusion Matrix")
                st.dataframe(clf_cm, use_container_width=True)
            fpr_high = float(clf_metrics.get('fpr_high', np.nan))
            if pd.notna(fpr_high):
                if fpr_high < 0.30:
                    st.success(f"False Positive Rate (High): {fpr_high:.2f} (< 0.30)")
                else:
                    st.warning(f"False Positive Rate (High): {fpr_high:.2f} (>= 0.30)")
            st.info("Staffing guidance: assign senior engineers to projects classified as High complexity to reduce risk and avoid schedule overruns.")
        elif st.session_state.get('clf_last_error'):
            st.error(f"Last Classification Error: {st.session_state['clf_last_error']}")
        else:
            st.caption("Click 'Train Classifier' to generate classification metrics.")

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
    st.header("Project Quotation & Resource Planner")
    
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

        c8, c9 = st.columns(2)
        i_duration = c8.number_input("Project Duration (days)", 1, 3650, 120, key='i_duration')
        i_revisions = c9.number_input("Revisions (count)", 0, 50, 2, key='i_revisions')
        
        st.markdown("---")
        st.subheader("2. Financial & Risk Settings")
        col_res_1, col_res_2, col_res_3, col_res_4 = st.columns(4)
        avg_hourly_rate = col_res_1.number_input("Avg. Cost Rate ($/h)", 80, 200, AVG_HOURLY_RATE_CAD, key='avg_hourly_rate')
        profit_markup = col_res_2.slider("Profit/Markup (%)", 5, 50, 25) / 100
        hours_per_week_slider = col_res_3.number_input("Hours per Engineer/Week", 20, 40, HOURS_PER_WEEK, key='hours_per_week')
        _ = col_res_4.empty()

        st.markdown("---")
        st.subheader("Civil Engineering Reasoning")
        st.caption(
            "Primary drivers: scope category, material type, surface area, levels/height, project duration, and revisions. "
            "These reflect structural scale, complexity of load paths/material behavior, and coordination effort."
        )

        # Derived features
        floor_area_ratio = (i_area / i_levels) if i_levels > 0 else np.nan
        actual_duration_days = i_duration

        input_payload = {
            'surface_area_m2': i_area,
            'building_height_m': i_height,
            'num_levels': i_levels,
            'floor_area_ratio': floor_area_ratio,
            'num_revisions': i_revisions,
            'actual_duration_days': actual_duration_days,
            'num_units': i_units,
            'area_per_unit': (i_area / (i_units + 1)),
            'height_per_level': (i_height / (i_levels + 1)),
            'complexity_interaction_index': (i_area * i_levels),
            'revision_intensity': (i_revisions / (actual_duration_days + 1)),
            'material_type': i_mat,
            'scope_category': i_scope,
            'project_type': i_type
        }
        selected_num_features = get_selected_num_features(NUM_FEATURES, LEAKY_NUM_FEATURES, is_metadata, include_leaky=True)
        selected_features = selected_num_features + CAT_FEATURES
        input_data = pd.DataFrame([{col: input_payload.get(col, np.nan) for col in selected_features}])
        
        # Prediction Button (This updates the session state)
        if st.button("RUN AI ESTIMATION", type="primary", use_container_width=True):
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
            st.subheader("Input Data Drift & Reliability Check")
            
            train_stats = st.session_state['train_stats']
            is_drifting = False
            drift_message = ""
            
            # Check numerical features against training distribution
            for feature in selected_num_features:
                if feature not in train_stats.index:
                    continue
                input_val = input_data[feature].iloc[0]
                mean_val = train_stats.loc[feature, 'mean']
                std_val = train_stats.loc[feature, 'std']
                
                # Simple drift detection: > 2 standard deviations away from the mean
                if pd.notna(input_val) and std_val > 0 and abs(input_val - mean_val) > 2 * std_val:
                    is_drifting = True
                    drift_message += f"- **{feature}** value ({input_val:.2f}) is significantly outside the historical range (Mean: {mean_val:.2f}, Std Dev: {std_val:.2f}).\n"
            
            if is_drifting:
                st.error("MODEL RELIABILITY WARNING: The input project significantly deviates from the training data distribution in the following areas. Predictions may be unreliable.")
                st.markdown(drift_message)
            else:
                st.success("Data Quality Check: Input parameters are well within the historical data distribution. Model predictions are expected to be reliable.")
            st.markdown("---")

            
            # 1. Summary & Financials
            st.markdown("#### Project Summary & Financials")
            # Structural red-flag check
            if 'total_project_hours' in df_modeling.columns:
                high_hours_threshold = df_modeling['total_project_hours'].quantile(0.8)
                if est_hours > high_hours_threshold and i_revisions < 3:
                    st.warning("Warning: High structural complexity with low revision history—check for incomplete design data.")
            
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            
            col_p1.metric("Estimated Effort (h)", f"{pred_val:.0f} Hours", f"Risk: +/- {risk_margin:.0f}h" if risk_margin > 0 else "Point Estimate")
            col_p2.metric("Base Cost (CAD)", f"${estimated_cost_base:,.0f}")
            col_p3.metric("Competitive Quote", f"${competitive_quote:,.0f}", f"{int(profit_markup*100)}% markup")
            col_p4.metric("Conservative Quote", f"${conservative_quote:,.0f}", "Risk + Profit", delta_color="normal")
            
            
            # 2. Resource Planning (Reactive to Slider)
            st.markdown("#### Resource & Timeline Planning")
            
            team_size = st.slider("Select Planned Team Size (Engineers) for Schedule", 1, 10, 2, key='team_size_live')
            
            weeks_needed_1_eng = pred_val / hours_per_week_slider
            real_duration = weeks_needed_1_eng / team_size
            
            c_res_1, c_res_2, c_res_3 = st.columns(3)
            
            c_res_1.metric("Duration (1 Engineer)", f"{weeks_needed_1_eng:.1f} Weeks")
            c_res_2.metric(f"Duration ({team_size} Engineers)", f"{real_duration:.1f} Weeks", delta="Project Duration")

            # Risk Flag
            high_risk = (i_revisions > 5)
            if real_duration > 15 or risk_margin > 50 or is_drifting or high_risk:
                st.error("High Risk Alert: Long duration, high uncertainty, data drift, or high revisions detected. Increase team size or pad the quote.")
            else:
                st.success("Standard Project: Project fits typical historical parameters.")


            st.markdown("---")
            
            # 3. Task Breakdown & Gantt
            st.markdown("#### Recommended Task Breakdown & Schedule")
            
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
    st.header("Clustering & Personas")
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

            cluster_labels = cluster_df.groupby('cluster', group_keys=False).apply(label_cluster, include_groups=False)
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
