import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_granular_synthetic_data(num_projects=100, seed=42):
    """Generates synthetic granular timesheet data."""
    np.random.seed(seed)

    project_types = ['Residential', 'Commercial', 'Institutional', 'Industrial']
    materials = ['Wood', 'Steel', 'Concrete', 'Mixed']
    projects = []
    project_effort_map = {}

    for i in range(num_projects):
        p_id = f"P-2022-{str(i + 1).zfill(3)}"
        p_type = np.random.choice(project_types, p=[0.25, 0.35, 0.15, 0.25])

        if p_type == 'Residential':
            area = np.random.randint(100, 600)
            levels = np.random.randint(1, 4)
        elif p_type == 'Commercial':
            area = np.random.randint(500, 2000)
            levels = np.random.randint(2, 8)
        else:
            area = np.random.randint(1000, 5000)
            levels = np.random.randint(1, 15)

        start_month = np.random.randint(1, 12)
        start_date = datetime(2022, start_month, 1) + timedelta(days=np.random.randint(0, 28))

        # Calc Effort (Hidden Truth)
        base_effort = (area * 0.1) + (levels * 20)
        if p_type in ['Institutional', 'Industrial']:
            base_effort *= 1.5
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
            if hours <= 0.25:
                hours = 0.5
            random_days = np.random.randint(0, max(1, date_range_days))
            log_date = proj['start_date'] + timedelta(days=random_days)

            timesheet_entries.append({
                'log_id': log_id_counter,
                'project_id': p_id,
                'employee_id': f"EMP-{np.random.randint(1, 12)}",
                'date_logged': log_date,
                'task_category': np.random.choice(
                    ['Design', 'Calculation', 'Drafting', 'Meeting', 'Site Visit'],
                    p=[0.25, 0.25, 0.3, 0.1, 0.1]
                ),
                'hours_worked': hours
            })
            log_id_counter += 1

    df_timesheets = pd.DataFrame(timesheet_entries)

    # Merge
    df_master = pd.merge(df_timesheets, df_projects, on='project_id', how='left')
    df_master = _parse_date_column(df_master, 'date_logged')
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


def label_data_source(profile):
    if profile == "granular":
        return "Uploaded Full Exploration Data (Granular + Metadata)"
    if profile == "metadata":
        return "Uploaded Project Metadata"
    return "Uploaded Data (Unknown Schema)"


def normalize_aliases(df):
    df = df.copy()
    if 'is_winter' not in df.columns and 'is_winter_day' in df.columns:
        df = df.rename(columns={'is_winter_day': 'is_winter'})
    if 'actual_duration_days' not in df.columns and 'actual_duration_d' in df.columns:
        df = df.rename(columns={'actual_duration_d': 'actual_duration_days'})
    if 'total_project_hours' not in df.columns and 'total_project_effort' in df.columns:
        df = df.rename(columns={'total_project_effort': 'total_project_hours'})
    return df


def get_selected_num_features(num_features, leaky_features, is_metadata, include_leaky=False):
    features = list(num_features)
    if is_metadata and 'design_hours_total' in features and not include_leaky:
        features.remove('design_hours_total')
    if include_leaky:
        features += leaky_features
    return features


def feature_engineer_data(df_analytics):
    """Aggregates granular data to project level and creates complex features."""
    df_analytics = df_analytics.copy()
    df_analytics = normalize_aliases(df_analytics)

    # Safety Check: Ensure Date Column is datetime (only if present)
    df_analytics = _parse_date_column(df_analytics, 'date_logged')

    if 'log_id' not in df_analytics.columns:
        # Already summary data
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
            'num_site_visits',
            'actual_duration_days',
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
        meta_cols = [c for c in meta_candidates if c in df_analytics.columns]

        df_modeling = df_analytics.groupby('project_id', as_index=False).agg(
            {
                'hours_worked': 'sum',
                'date_logged': 'count',
                **{c: 'first' for c in meta_cols if c not in ['hours_worked', 'date_logged']}
            }
        )
        df_modeling = df_modeling.rename(columns={'hours_worked': 'total_project_hours', 'date_logged': 'num_logs'})

    df_modeling = normalize_aliases(df_modeling)

    # Coerce numeric columns
    num_cols = [
        'surface_area_m2', 'num_levels', 'num_units', 'building_height_m', 'floor_area_ratio',
        'expected_duration_days', 'actual_duration_days',
        'num_site_visits', 'total_project_hours', 'design_hours_total', 'avg_hours_per_employee',
        'num_revisions', 'is_winter'
    ]
    for col in num_cols:
        if col in df_modeling.columns:
            df_modeling[col] = pd.to_numeric(df_modeling[col], errors='coerce')

    # Fill missing revisions
    if 'num_revisions' in df_modeling.columns:
        df_modeling['num_revisions'] = df_modeling['num_revisions'].fillna(0)

    # Feature engineering
    if 'building_height_m' in df_modeling.columns and 'surface_area_m2' in df_modeling.columns:
        df_modeling['height_to_area_ratio'] = df_modeling['building_height_m'] / df_modeling['surface_area_m2'].replace(0, np.nan)

    if 'surface_area_m2' in df_modeling.columns and 'num_levels' in df_modeling.columns:
        df_modeling['area_per_level'] = df_modeling['surface_area_m2'] / df_modeling['num_levels'].replace(0, np.nan)

    if 'num_levels' in df_modeling.columns and 'building_height_m' in df_modeling.columns:
        corr = df_modeling[['num_levels', 'building_height_m']].corr().iloc[0, 1]
        if pd.notna(corr) and corr > 0.9:
            df_modeling['avg_floor_height'] = df_modeling['building_height_m'] / df_modeling['num_levels'].replace(0, np.nan)
        else:
            df_modeling['avg_floor_height'] = np.nan

    # Calculate Advanced Complexity Index (if required inputs exist)
    if 'complexity_index' not in df_modeling.columns:
        def _complexity_idx(row):
            try:
                mat = str(row.get('material_type', ''))
                mat_factor = {'Mixed': 2.0, 'Steel': 1.5, 'Concrete': 1.2}.get(mat, 1.0)
                levels = row.get('num_levels', np.nan)
                area = row.get('surface_area_m2', np.nan)
                if pd.isna(levels) or pd.isna(area) or levels <= 0:
                    return np.nan
                c_idx = (levels ** 1.2) * (area / levels) / 500 * mat_factor
                return max(0.5, c_idx)
            except Exception:
                return np.nan
        df_modeling['complexity_index'] = df_modeling.apply(_complexity_idx, axis=1)

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

    # Derive a classification target from rule-based complexity logic (no effort leakage)
    if 'project_complexity_class' not in df_modeling.columns:
        duration_col = None
        for col in ['expected_duration_days', 'actual_duration_days']:
            if col in df_modeling.columns:
                duration_col = col
                break

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

    return df_modeling
