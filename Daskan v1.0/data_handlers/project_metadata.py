import pandas as pd
from .common import normalize_aliases, _detect_data_profile


def load_project_metadata(raw_df):
    """Handle project metadata dataset."""
    df = raw_df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = normalize_aliases(df)
    profile = _detect_data_profile(df)
    if profile != "metadata":
        return None, None, profile, None

    # Coerce types without creating new engineered columns
    num_cols = [
        'is_winter',
        'surface_area_m2',
        'num_levels',
        'floor_area_ratio',
        'actual_duration_days',
        'total_project_hours',
        'design_hours_total',
        'num_revisions'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_modeling = df
    df_analytics = df.copy()
    data_source = "Uploaded Project Metadata (No Feature Engineering)"
    return df_modeling, df_analytics, profile, data_source
