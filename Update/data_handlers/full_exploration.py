import pandas as pd
from .common import feature_engineer_data, normalize_aliases, _detect_data_profile


def load_full_exploration(raw_df):
    """Handle full exploration (granular) dataset."""
    df = raw_df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = normalize_aliases(df)
    profile = _detect_data_profile(df)
    if profile != "granular":
        return None, None, profile, None

    df_modeling = feature_engineer_data(df)
    df_analytics = df
    data_source = "Uploaded Full Exploration Data (Granular + Metadata)"
    return df_modeling, df_analytics, profile, data_source
