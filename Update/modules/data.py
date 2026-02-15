import pandas as pd

from data_handlers.common import normalize_aliases, label_data_source, _detect_data_profile, feature_engineer_data
from data_handlers.full_exploration import load_full_exploration
from data_handlers.project_metadata import load_project_metadata


def load_uploaded_csv(uploaded_file):
    # Auto-detect delimiter (comma vs semicolon) and normalize headers
    raw_df = pd.read_csv(uploaded_file, sep=None, engine="python")
    raw_df.columns = [c.strip() for c in raw_df.columns]
    raw_df = normalize_aliases(raw_df)
    profile = _detect_data_profile(raw_df)

    if profile == "granular":
        df_modeling, df_analytics, _, data_source = load_full_exploration(raw_df)
    elif profile == "metadata":
        df_modeling, df_analytics, _, data_source = load_project_metadata(raw_df)
    else:
        df_analytics = raw_df.copy()
        df_modeling = feature_engineer_data(raw_df)
        data_source = label_data_source(profile)

    return df_modeling, df_analytics, profile, data_source


def compute_train_stats(df_modeling, num_cols):
    num_cols = [c for c in num_cols if c in df_modeling.columns]
    if not num_cols:
        return pd.DataFrame(columns=['mean', 'std'])
    return df_modeling[num_cols].agg(['mean', 'std']).T
