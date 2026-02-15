import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV


@st.cache_data(show_spinner="Calculating SHAP values...")
def get_shap_data(_models, X_train):
    """Caches the expensive SHAP calculation."""
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

    if isinstance(final_estimator, (LinearRegression, Ridge, RidgeCV)):
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

    if isinstance(final_estimator, (LinearRegression, Ridge, RidgeCV)):
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

    num_bins = 20
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    bin_edges = np.unique(np.quantile(x, quantiles))
    if len(bin_edges) < 3:
        return None

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

        preds_high = model_pipe.predict(X_high)
        preds_low = model_pipe.predict(X_low)
        effects.append(np.mean(preds_high - preds_low))
        centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)

    if not effects:
        return None

    ale_vals = np.cumsum(effects) - np.mean(np.cumsum(effects))
    return pd.DataFrame({feature_name: centers, "eff": ale_vals})
