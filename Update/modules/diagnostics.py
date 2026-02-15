import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def detect_leakage(feature_list, target_col, allow_leakage=False):
    if allow_leakage:
        return []
    leak_map = {
        "total_project_hours": [
            "actual_duration_days",
            "actual_duration_d",
            "actual_duration_days",
            "corrected_end_date",
            "planned_end_date",
            "corrected_end_ordinal",
            "planned_end_ordinal"
        ]
    }
    leak_cols = set(leak_map.get(target_col, []))
    offenders = [c for c in feature_list if c in leak_cols]
    return offenders


def select_features(df, num_features, cat_features, corr_threshold=0.98):
    kept_num = []
    dropped = []

    # Drop low-variance numeric features
    for col in num_features:
        if col not in df.columns:
            continue
        if df[col].nunique(dropna=True) <= 1:
            dropped.append((col, "low_variance"))
        else:
            kept_num.append(col)

    # Drop highly correlated numeric features (skip on very small datasets)
    if kept_num and len(df) >= 50:
        corr = df[kept_num].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        for col in upper.columns:
            if any(upper[col] > corr_threshold):
                dropped.append((col, "high_correlation"))
                if col in kept_num:
                    kept_num.remove(col)

    kept_cat = [c for c in cat_features if c in df.columns]
    return kept_num, kept_cat, dropped


def stability_report(y_true, y_pred, df, group_cols):
    rows = []
    y_true = pd.Series(y_true, index=df.index)
    y_pred = pd.Series(y_pred, index=df.index)
    for col in group_cols:
        if col not in df.columns:
            continue
        for val, idx in df.groupby(col).groups.items():
            y_t = y_true.loc[idx]
            y_p = y_pred.loc[idx]
            if len(y_t) < 5:
                continue
            rows.append({
                "group": col,
                "value": str(val),
                "count": len(y_t),
                "mae": mean_absolute_error(y_t, y_p)
            })
    if not rows:
        return pd.DataFrame(columns=["group", "value", "count", "mae"])
    return pd.DataFrame(rows).sort_values(["group", "mae"], ascending=[True, False])


def residual_outliers(y_true, y_pred, df, top_n=10):
    resid = (y_true - y_pred).abs()
    out = df.copy()
    out["abs_error"] = resid
    out = out.sort_values("abs_error", ascending=False).head(top_n)
    return out
