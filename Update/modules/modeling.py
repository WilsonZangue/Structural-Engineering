import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    ExtraTreesRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import r2_score, recall_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, cross_val_score

# Optional regressors
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    _HAS_CAT = True
except Exception:
    _HAS_CAT = False

def get_regression_registry():
    return {
        "ridge": {"label": "Ridge Regression", "available": True},
        "gbr": {"label": "Gradient Boosting Regressor (sklearn)", "available": True},
        "rf": {"label": "Random Forest (Extra Trees)", "available": True},
        "linear": {"label": "Linear Regression", "available": True},
    }


def build_model(task, model_choice, num_features, cat_features, use_log_target=True, n_rows=None):
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
        # Data-aware defaults (light tuning based on dataset size)
        n_rows = int(n_rows) if n_rows is not None else None
        small_data = n_rows is not None and n_rows < 500

        if model_choice == "linear":
            base_model = LinearRegression()
        elif model_choice == "ridge":
            base_model = Ridge(alpha=1.0, random_state=42)
        elif model_choice == "rf":
            base_model = ExtraTreesRegressor(
                n_estimators=250 if small_data else 800,
                max_depth=5 if small_data else 24,
                min_samples_split=6 if small_data else 3,
                min_samples_leaf=4 if small_data else 1,
                max_features='sqrt',
                random_state=42
            )
        elif model_choice == "gbr":
            base_model = GradientBoostingRegressor(
                loss='absolute_error',
                n_estimators=200 if small_data else 600,
                max_depth=2 if small_data else 4,
                learning_rate=0.05 if small_data else 0.05,
                subsample=0.7 if small_data else 0.9,
                min_samples_leaf=5 if small_data else 1,
                min_samples_split=6 if small_data else 2,
                max_features='sqrt',
                random_state=42
            )
        else:
            raise ValueError("Selected regressor is not available in this environment.")
        pipe = Pipeline([('prep', preprocessor), ('model', base_model)])
        if use_log_target:
            return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
        return pipe

    if task == "classification":
        n_rows = int(n_rows) if n_rows is not None else None
        small_data = n_rows is not None and n_rows < 500
        if model_choice == "Logistic Regression":
            base_model = LogisticRegression(max_iter=300, class_weight="balanced")
        else:
            base_model = GradientBoostingClassifier(
                n_estimators=200 if small_data else 400,
                max_depth=3 if small_data else 4,
                learning_rate=0.06 if small_data else 0.05,
                subsample=0.85 if small_data else 0.9,
                random_state=42
            )
        return Pipeline([('prep', preprocessor), ('model', base_model)])

    raise ValueError(f"Unknown task: {task}")


def train_and_tune_model(
    X_train,
    y_train,
    model_choice,
    num_features,
    cat_features,
    use_log_target,
    sample_weight=None,
    random_state=42,
    n_iter_scale=1.0
):
    """Trains a tuned model using Regularized Linear, Extra Trees, or Gradient Boosting family."""

    def _regression_strata(y, max_bins=8):
        y = np.asarray(y)
        y = y[~np.isnan(y)]
        if y.shape[0] < 20:
            return None
        n_bins = min(max_bins, max(2, y.shape[0] // 25))
        try:
            bins = np.unique(np.quantile(y, np.linspace(0, 1, n_bins + 1)))
            if bins.shape[0] < 3:
                return None
            labels = np.digitize(y, bins[1:-1], right=True)
        except Exception:
            return None
        counts = np.bincount(labels)
        if counts.shape[0] < 2 or counts.min() < 2:
            return None
        return labels

    def _sample_weight_params():
        if sample_weight is None:
            return None
        return {"model__sample_weight": sample_weight}

    reg = build_model(
        "regression",
        model_choice,
        num_features,
        cat_features,
        use_log_target=use_log_target,
        n_rows=len(X_train)
    )

    # Lightweight tuning for non-linear models
    n_rows = len(X_train)
    small_data = n_rows < 500
    n_iter = 40 if small_data else 80
    n_iter = max(5, int(round(n_iter * float(n_iter_scale))))
    cv = 5 if small_data else 10

    # Randomized search on pipeline params
    if isinstance(reg, TransformedTargetRegressor):
        param_prefix = "regressor__model__"
    else:
        param_prefix = "model__"

    def _get_estimator(model):
        if isinstance(model, TransformedTargetRegressor):
            model = model.regressor if hasattr(model, "regressor") else model.regressor_
        if isinstance(model, Pipeline):
            return model.named_steps.get("model", model)
        return model

    estimator = _get_estimator(reg)

    y_arr = np.asarray(y_train, dtype=float)
    strata = _regression_strata(y_arr)
    if strata is not None:
        cv_splitter = list(StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state).split(X_train, strata))
    else:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    if model_choice == "linear":
        sw_params = _sample_weight_params()
        if sw_params:
            reg.fit(X_train, y_train, **sw_params)
        else:
            reg.fit(X_train, y_train)
        cv_scores = cross_val_score(reg, X_train, y_train, cv=cv_splitter, scoring='r2', n_jobs=-1)
        return reg, float(np.mean(cv_scores))
    elif model_choice == "ridge":
        # Use RidgeCV directly to avoid nested CV in an outer RandomizedSearchCV.
        alphas = np.logspace(-3, 3, 60)
        ridge_cv = RidgeCV(alphas=alphas, cv=cv_splitter, scoring='r2')

        if isinstance(reg, TransformedTargetRegressor):
            reg.regressor.set_params(model=ridge_cv)
            if sample_weight is not None:
                reg.fit(X_train, y_train, regressor__model__sample_weight=sample_weight)
            else:
                reg.fit(X_train, y_train)
            best_score = reg.regressor_.named_steps['model'].best_score_
        else:
            reg.set_params(model=ridge_cv)
            if sample_weight is not None:
                reg.fit(X_train, y_train, model__sample_weight=sample_weight)
            else:
                reg.fit(X_train, y_train)
            best_score = reg.named_steps['model'].best_score_

        return reg, float(best_score)
    elif model_choice == "rf":
        if small_data:
            param_dist = {
                f"{param_prefix}n_estimators": [150, 250, 350],
                f"{param_prefix}max_depth": [2, 3, 4, 5],
                f"{param_prefix}min_samples_split": [4, 6, 8],
                f"{param_prefix}min_samples_leaf": [3, 5, 8],
                f"{param_prefix}max_features": ["sqrt", 0.7],
            }
        else:
            param_dist = {
                f"{param_prefix}n_estimators": [300, 500, 800],
                f"{param_prefix}max_depth": [None, 12, 18, 24],
                f"{param_prefix}min_samples_split": [2, 3, 5],
                f"{param_prefix}min_samples_leaf": [1, 2, 4],
                f"{param_prefix}max_features": ["sqrt", 0.7, 1.0],
            }
    elif model_choice == "gbr":
        if small_data:
            param_dist = {
                f"{param_prefix}n_estimators": [120, 180, 240],
                f"{param_prefix}max_depth": [2, 3],
                f"{param_prefix}learning_rate": [0.03, 0.05, 0.08],
                f"{param_prefix}subsample": [0.6, 0.7, 0.8],
                f"{param_prefix}min_samples_leaf": [3, 5, 8],
                f"{param_prefix}min_samples_split": [4, 6, 8],
                f"{param_prefix}max_features": ["sqrt", None],
            }
        else:
            param_dist = {
                f"{param_prefix}n_estimators": [200, 400, 600],
                f"{param_prefix}max_depth": [2, 3, 4],
                f"{param_prefix}learning_rate": [0.03, 0.05, 0.08],
                f"{param_prefix}subsample": [0.8, 0.9, 1.0],
                f"{param_prefix}min_samples_leaf": [1, 2, 4],
                f"{param_prefix}min_samples_split": [2, 3, 5],
                f"{param_prefix}max_features": ["sqrt", None],
            }
    else:
        param_dist = {
            f"{param_prefix}n_estimators": [200, 400, 600],
            f"{param_prefix}max_depth": [3, 5, 7],
            f"{param_prefix}learning_rate": [0.03, 0.05, 0.08],
        }

    search = RandomizedSearchCV(
        reg,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_splitter,
        scoring='r2',
        random_state=random_state,
        n_jobs=-1
    )
    sw_params = _sample_weight_params()
    if sw_params:
        search.fit(X_train, y_train, **sw_params)
    else:
        search.fit(X_train, y_train)
    best_model = search.best_estimator_
    r2 = r2_score(y_train, best_model.predict(X_train))
    _ = r2  # kept for debugging parity if needed later
    return best_model, float(search.best_score_)


def train_classifier(X_train, y_train, model_choice, num_features, cat_features, sample_weight=None):
    """Train classifier with recall-focused tuning for High complexity."""
    clf = build_model(
        "classification",
        model_choice,
        num_features,
        cat_features,
        use_log_target=False,
        n_rows=len(X_train)
    )

    def _sample_weight_params():
        if sample_weight is None:
            return None
        return {"model__sample_weight": sample_weight}

    if model_choice == "Logistic Regression":
        sw_params = _sample_weight_params()
        if sw_params:
            clf.fit(X_train, y_train, **sw_params)
        else:
            clf.fit(X_train, y_train)
        return clf

    # Gradient Boosting Classifier tuning for recall on High class
    def _high_class_recall(y_true, y_pred):
        return float(
            recall_score(
                y_true,
                y_pred,
                labels=["High"],
                average="macro",
                zero_division=0
            )
        )

    scorer = make_scorer(_high_class_recall)
    n_rows = len(X_train)
    small_data = n_rows < 500
    n_iter = 20 if small_data else 50
    cv = 5 if small_data else 7

    # Guard CV against small/imbalanced class counts.
    cls_vals, cls_counts = np.unique(np.asarray(y_train), return_counts=True)
    min_class_count = int(cls_counts.min()) if cls_counts.size else 0
    if cls_vals.size < 2 or min_class_count < 2:
        sw_params = _sample_weight_params()
        if sw_params:
            clf.fit(X_train, y_train, **sw_params)
        else:
            clf.fit(X_train, y_train)
        return clf
    cv = min(cv, min_class_count)
    if cv < 2:
        sw_params = _sample_weight_params()
        if sw_params:
            clf.fit(X_train, y_train, **sw_params)
        else:
            clf.fit(X_train, y_train)
        return clf

    param_dist = {
        "model__n_estimators": [150, 250, 350, 450],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__subsample": [0.8, 0.9, 1.0],
    }
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_splitter,
        scoring=scorer,
        error_score='raise',
        random_state=42,
        n_jobs=-1
    )
    sw_params = _sample_weight_params()
    if sw_params:
        search.fit(X_train, y_train, **sw_params)
    else:
        search.fit(X_train, y_train)
    return search.best_estimator_
