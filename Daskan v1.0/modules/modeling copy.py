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
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, recall_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, KFold

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
        "gbr": {"label": "Gradient Boosting (XGBoost / LightGBM / CatBoost if available)", "available": True},
        "rf": {"label": "Random Forest (Extra Trees)", "available": True},
        "linear": {"label": "Regularized Linear Regression (Baseline + Explainability)", "available": True},
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
            base_model = Ridge(alpha=1.0, random_state=42)
        elif model_choice == "rf":
            base_model = ExtraTreesRegressor(
                n_estimators=400 if small_data else 800,
                max_depth=14 if small_data else 24,
                min_samples_split=3,
                min_samples_leaf=2 if small_data else 1,
                max_features='sqrt',
                random_state=42
            )
        elif model_choice == "gbr":
            if _HAS_XGB:
                base_model = xgb.XGBRegressor(
                    n_estimators=400 if small_data else 800,
                    max_depth=6 if small_data else 8,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    n_jobs=-1,
                    random_state=42
                )
            elif _HAS_LGB:
                base_model = lgb.LGBMRegressor(
                    n_estimators=400 if small_data else 800,
                    max_depth=-1,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    random_state=42
                )
            elif _HAS_CAT:
                base_model = CatBoostRegressor(
                    iterations=400 if small_data else 800,
                    depth=6 if small_data else 8,
                    learning_rate=0.05,
                    loss_function="MAE",
                    verbose=False,
                    random_state=42
                )
            else:
                base_model = GradientBoostingRegressor(
                    loss='absolute_error',
                    n_estimators=300 if small_data else 600,
                    max_depth=3 if small_data else 4,
                    learning_rate=0.06 if small_data else 0.05,
                    subsample=0.85 if small_data else 0.9,
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


def train_and_tune_model(X_train, y_train, model_choice, num_features, cat_features, use_log_target, sample_weight=None):
    """Trains a tuned model using Regularized Linear, Extra Trees, or Gradient Boosting family."""

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

    if model_choice == "linear":
        param_dist = {
            f"{param_prefix}alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        }
    elif model_choice == "rf":
        param_dist = {
            f"{param_prefix}n_estimators": [300, 500, 800],
            f"{param_prefix}max_depth": [None, 12, 18, 24],
            f"{param_prefix}min_samples_split": [2, 3, 5],
            f"{param_prefix}min_samples_leaf": [1, 2, 4],
            f"{param_prefix}max_features": ["sqrt", 0.7, 1.0],
        }
    elif model_choice == "gbr":
        if _HAS_XGB and estimator.__class__.__name__ == "XGBRegressor":
            param_dist = {
                f"{param_prefix}n_estimators": [300, 500, 800],
                f"{param_prefix}max_depth": [4, 6, 8],
                f"{param_prefix}learning_rate": [0.03, 0.05, 0.08],
                f"{param_prefix}subsample": [0.8, 0.9, 1.0],
                f"{param_prefix}colsample_bytree": [0.7, 0.8, 0.9],
            }
        elif _HAS_LGB and estimator.__class__.__name__ == "LGBMRegressor":
            param_dist = {
                f"{param_prefix}n_estimators": [300, 500, 800],
                f"{param_prefix}max_depth": [-1, 6, 10],
                f"{param_prefix}learning_rate": [0.03, 0.05, 0.08],
                f"{param_prefix}subsample": [0.8, 0.9, 1.0],
                f"{param_prefix}colsample_bytree": [0.7, 0.8, 0.9],
            }
        elif _HAS_CAT and estimator.__class__.__name__ == "CatBoostRegressor":
            param_dist = {
                f"{param_prefix}iterations": [300, 500, 800],
                f"{param_prefix}depth": [4, 6, 8],
                f"{param_prefix}learning_rate": [0.03, 0.05, 0.08],
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
        cv=KFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1
    )
    sw_params = _sample_weight_params()
    if sw_params:
        search.fit(X_train, y_train, **sw_params)
    else:
        search.fit(X_train, y_train)
    best_model = search.best_estimator_
    r2 = r2_score(y_train, best_model.predict(X_train))
    return best_model, r2


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
    scorer = make_scorer(recall_score, labels=["High"], average="macro", zero_division=0)
    n_rows = len(X_train)
    small_data = n_rows < 500
    n_iter = 20 if small_data else 50
    cv = 5 if small_data else 7

    param_dist = {
        "model__n_estimators": [150, 250, 350, 450],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__subsample": [0.8, 0.9, 1.0],
    }
    search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scorer,
        random_state=42,
        n_jobs=-1
    )
    sw_params = _sample_weight_params()
    if sw_params:
        search.fit(X_train, y_train, **sw_params)
    else:
        search.fit(X_train, y_train)
    return search.best_estimator_
