# Daskan - Structural Project Intelligence Dashboard

## What This Project Is
This repository contains a **Streamlit-based machine learning application** for structural engineering project intelligence. The main app (`app5.py`) ingests project datasets, performs analytics, trains predictive models, explains model behavior, classifies project complexity, generates quotations, and identifies project personas through clustering.

At a high level, it answers:
- How much effort (hours) will a structural project require?
- Which factors drive that estimate?
- Is the estimate reliable for this specific input?
- What is a reasonable quote and staffing plan?
- Which project “persona” does this job resemble?

## Core Application Entry Point
- Main UI and orchestration: `app5.py`

The app is organized into 6 Streamlit tabs:
1. Deep Dive Analytics
2. AI Model Engine (regression training/tuning/deployment gate)
3. Model Explainability (SHAP/ALE)
4. Classification (Low/Medium/High complexity)
5. Smart Quotation (pricing/risk/resource planning)
6. Clustering & Personas (K-Means + PCA)

---

## Repository Architecture

### Main App
- `app5.py`
  - Streamlit UI layout
  - Session-state lifecycle
  - Data loading and tab orchestration
  - Model training triggers and diagnostics display
  - Quotation and staffing logic

### Modeling/ML Modules
- `modules/modeling.py`
  - Builds preprocessing + model pipelines
  - Regression model registry
  - Hyperparameter tuning (`RandomizedSearchCV`)
  - Classifier training tuned for High-complexity recall

- `modules/explainability.py`
  - Cached SHAP calculations
  - SHAP interaction values (tree models)
  - Custom 1D ALE implementation

- `modules/diagnostics.py`
  - Leakage detection rules
  - Feature selection utilities
  - Stability report by group
  - Residual outlier listing

- `modules/data.py`
  - CSV ingestion and profile routing
  - Training stats computation for drift checks

### Data Handlers
- `data_handlers/common.py`
  - Schema normalization and alias mapping
  - Data profile detection
  - Date parsing helpers
  - Feature engineering from granular logs
  - Rule-based complexity-class generation

- `data_handlers/full_exploration.py`
  - Handling of granular/full exploration data

- `data_handlers/project_metadata.py`
  - Handling of metadata-only datasets

---

## Data Profiles and Ingestion Logic
The app supports multiple schema shapes and auto-detects them.

### 1) Granular profile
Detected when columns like `date_logged` and `hours_worked` exist.

Path:
- `modules.data.load_uploaded_csv()` -> `data_handlers.full_exploration.load_full_exploration()` -> `feature_engineer_data()`

Behavior:
- Aggregates timesheet logs to project-level totals (`total_project_hours`)
- Preserves metadata columns where available
- Engineers additional features

### 2) Metadata profile
Detected when `total_project_hours` or `total_project_effort` exists.

Path:
- `modules.data.load_uploaded_csv()` -> `data_handlers.project_metadata.load_project_metadata()`

Behavior:
- Coerces key numeric columns
- Uses uploaded rows directly as modeling table (no major re-aggregation)

### 3) Unknown profile fallback
Path:
- `modules.data.load_uploaded_csv()` -> `feature_engineer_data()` fallback

Behavior:
- Best-effort feature engineering + labeling

### Alias normalization
`normalize_aliases()` handles column name variants (examples):
- `total_project_effort` -> `total_project_hours`
- `actual_duration_d` -> `actual_duration_days`
- `is_winter_day` -> `is_winter`

---

## Feature Engineering and Targets

### Regression target
- `target_col = total_project_hours`

### Primary numeric features (in `app5.py`)
- `surface_area_m2`
- `num_levels`
- `num_units`
- `building_height_m`
- `floor_area_ratio`
- `actual_duration_days`
- `num_revisions`

### Primary categorical features
- `project_type`
- `material_type`
- `scope_category`

### Engineered features (from `feature_engineer_data`)
Depending on available columns:
- `height_to_area_ratio`
- `area_per_level`
- `avg_floor_height` (if levels-height correlation is strong)
- Ordinal date features from planned/corrected dates
- `complexity_index` (rule-based numeric complexity proxy)

### Rule-based complexity class
`project_complexity_class` is generated when missing, using:
- Scale score (area/levels/height)
- Duration score (quartile-based)
- Revision score
- Context bump (scope/material)

Class mapping:
- `Low`, `Medium`, `High`

This class is used by the classification tab as the target.

---

## Preprocessing Pipeline (Model Input)
Defined in `modules/modeling.py`:

1. Numeric preprocessing
- Median imputation for missing numeric values
- `StandardScaler` for most numeric features
- `RobustScaler` for outlier-prone features (`num_revisions`, `design_hours_total` when present)

2. Categorical preprocessing
- Constant imputation (`Unknown`)
- `OneHotEncoder(handle_unknown='ignore')`

3. Combined with `ColumnTransformer`

This preprocessing is packaged with estimators in sklearn `Pipeline` objects.

---

## Regression Models and Tuning Logic

### Available model registry
`get_regression_registry()` returns:
- `linear`: Ridge regression baseline
- `rf`: ExtraTreesRegressor (random-forest-style ensemble)
- `gbr`: GradientBoostingRegressor

### Build-time defaults (data-size aware)
`build_model()` adapts defaults for small vs larger data (`n_rows < 500`).

### Hyperparameter tuning
`train_and_tune_model()` uses:
- `RandomizedSearchCV`
- CV strategy: `KFold(shuffle=True, random_state=42)`
- Scoring objective: `neg_mean_absolute_error`
- Iterations/folds adjusted by dataset size

Parameter spaces include:
- Ridge `alpha`
- ExtraTrees: trees/depth/split/leaf/max_features
- Gradient Boosting: estimators/depth/learning rate/subsample/split/leaf/max_features

### Target transformation support
Pipelines support optional log-target wrapping via `TransformedTargetRegressor` (`log1p`/`expm1`). In `app5.py`, this is currently set to `False` during training.

### Sample weighting
Training creates weights that up-weight high-effort projects (`y > 75th percentile`, weight 1.5 vs 1.0). This is used in initial splitting/tuning flows.

---

## AI Model Engine Tab (Tab 2) - End-to-End Logic
This is the operational heart of the app.

1. Readiness checks
- Confirms target exists and enough rows/labels are present
- Warns on missing training features

2. Leakage guard (optional checkbox)
- Uses `detect_leakage()` to remove post-outcome leakage columns

3. Feature reduction and hygiene
- Drops low-variance numeric features
- Drops highly correlated numeric features (> 0.98) on sufficient sample sizes
- Drops features with >40% missing rate

4. Train/test split
- Fixed `test_size=0.2`, `random_state=42`

5. Optional outlier exclusion
- Trains a preliminary model
- Computes residuals on train set
- Removes top-N highest residual rows (configurable)
- Retrains tuned model on filtered training set

6. Model evaluation metrics
- Test MAE
- Test R²
- Approx prediction interval shown as `+/- 1.5 * MAE`
- Baseline mean-predictor comparison for context

7. Diagnostics and MLOps simulation
- Stability report by groups (`project_type`, `material_type`, `scope_category`)
- Top residual outliers table
- Model card CSV export (timestamp/profile/model/metrics/features)
- Approval gate with threshold `R² > 0.70`
- Approved model version stored in session state and used for quoting readiness

8. Additional comparisons
- Baseline model comparison panel (Ridge vs ExtraTrees vs GradientBoosting)
- Permutation importance for current model
- Backtest/model sweep with CV + holdout metrics across all registered regressors

---

## Explainability Tab (Tab 3)
This tab explains why the regression model predicts what it predicts.

### SHAP global explanations
From `modules.explainability.get_shap_data()`:
- Uses cached SHAP computations for performance
- Chooses explainer type by estimator:
  - `shap.LinearExplainer` for linear/ridge
  - `shap.TreeExplainer` for tree models
- Produces:
  - Plotly beeswarm-like global view
  - Mean absolute SHAP bar chart

### Dependence analysis
- Select a primary feature
- Optional interaction coloring (manual or auto)
- Displays SHAP dependence scatter

### Interaction analysis
From `get_shap_interaction_data()`:
- Tree models only
- Computes pairwise interaction strengths
- Lists and charts strongest feature pairs

### ALE (Accumulated Local Effects)
From `get_ale_data()`:
- Custom 1D ALE implementation (quantile bins)
- Used as a correlation-safe alternative to SHAP dependence for numeric features
- Skips one-hot categorical feature names

### Classic SHAP plots
- Matplotlib SHAP summary
- SHAP waterfall for a selected example row

---

## Classification Tab (Tab 4)
Purpose: predict project complexity class (`Low/Medium/High`).

### Class target
- `project_complexity_class` (rule-derived if missing)

### Models
From `build_model(task="classification")` and `train_classifier()`:
- Logistic Regression (`class_weight='balanced'`)
- Gradient Boosting Classifier (with randomized tuning)

### Optimization focus
For gradient boosting classifier, tuning objective emphasizes recall of `High` class:
- scorer: recall on label `High`

### Training details
- Train/test split 80/20
- Additional class weighting in app logic (`High` weighted 2.0)

### Metrics shown
- Accuracy
- Macro Precision
- Recall (High)
- Macro F1
- Confusion matrix
- False Positive Rate for `High` class with threshold guidance (`< 0.30` preferred)

### Classifier explainability
- SHAP summary plot for Gradient Boosting classifier (tree explainer)

---

## Smart Quotation Tab (Tab 5)
Purpose: convert model output into operational estimates and pricing.

### Input payload
User enters:
- Project geometry and scale
- Type/scope/material
- Duration/revisions
- Cost rate, markup, team capacity

Derived input includes `floor_area_ratio` and mapped schema features.

### Prediction
- Uses trained regression model from session state to predict effort hours.

### Risk margin heuristic
Risk margin in hours is computed as:
- `risk_margin = pred_val * (0.10 + 0.05 * complexity_flags)`

`complexity_flags` increments for:
- Area >= 2500
- Levels >= 6 or height >= 25
- Duration >= 180 days
- Revisions >= 4

### Financial outputs
- Base cost = `estimated_hours * hourly_rate`
- Competitive quote = `base_cost * (1 + markup)`
- Conservative quote = `(base_cost + risk_margin * hourly_rate) * (1 + markup)`

### Drift/reliability guard
Compares input numeric values to training statistics (`mean`, `std`) and flags drift when:
- `abs(input - mean) > 2 * std`

### Resource planning
- Duration for 1 engineer and selected team size
- High-risk alerts if long duration, high risk margin, drift, or high revisions

### Task plan and Gantt
If granular logs are available:
- Builds historical task-category distribution by project type
- Allocates predicted hours across tasks
- Converts to weeks using team capacity
- Renders conceptual Plotly timeline (Gantt-style)

---

## Clustering & Personas Tab (Tab 6)
Purpose: discover unsupervised project archetypes.

### Method
- Numeric feature matrix from `NUM_FEATURES`
- Median imputation + `StandardScaler`
- `KMeans(n_clusters=k, random_state=42, n_init=10)`
- Optional silhouette score when valid

### Persona labeling
Cluster labels are post-processed with domain heuristics (materials + average levels), generating names such as:
- Light Timber Residential
- Complex Concrete High-Rise
- Steel Mid/High-Rise
- Dense Mid-Rise Mixed
- Low-Rise Mixed

### Visualization
- Cluster size table
- Mean profile table by cluster
- PCA 2D projection scatter (`PC1`, `PC2`)

---

## Data Quality and Diagnostics Logic

### Data quality checks (`run_data_quality_checks`)
- Missing-rate report on required columns
- Basic numeric range checks
- Duplicate `project_id` detection
- Training metadata warnings (target missing/high-missing)

### Correlation analysis
- Numeric-only correlation heatmap
- Excludes all-null and constant features before correlation

### Stability report (`stability_report`)
- Computes group-level MAE where group sample count >= 5

### Residual outliers (`residual_outliers`)
- Lists top-N largest absolute errors for targeted review

---

## Session State and Workflow Control
Key state fields in `app5.py` include:
- `models`: current trained regression model
- `train_r2`: latest test R²
- `y_test`, `y_preds`: held-out predictions for diagnostics
- `approved_model_version`, `approved_r2`: deployment gate artifacts
- `train_stats`: feature means/std for drift checks
- `quote_generated`, `pred_val`, `risk_margin`: quotation outputs
- `last_train_metrics`, `last_train_features`: model card/export context

This enables tab-to-tab continuity without retraining every interaction.

---

## Algorithms Summary (Quick Reference)

### Supervised Regression
- Ridge Regression
- Extra Trees Regressor
- Gradient Boosting Regressor
- Randomized hyperparameter search + KFold CV

### Supervised Classification
- Logistic Regression
- Gradient Boosting Classifier
- Recall-focused tuning for `High` complexity

### Explainability
- SHAP (global, dependence, interactions, waterfall)
- Permutation Importance
- ALE (custom implementation)

### Unsupervised Learning
- K-Means clustering
- PCA for 2D projection
- Silhouette score quality indicator

### Statistical/Rule-based components
- Correlation matrix analysis
- Missingness and sanity checks
- Drift check using z-score style threshold (2 sigma)
- Rule-based complexity scoring system

---

## Practical Interpretation of Model Outputs
- **MAE**: average absolute error in hours. This is the most intuitive uncertainty measure for estimators.
- **R²**: variance explained on test data. The app treats 0.70 as a production-quality gate.
- **Permutation importance**: performance drop when a feature is shuffled (model dependence proxy).
- **SHAP**: signed contribution of each transformed feature per prediction.
- **ALE**: local effect estimate that is more robust under correlated features.
- **Silhouette score**: how separated clusters are (higher is generally better).

---

## How to Run
From repository root:

```bash
streamlit run app5.py
```

Then upload either:
- Granular full-exploration CSV (timesheet-level), or
- Metadata CSV (project-level)

---

## Typical End-to-End Usage
1. Upload data in sidebar.
2. Validate schema/quality in Analytics tab.
3. Train and tune a regression model in AI Model Engine.
4. Review diagnostics, outliers, and feature importance.
5. Approve model if R² exceeds threshold.
6. Use Explainability tab to inspect feature effects and interactions.
7. Train classification model for complexity-risk operations.
8. Generate quote and staffing/timeline plan in Smart Quotation.
9. Review project personas via clustering for portfolio strategy.

---

## Current Design Assumptions and Limitations
- The production gate is based on a single R² threshold (`0.70`), not full governance.
- Drift detection uses simple mean/std boundaries (2 sigma), not advanced drift tests.
- Complexity class is rule-derived, so classification quality depends on heuristic target quality.
- Some optional imports (xgboost/lightgbm/catboost) are checked in code but not registered in current active regression options.
- The app relies heavily on uploaded data quality and schema consistency.

---

## Why This Codebase Is Valuable
This codebase is a strong applied ML decision-support tool: it combines data validation, model training/tuning, interpretability, deployment gating, cost estimation, and operational planning in one coherent Streamlit workflow. It is not just a model demo; it is an end-to-end estimation system designed for real project decision contexts.


