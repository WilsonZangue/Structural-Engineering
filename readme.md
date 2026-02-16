# Daskan - Structural Project Intelligence Dashboard

## 1. Executive Summary
This repository provides an end-to-end machine learning decision-support application for structural engineering project planning.  
The application is implemented in Streamlit (`app5.py`) and integrates:
- Data ingestion and schema harmonization
- Data quality diagnostics
- Regression model training, tuning, and evaluation
- Explainability (SHAP, permutation importance, ALE)
- Multiclass project complexity classification
- Quotation and staffing/timeline planning
- Unsupervised clustering for project personas

The system is designed to answer operational questions such as:
- How many total engineering hours should be expected for a project?
- Which factors are driving the estimate?
- Is the estimate reliable for the current input profile?
- What quoting range is justified based on risk?
- Which portfolio segment/persona does the project match?

## 1.1 What This Project Is
This repository contains a Streamlit-based machine learning application for structural engineering project intelligence.  
The main app (`app5.py`) ingests project datasets, performs analytics, trains predictive models, explains model behavior, classifies project complexity, generates quotations, and identifies project personas through clustering.

---

## 2. Repository and Code Structure

### 2.1 Primary Entry Point
- `app5.py`: Main Streamlit application. It orchestrates all workflows and user interactions.

### 2.2 Core Python Packages
- `modules/modeling.py`: Pipeline construction, regression tuning, classification tuning.
- `modules/explainability.py`: Cached SHAP utilities and ALE computation helper.
- `modules/diagnostics.py`: Leakage checks, feature filtering, stability reporting, residual outlier extraction.
- `modules/data.py`: Uploaded CSV ingestion router and training-statistics computation.

### 2.3 Data Handler Layer
- `data_handlers/common.py`: Shared schema normalization, profile detection, feature engineering, complexity class derivation.
- `data_handlers/full_exploration.py`: Loader for granular/timesheet-profile datasets.
- `data_handlers/project_metadata.py`: Loader for metadata-only datasets.

### 2.4 Supporting Data and Assets
- Synthetic and sample datasets (`Synthetic Data/`, root CSV files)
- Branding assets (`DaskanLogo.png`, `cropped-DaskanLogo.png`)

---

## 3. Full Application Architecture (`app5.py`)
`app5.py` contains four major layers:

1. Configuration and constants  
- Visual theme constants  
- Feature lists (`NUM_FEATURES`, `LEAKY_NUM_FEATURES`, `CAT_FEATURES`)  
- Page-level Streamlit configuration

2. Session-state lifecycle  
- Initializes model artifacts, metrics, approval state, and quote state  
- Resets target column state when necessary

3. Data ingestion and preprocessing setup  
- Loads and profiles uploaded CSV  
- Runs domain interaction feature engineering  
- Computes training reference statistics for drift checks

4. Six operational tabs  
- Tab 1: Deep Dive Analytics  
- Tab 2: AI Model Engine  
- Tab 3: Model Explainability  
- Tab 4: Complexity Classification  
- Tab 5: Smart Quotation  
- Tab 6: Clustering and Personas

---

## 4. Data Ingestion, Profiles, and Normalization

### 4.1 Ingestion Router (`modules/data.py`)
Function: `load_uploaded_csv(uploaded_file)`
- Reads CSV with delimiter auto-detection (`sep=None`, Python engine)
- Trims header whitespace
- Normalizes known alias columns
- Detects profile (`granular`, `metadata`, `unknown`)
- Routes to profile-specific handlers

Outputs:
- `df_modeling`: modeling table
- `df_analytics`: analytics table
- `profile`: detected schema profile
- `data_source`: human-readable source label

### 4.2 Profile Detection (`data_handlers/common.py`)
Function: `_detect_data_profile(df)`
- `granular` if both `date_logged` and `hours_worked` exist
- `metadata` if `total_project_hours` or `total_project_effort` exists
- `unknown` otherwise

### 4.3 Alias Normalization
Function: `normalize_aliases(df)`
Current explicit mappings:
- `is_winter_day` -> `is_winter`
- `actual_duration_d` -> `actual_duration_days`
- `total_project_effort` -> `total_project_hours`

### 4.4 Granular Handler (`data_handlers/full_exploration.py`)
Function: `load_full_exploration(raw_df)`
- Confirms granular profile
- Applies shared feature engineering (`feature_engineer_data`)
- Returns:
  - `df_modeling`: aggregated/engineered project-level table
  - `df_analytics`: original granular table for timeline/task analytics

### 4.5 Metadata Handler (`data_handlers/project_metadata.py`)
Function: `load_project_metadata(raw_df)`
- Confirms metadata profile
- Coerces key numeric fields
- Does not perform granular aggregation
- Returns metadata table for both modeling and analytics

### 4.6 Unknown Profile Fallback
In `modules/data.py`, unknown profiles use `feature_engineer_data(raw_df)` as best-effort normalization/engineering.

---

## 5. Feature Engineering and Label Construction

### 5.1 Shared Engineering (`data_handlers/common.py`)
Function: `feature_engineer_data(df_analytics)`

Core behavior:
- Parses date columns robustly (`_parse_date_column`, `_infer_dayfirst`)
- If granular logs exist (`log_id` present), aggregates to project-level:
  - `total_project_hours` from `hours_worked` sum
  - `num_logs` from `date_logged` count
  - Retains first value of metadata candidates
- Numeric coercion of selected modeling columns
- Missing-value handling for selected columns (`num_revisions` default fill to 0)

Engineered numeric features:
- `height_to_area_ratio`
- `area_per_level`
- `avg_floor_height` (only when level-height correlation is strong)
- `complexity_index` (rule-based continuous complexity proxy)
- Date ordinal features:
  - `planned_start_ordinal`
  - `planned_end_ordinal`
  - `corrected_start_ordinal`
  - `corrected_end_ordinal`
- Date-derived discrete features (when source date exists):
  - `month_started`
  - `quarter`

### 5.2 Rule-Based Complexity Class Generation
If absent, `project_complexity_class` is generated in engineering logic and reused by classification flows.

Scoring components:
- Scale score from area, levels, and height
- Duration score from quartile position
- Revision score from revision count thresholds
- Context bump from scope/material context

Class mapping:
- `Low`
- `Medium`
- `High`

### 5.3 Additional App-Level Interaction Features (`app5.py`)
Function: `add_interaction_features(df)`
- `area_per_unit = surface_area_m2 / (num_units + 1)`
- `height_per_level = building_height_m / (num_levels + 1)`
- `complexity_interaction_index = surface_area_m2 * num_levels`
- `revision_intensity = num_revisions / (actual_duration_days + 1)`

### 5.4 Drift Reference Statistics
Function: `compute_train_stats(df_modeling, num_cols)` in `modules/data.py`
- Computes `mean` and `std` for available numeric training columns
- Stored in session state and used during quotation-time drift checks

### 5.5 Data Handling and Modeling Thought Process (Design Rationale)
This section documents the modeling rationale used after dataset import and before/through model training.

1. Why feature engineering is done after import  
- Imported project data can be correct at column level but still weak at relationship level.  
- The target (`total_project_hours`) is usually driven by interactions (scale per unit, vertical intensity, change pressure), not only by raw single columns.  
- The interaction layer is added immediately after ingestion so every downstream stage (EDA, split, train, explainability, quotation) uses the same enriched feature space.

2. Why these interaction features were added  
- `area_per_unit = surface_area_m2 / (num_units + 1)`  
  Rationale: separates large-area projects that are distributed across many units from projects with concentrated area per unit.  
  Modeling value: improves sensitivity to density/complexity differences that raw `surface_area_m2` alone does not capture.

- `height_per_level = building_height_m / (num_levels + 1)`  
  Rationale: approximates average floor-to-floor intensity and vertical design burden.  
  Modeling value: helps distinguish projects with similar height but different number of levels, which often have different effort requirements.

- `complexity_interaction_index = surface_area_m2 * num_levels`  
  Rationale: captures the compound effect of horizontal scale and vertical repetition.  
  Modeling value: gives tree and linear-family models a direct signal for multiplicative complexity that would otherwise need to be inferred indirectly.

- `revision_intensity = num_revisions / (actual_duration_days + 1)`  
  Rationale: normalizes revision count by project duration so short, revision-heavy projects are properly identified as high-friction workloads.  
  Modeling value: improves prediction in cases where absolute revision count alone underestimates effort volatility.

3. Why this typically improves model usefulness and performance  
- Better signal representation: domain interactions convert raw inputs into workload-relevant predictors.  
- Lower ambiguity: similar raw projects become more separable in feature space.  
- More stable tuning: hyperparameter search spends less effort compensating for missing domain structure in the data.  
- Better generalization: engineered ratios/interactions often transfer better across project scales than raw magnitudes alone.

4. Expected impact by model family  
- Linear Regression / Ridge Regression: largest benefit, because engineered interactions make nonlinear relationships learnable in linear space.  
- ExtraTrees / GradientBoosting: still beneficial, because explicit interaction variables reduce the burden on deep splits and can improve robustness on small/medium datasets.  
- Classification models: interaction terms improve separability of `Low`/`Medium`/`High` complexity when class boundaries are driven by combined effects rather than single fields.

5. Connection to training and tuning workflow  
- Split is performed before model fitting; preprocessing and transforms are fit on training folds only, reducing leakage risk.  
- Imputation and scaling make ratio/interaction features numerically stable for linear and distance-sensitive components.  
- CV-based tuning evaluates models on held-out folds where these engineered terms often improve consistency of R-squared, MAE, and class-level recall behavior.

6. Practical interpretation for documentation  
- These engineered features are not arbitrary transformations; they encode structural engineering workload logic.  
- They are intended to increase the usefulness of imported datasets by adding domain context that raw schema fields do not explicitly provide.  
- In operational use, this usually improves estimate quality, reduces underestimation on complex projects, and produces more defensible model explanations.

### 5.6 Before vs After Feature Space (Documentation View)
| Stage | Feature Set | Example Columns | Modeling Limitation / Benefit |
|---|---|---|---|
| Before engineering | Raw metadata only | `surface_area_m2`, `num_levels`, `num_units`, `building_height_m`, `actual_duration_days`, `num_revisions`, categorical context | Limitation: model sees individual magnitudes but not workload intensity relationships between variables. |
| After engineering | Raw metadata + interaction features | `area_per_unit`, `height_per_level`, `complexity_interaction_index`, `revision_intensity` (plus original columns) | Benefit: model receives explicit density, vertical intensity, compound scale, and revision-pressure signals. |
| Training impact | Enriched feature space in the same pipeline/CV workflow | Same train/test split and CV settings, but richer predictors | Benefit: typically improved fit stability, better outlier handling, and more meaningful tuning behavior. |
| Explainability impact | Higher-information feature attribution | SHAP and permutation now include domain interaction terms | Benefit: explanations align better with engineering logic and are easier to defend in documentation. |
| Operational impact | Better quote and risk sensitivity | Interaction-driven inputs feed quotation and drift checks | Benefit: reduced risk of optimistic estimates for high-complexity or revision-dense projects. |

---

## 6. Canonical Feature and Target Definitions

### 6.1 Regression Target
- `total_project_hours`

### 6.2 Numeric Feature Set (`NUM_FEATURES` in `app5.py`)
- `surface_area_m2`
- `num_levels`
- `num_units`
- `building_height_m`
- `floor_area_ratio`
- `actual_duration_days`
- `num_revisions`
- `area_per_unit`
- `height_per_level`
- `complexity_interaction_index`
- `revision_intensity`

### 6.3 Categorical Feature Set (`CAT_FEATURES`)
- `project_type`
- `material_type`
- `scope_category`

### 6.4 Leak-Managed Feature List
- `LEAKY_NUM_FEATURES` is defined for explicit leak-prone numeric fields (currently empty in active configuration).

---

## 7. Session State Design and Runtime Contract
The application uses session state as a shared runtime contract across tabs.

Primary keys:
- `models`: trained regression model object
- `mode`: model mode marker
- `train_r2`: latest regression test R-squared
- `y_test`, `y_preds`: held-out regression predictions for diagnostics
- `approved_model_version`, `approved_r2`: deployment gate artifacts
- `train_stats`: numeric mean/std table for drift checks
- `last_train_metrics`, `last_train_features`: model card context
- `quote_generated`, `pred_val`, `risk_margin`, `input_data`, `profit_markup`: quotation state
- `clf_model`, `clf_choice_model`: classification model state
- `backtest_results`: backtest evaluation artifact table

Design objective:
- Preserve continuity between training, explainability, diagnostics, and quotation workflows without retraining on each interaction.

---

## 8. Model Construction and Training Layer (`modules/modeling.py`)

### 8.1 Regression Registry
Function: `get_regression_registry()`
- `linear`: Linear Regression
- `ridge`: Ridge Regression
- `rf`: ExtraTrees regressor
- `gbr`: GradientBoosting regressor

### 8.2 Pipeline Builder
Function: `build_model(task, model_choice, num_features, cat_features, use_log_target=True, n_rows=None)`

Preprocessing stack:
- Numeric:
  - Median imputation
  - `StandardScaler` for standard numeric columns
  - `RobustScaler` for robust-designated numeric columns
- Categorical:
  - Constant imputation (`Unknown`)
  - One-hot encoding (`handle_unknown='ignore'`)

Pipeline implementation:
- `ColumnTransformer` + estimator in a sklearn `Pipeline`
- Optional `TransformedTargetRegressor` wrapper (`log1p`/`expm1`) for regression

### 8.3 Regression Training and Tuning
Function: `train_and_tune_model(X_train, y_train, model_choice, num_features, cat_features, use_log_target, sample_weight=None)`

Capabilities:
- Regression target stratification helper for CV when feasible
- Weighted fitting support via `sample_weight`
- Data-size-adaptive CV fold selection (`cv`)
- Model-specific training/tuning paths:
  - `linear`: direct fit + cross-validated `R2` summary
  - `ridge`: `RidgeCV` with log-spaced alpha grid (`10^-3` to `10^3`, 60 values)
  - `rf` and `gbr`: `RandomizedSearchCV` over model-specific search spaces
- Returns:
  - best estimator
  - best cross-validated score

### 8.4 Classification Training and Tuning
Function: `train_classifier(X_train, y_train, model_choice, num_features, cat_features, sample_weight=None)`

Capabilities:
- Logistic regression direct training path
- Gradient boosting randomized tuning path
- Stratified CV (`StratifiedKFold`)
- Custom scoring focused on `High` class recall
- Explicit `error_score='raise'` for transparent error diagnosis
- Returns fitted best estimator

---

## 9. Data Diagnostics Layer (`modules/diagnostics.py`)

### 9.1 Leakage Detection
Function: `detect_leakage(feature_list, target_col, allow_leakage=False)`
- Applies target-specific leak maps
- Returns offending features to be excluded before training

### 9.2 Feature Filtering
Function: `select_features(df, num_features, cat_features, corr_threshold=0.98)`
- Drops low-variance numeric features
- Drops highly correlated numeric features on adequate sample sizes
- Returns retained numeric/categorical sets plus dropped-feature reasons

### 9.3 Stability Diagnostics
Function: `stability_report(y_true, y_pred, df, group_cols)`
- Computes group-level MAE by selected categorical segments
- Requires minimum sample count per group

### 9.4 Residual Outlier Extraction
Function: `residual_outliers(y_true, y_pred, df, top_n=10)`
- Produces top absolute-error rows for targeted review

---

## 10. Explainability Layer (`modules/explainability.py`)

### 10.1 SHAP Cache and Core Data
Function: `get_shap_data(_models, X_train)`
- Cached via `st.cache_data`
- Resolves wrapped pipelines (`TransformedTargetRegressor`)
- Produces:
  - SHAP explainer object
  - SHAP values
  - transformed training matrix
  - transformed feature names

Explainer selection:
- Linear explainers for linear models (`LinearRegression`, `Ridge`, `RidgeCV`)
- Tree explainers for tree-based models

### 10.2 SHAP Interaction Values
Function: `get_shap_interaction_data(_models, X_train)`
- Cached interaction matrix for tree-based models
- Returns interaction values + feature names
- Returns `None` for linear estimators

### 10.3 ALE Utility
Function: `get_ale_data(model_pipe, X_train_raw, feature_name)`
- Computes one-dimensional ALE profiles without external dependencies
- Handles insufficient-variation conditions gracefully

---

## 11. Split Strategy and Evaluation Methodology

### 11.1 Regression Split Utility (`app5.py`)
Function: `split_regression_best_practice(X, y, sample_weight=None, split_mode="auto", test_size=0.2, random_state=42)`

Supported modes:
- `auto`: stratified random split on target bins (fallback to random)
- `group`: group-aware split by `project_id` when feasible
- `time`: chronological split when valid date feature exists

Outputs:
- train/test features and targets
- optional split weights
- effective split mode used after fallback logic

### 11.2 Classification Split
- Uses `train_test_split(..., stratify=yc)` when class diversity allows

### 11.3 Tuning Evaluation
- Regression tuning score: `r2`
- Classification tuning score: custom recall on `High`
- Ridge-specific tuning: internal CV score from `RidgeCV` (`best_score_`)
- Additional UI metrics include MAE, R-squared, macro precision, macro F1, confusion matrix, and high-class false positive rate

---

## 12. Tab-by-Tab Functional Specification

## 12.1 Tab 1: Deep Dive Analytics
Core capabilities:
- Portfolio KPI summary
- Effort vs. scale scatter visualization
- Effort distribution histogram
- Data quality diagnostics with missingness and range checks
- Metadata schema health report
- Rule-based complexity table by project
- Correlation heatmap with defensive filtering of invalid numeric columns
- Project-level timeline (S-curve) and task-breakdown charts for granular datasets

Key helper functions used:
- `run_data_quality_checks(...)`
- Local `_compute_complexity_class(...)`

## 12.2 Tab 2: AI Model Engine
Core capabilities:
- Feature availability checks and warnings
- Optional leakage guard exclusion
- Feature selection and high-missing feature drop
- Readiness gate (minimum sample and target availability checks)
- Multiple split strategies with fallback messaging
- Regression model selection from registry
- Optional outlier-removal retraining workflow
- Hyperparameter tuning and held-out testing
- Baseline MAE benchmark (mean predictor)
- Cross-validation summary (MAE and R-squared)
- Group stability report and top residual outliers
- Model card CSV export
- Permutation feature importance (current model)
- Backtest model sweep across all registered regressors
- Predicted-vs-actual scatter for holdout evaluation
- MLOps-style approval gate with version stamp

## 12.3 Tab 3: Model Explainability
Core capabilities:
- SHAP global beeswarm-style visualization
- SHAP mean absolute importance bar chart
- Feature dependence view
- Interaction-strength ranking and pair exploration
- ALE alternative view for correlation-safe effect interpretation
- Classic SHAP summary and waterfall plots

## 12.4 Tab 4: Complexity Classification
Core capabilities:
- Automatic generation of complexity classes when not provided
- Classifier choice:
  - Logistic Regression
  - Gradient Boosting Classifier
- Class weighting strategy to emphasize `High` class
- Metrics:
  - Accuracy
  - Macro precision
  - Recall for `High`
  - Macro F1
  - Confusion matrix
  - High-class false positive rate
- Optional SHAP view for gradient boosting classifier

## 12.5 Tab 5: Smart Quotation
Core capabilities:
- Interactive project specification input form
- Real-time derived feature construction from user input
- Model-based effort estimation
- Risk-margin heuristic based on structural complexity signals
- Financial outputs:
  - Base cost
  - Competitive quote
  - Conservative quote
- Drift and reliability check against training distributions
- Resource/timeline planning by team-size assumption
- Granular-data-dependent task breakdown and conceptual Gantt-like timeline

## 12.6 Tab 6: Clustering and Personas
Core capabilities:
- Numeric-feature clustering pipeline (imputation, scaling, K-Means)
- Cluster-quality indicator (silhouette score when valid)
- Cluster distribution summary
- Cluster profile table (mean feature values)
- PCA-based 2D projection chart
- Rule-based persona naming based on structural patterns

---

## 13. Training, Testing, and Tuning Variable Reference

### 13.1 Feature-Selection Variables
- `selected_num_features`: active numeric training features after leakage and feature filtering
- `selected_features`: active full feature list (`numeric + categorical`)
- `available_features`: subset of active features present in current `df_modeling`

### 13.2 Matrix and Label Variables
- `X`, `y`: regression feature matrix and target series
- `X_train`, `X_test`, `y_train`, `y_test`: regression split artifacts
- `Xc`, `yc`: classification matrix and labels
- `Xc_train`, `Xc_test`, `yc_train`, `yc_test`: classification split artifacts

### 13.3 Weight Variables
- `sample_weight`: regression weights (high-effort weighting)
- `class_weights`: classification per-row weights
- `high_weight`: scalar emphasis factor for `High` class

### 13.4 Model and Configuration Variables
- `model_choice`: selected regression algorithm key
- `clf_choice`: selected classifier family
- `use_log_target`: optional target-transform switch for regression
- `cv_folds`: CV fold count used in diagnostics

### 13.5 Output and Evaluation Variables
- `preds`: prediction vector on holdout
- `mae_test`: test MAE
- `r2_test`: test R-squared
- `acc`, `precision`, `recall_high`, `f1`: classification metrics

### 13.6 Tuning Variables in `modules/modeling.py`
- `cv`: number of fold partitions
- `cv_splitter`: fold generator object
- `param_dist`: randomized-search parameter grid (used by `rf` and `gbr`)
- `n_iter`: number of random search trials (used by `rf` and `gbr`)
- `search`: `RandomizedSearchCV` object (used by `rf` and `gbr`)
- `alphas`: log-spaced ridge alpha candidates (used by `ridge` with `RidgeCV`)
- `best_model`: selected best estimator for search-based branches

---

## 14. Quality Controls and Safety Checks
Implemented controls include:
- Missing-column detection and runtime warnings
- Optional leakage guard for post-outcome variables
- Low-variance and high-correlation feature suppression
- High-missing-feature exclusion (`>40%`)
- Minimum data-volume checks before training
- Split-strategy fallback logic
- Drift detection for inference-time inputs
- Group-level stability reporting
- Residual-outlier surfacing for human review

---

## 15. Operational Workflow
Recommended usage order:
1. Upload a dataset.
2. Review quality and schema diagnostics in Tab 1.
3. Train and tune regression in Tab 2.
4. Inspect diagnostics and explainability in Tabs 2 and 3.
5. Optionally train classification in Tab 4.
6. Approve model for operational quoting.
7. Generate quotation and resource plans in Tab 5.
8. Analyze personas in Tab 6 for portfolio strategy.

---

## 16. Execution Instructions
From repository root:

```bash
streamlit run app5.py
```

Input requirements:
- CSV upload through sidebar
- Supported profiles: granular timesheet logs or project metadata

---

## 17. Algorithms Summary (Quick Reference)

### 17.1 Supervised Regression
- Linear Regression
- Ridge Regression (`RidgeCV` tuning path)
- Extra Trees Regressor
- Gradient Boosting Regressor

### 17.2 Supervised Classification
- Logistic Regression
- Gradient Boosting Classifier
- Recall-focused tuning for the `High` complexity class

### 17.3 Explainability
- SHAP (global summaries, dependence, interactions when supported, waterfall)
- Permutation Importance
- ALE for correlation-safe effect views

### 17.4 Unsupervised Learning
- K-Means clustering
- PCA projection for 2D cluster visualization
- Silhouette score quality indicator (when valid)

### 17.5 Statistical and Rule-Based Components
- Correlation matrix analysis
- Missingness and range checks
- Drift detection based on training mean/std thresholds
- Rule-based complexity scoring system

---

## 18. Practical Interpretation of Model Outputs
- **MAE**: average absolute error in hours; the most intuitive accuracy measure for planning.
- **R-squared**: fraction of variance explained on held-out data; used for production gate decisions.
- **Permutation Importance**: performance drop when a feature is shuffled; indicates model dependence on that feature.
- **SHAP**: signed contribution of each transformed feature to an individual prediction.
- **Silhouette Score**: separation quality of clusters; higher values generally indicate cleaner segmentation.

---

## 19. Best Practices for Splitting Data in ML (Implemented)
1. Randomized split with target stratification for regression.
- Default split mode is Auto (Stratified Random).
- Target (`total_project_hours`) is binned when possible for stratification.
- Falls back safely to random split when stratification is not feasible.

2. Group-aware split support.
- Group-Aware mode uses `project_id` to prevent leakage across repeated projects.
- Falls back with explicit mode reporting when grouping cannot be applied.

3. Time-aware split support.
- Time-Aware mode uses chronological holdout when date columns exist.
- Falls back safely if time columns are unavailable or invalid.

4. Class-balanced split for classification.
- Classification uses stratified train/test splits for `project_complexity_class` when class diversity permits.

5. Cross-validation best-practice updates.
- Regression uses target-stratified fold behavior when feasible, with `KFold` fallback.
- Classification uses `StratifiedKFold` for class-preserving CV.

6. Leakage-conscious workflow.
- Data is split before fitting.
- Preprocessing is fit inside pipelines and CV folds.
- Leakage guard remains available for post-outcome feature exclusion.

7. Test-set hygiene.
- Final test split remains isolated for metric reporting and diagnostics.

---

## 20. Current Assumptions and Limitations
- Production approval is a threshold-based simulation, not a full MLOps governance stack.
- Drift detection is currently mean/std threshold-based rather than using specialized drift tests.
- Classification labels may be heuristic-derived when not provided by source data.
- Model behavior and reliability remain dependent on data quality, coverage, and schema consistency.

---

## 21. Why This Codebase Is Valuable
This codebase is a practical machine-learning decision-support system rather than a single-model demonstration.  
It combines data validation, feature engineering, model training and tuning, explainability, approval gating, financial quotation, and operational planning in one coherent workflow.  
That integration is what makes it valuable for real project decision contexts.
