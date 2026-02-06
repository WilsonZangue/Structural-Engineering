# Daskan Intelligence | AI Project Estimator - PRO

A Streamlit dashboard for synthetic data generation, project analytics, machine learning model training, explainability, and smart project quotation for construction/engineering projects.

---

## Features

- **Synthetic Data Generation:** Create realistic project and timesheet data for experimentation.
- **Deep Dive Analytics:** Visualize project effort, complexity, and timelines.
- **Data Quality Checks:** Automated checks for missing values, outliers, and metadata issues.
- **Model Training & Tuning:** Train and compare Linear Regression, Random Forest, Gradient Boosting (with Quantile Regression), and Stacking Regressor models.
- **Explainability (XAI):** SHAP-based feature importance and interaction analysis.
- **Smart Quotation:** Predict project effort, cost, and generate resource-loaded schedules with risk and drift analysis.

---

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd Synthetic\ Data\ Generator
```

### 2. Create a Virtual Environment

```sh
python -m venv venv
# Activate:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```sh
pip install streamlit pandas numpy plotly scikit-learn shap matplotlib scipy
```

---

## Running the App

```sh
streamlit run app5.py
```

The app will open in your browser (usually at http://localhost:8501).

---

## Usage Guide

### Data Source

- **Upload CSV:** Use the sidebar to upload your own granular project data (must include columns like `date_logged`, `hours_worked`, `task_category`).
- **Or Generate Synthetic Data:** Use the sliders to generate a synthetic dataset for demo/testing.

### Tabs Overview

#### 1. Deep Dive Analytics
- Explore project effort, complexity, and timelines
- Visualize correlations and check data quality
- View S-curve progress and task breakdown for individual projects
- Analyze feature relationships with interactive heatmaps

#### 2. AI Model Engine
- Select and train ML models (Linear Regression, Random Forest, Gradient Boosting, Stacking Regressor)
- Tune hyperparameters with Randomized Search CV
- Compare model performance on test sets
- Review predicted vs. actual effort charts
- Approve models for production use (R² > 0.8 threshold)
- Simulate MLOps workflows with model versioning

#### 3. Model Explainability (XAI)
- Analyze feature importance and interactions using SHAP values
- View global and local model behavior
- Compare SHAP and ALE (Accumulated Local Effects) explanations
- Identify strongest feature interactions with TreeSHAP
- Understand why the model makes specific predictions

#### 4. Smart Quotation
- Input new project specifications to estimate effort and cost
- View data drift warnings if inputs deviate from training data
- Generate resource-loaded schedules and Gantt charts
- Apply profit markup and adjust team size dynamically
- Calculate financial quotes with risk margins (quantile models only)

---

## File Structure

```
Synthetic Data Generator/
├── app5.py                          # Main Streamlit application
├── README.md                        # This file
├── cropped-DaskanLogo.png          # Logo for branding
└── (optional) data files (*.csv)   # Example datasets
```

---

## Key Features Explained

### Model Training

- **Quantile Regression:** Available with Gradient Boosting to generate confidence intervals (Q05, Q50, Q95)
- **Stacking Regressor:** Combines Random Forest and Gradient Boosting with Ridge meta-learner
- **Hyperparameter Tuning:** Uses Randomized Search CV for optimizing RF and GBR models
- **Linear Regression:** Closed-form solution, no tuning required

### Explainability with SHAP

- **Beeswarm Summary Plot:** Shows global feature importance with color-coded impact direction
- **Bar Plot Importance:** Ranked feature importance across all predictions
- **Dependence Plots:** Reveals how individual features affect predictions with interaction coloring
- **Feature Interactions:** Displays strongest feature pairs using TreeSHAP
- **ALE Plots:** Alternative view that accounts for feature correlations (requires pyALE)

### Data Drift Detection

- Compares input project parameters against training data distribution
- Flags inputs >2 standard deviations from mean as unreliable
- Provides reliability warnings for out-of-distribution predictions
- Prevents over-confident estimates on unusual projects

### Financial Planning & Quotation

- Dynamically calculates project duration based on team size
- Generates conceptual task breakdown from historical data
- Applies profit markup to base cost for final quote
- Produces Gantt chart visualization for timeline planning
- Calculates risk margins using quantile regression (Q05, Q50, Q95)

### MLOps Simulation

- Model approval workflow with R² thresholds (production-ready if R² > 0.8)
- Model versioning with timestamp-based tracking
- Test set performance metrics (R², MAE)
- Approved model tracking for quotation use

---

## Configuration Constants

```python
DASKAN_GREEN = "#049449"                    # Brand color
AVG_HOURLY_RATE_CAD = 115                   # Default hourly rate (CAD)
HOURS_PER_WEEK = 30                         # Per engineer (productive hours)

# Model feature sets
NUM_FEATURES = [
    'surface_area_m2',
    'num_levels',
    'building_height_m',
    'num_units',
    'project_duration_days',
    'num_revisions'
]

CAT_FEATURES = [
    'project_type',
    'scope_category',
    'material_type'
]
```

---

## Data Format Requirements

### Input CSV Format (Granular Timesheet Data)

Required columns:
- `log_id` - Unique timesheet entry identifier
- `project_id` - Project identifier (e.g., "P-2022-001")
- `employee_id` - Employee/engineer identifier
- `date_logged` - Date of work log (YYYY-MM-DD or DD/MM/YYYY)
- `task_category` - Type of task (e.g., "Design", "Drafting", "Meeting")
- `hours_worked` - Hours spent on task
- `project_type` - Type of project (Residential, Commercial, Institutional, Industrial)
- `material_type` - Primary material (Wood, Steel, Concrete, Mixed)
- `surface_area_m2` - Building surface area
- `num_levels` - Number of floors/levels
- `building_height_m` - Total building height
- `num_units` - Number of units
- `start_date` - Project start date
- `end_date` - Project end date

### Optional Columns

- `scope_category` - Project scope classification
- `project_duration_days` - Planned duration
- `num_revisions` - Number of design revisions

---

## Troubleshooting

### "Import 'pyALE' could not be resolved"
- ALE plots are optional. The app falls back gracefully if pyALE is unavailable
- To enable ALE functionality: `pip install pyALE`

### Data Upload Issues
- Ensure CSV has required columns: `date_logged`, `hours_worked`, `task_category`, `project_id`
- Date columns should be in standard format (YYYY-MM-DD or DD/MM/YYYY)
- Use UTF-8 encoding when saving CSV files

### Model Training Fails
- Verify dataset has at least 20 rows with complete target values (`total_project_effort`)
- Check that all NUM_FEATURES and CAT_FEATURES columns exist in your data
- Ensure no critical columns have >50% missing values

### SHAP Computation is Slow
- SHAP values are cached to improve performance
- Clear Streamlit cache if needed: `streamlit cache clear`
- First run may be slower; subsequent runs will use cached values

### Quantile Regression Not Available
- Quantile Regression checkbox is disabled for non-Gradient Boosting models
- Only available with Gradient Boosting Regressor

---

## Performance Notes

- **Synthetic Data Generation:** Creates 50-500 realistic projects with timesheet entries in seconds
- **Model Training:** Randomized Search CV completes in 30-60 seconds for typical datasets
- **SHAP Calculation:** First SHAP computation takes 10-30 seconds; subsequent runs use cached values
- **Quotation:** Real-time predictions (<1 second) once model is trained

---

## Business Rules & Thresholds

| Rule | Value | Description |
|------|-------|-------------|
| Production R² Threshold | > 0.80 | Model must exceed this to be marked as production-ready |
| Data Drift Detection | ±2σ | Flags inputs outside 2 standard deviations from training mean |
| Risk Margin Calculation | Q95 - Q05 / 2 | Used only with Quantile Regression models |
| Minimum Project Size | 10 hours | Synthetic data generator ensures minimum effort floor |
| Winter Adjustment | Dec-Mar | Projects with logs in winter months are flagged |

---

## Example Workflow

1. **Generate or Upload Data**
   - Use synthetic data generator (50-500 projects) or upload your granular CSV

2. **Explore Analytics**
   - Check correlation heatmaps and project distributions
   - Verify data quality checks pass

3. **Train Models**
   - Select algorithm (RF, GBR with/without Quantile, Stacking, or LR)
   - Run Training & Tuning
   - Review Test R² and MAE metrics

4. **Approve Model**
   - If R² > 0.8, approve for production use
   - Model gets versioned with timestamp

5. **Generate Quotes**
   - Input new project specifications
   - Review data drift warnings
   - See effort estimate with risk margins
   - Adjust team size for timeline
   - View final quote with markup applied
   - Download Gantt chart for client presentation

6. **(Optional) Explain Predictions**
   - Use SHAP plots to explain model decisions
   - Share feature importance with stakeholders

---

## Advanced Features

### Stacking Regressor Ensemble
- Combines Random Forest and Gradient Boosting as base learners
- Uses Ridge regression as meta-model
- Provides balanced predictions by leveraging strengths of both algorithms

### Quantile Regression
- Generates three predictions: Q05 (pessimistic), Q50 (median), Q95 (optimistic)
- Provides confidence intervals for financial risk assessment
- Useful for bid/quote strategy

### SHAP Interaction Analysis
- Identifies feature pairs that strongly interact in the model
- Helps understand non-linear relationships
- Supports better domain expert validation

---

## Limitations & Future Enhancements

### Current Limitations
- Assumes projects are independent (no dependencies between tasks)
- Synthetic data uses simplified distributions (improvements possible)
- No multi-year trend analysis or seasonal decomposition
- Team skill levels not explicitly modeled

### Potential Enhancements
- Historical model versioning and performance tracking
- Integration with project management tools (Jira, Asana)
- Advanced ensemble methods (boosting, blending)
- Bayesian uncertainty quantification
- Real-time model retraining pipeline
- Cost breakdown by discipline/department

---

## Requirements

- Python 3.8+
- Streamlit 1.0+
- scikit-learn 1.0+
- SHAP 0.40+
- Plotly 5.0+
- Pandas 1.3+
- NumPy 1.20+
- Matplotlib 3.3+
- SciPy 1.7+
- (Optional) pyALE 1.0+ (for ALE plots)

---

## Support & Contact

For questions, bug reports, or feature requests, please contact the project maintainer or refer to the Daskan Intelligence documentation.

---

*Built with Streamlit, scikit-learn, SHAP, and Plotly for enterprise engineering project estimation and resource planning.*
