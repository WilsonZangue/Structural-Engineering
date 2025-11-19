Workflow Optimization
# 🏗️ Daskan Intelligence | Structural-Engineering

This project delivers a sophisticated, AI-driven dashboard for predicting the effort (in engineering hours) required for structural projects. Built upon the principles of advanced machine learning and explainability, it addresses the need for accurate resource planning, risk management, and transparent decision-making within engineering firms.

The application is developed using **Streamlit** and employs an **Ensemble Quantile Regression** approach to provide not just a single point estimate, but a **Prediction Interval** crucial for risk assessment and confident quotation.

-----

## ✨ Key Sophistication & Features

This model and application incorporate several advanced techniques suitable for an academic thesis:

| Feature | Technical Focus | Business Value |
| :--- | :--- | :--- |
| **Quantile Regression** | Uses **Gradient Boosting Regressors (GBR)** to train three models (Q05, Q50, Q95). | Provides a **90% Prediction Interval** (Confidence Range) for risk-managed project quoting. |
| **Automated HPO** | Implements **`RandomizedSearchCV`** within the training pipeline. | Automatically finds optimal hyperparameters, ensuring the model is not relying on arbitrary settings and maximizing predictive accuracy. |
| **SHAP Explainability (XAI)** | Integrates **SHapley Additive exPlanations (SHAP)** plots. | Offers global, transparent insight into which features (e.g., complexity index, area) drive the effort prediction, moving beyond simple accuracy metrics. |
| **Advanced Feature Engineering** | Calculates a custom **Complexity Index** based on levels, area, and material type. | Encodes essential domain knowledge into a single powerful feature, increasing model performance and interpretability. |
| **Data Quality Pipeline** | Uses **`ColumnTransformer`** and **`Pipeline`** for robust data cleaning (Imputation, Scaling, One-Hot Encoding). | Prevents data leakage and ensures that the model is trained on properly preprocessed data. |

-----

## ⚙️ Installation and Setup

To run the application locally, you will need Python 3.8+ and the following packages.

### 1\. Prerequisites

Ensure you have Python installed.

### 2\. Create and Activate Environment

```bash
# Create a new virtual environment (recommended)
python -m venv venv
# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3\. Install Dependencies

Install all required Python libraries.

```bash
pip install streamlit pandas numpy plotly scikit-learn shap matplotlib
```

### 4\. Run the Application

Save the provided Python code as `app.py` and execute it from your terminal:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser (usually at `http://localhost:8501`).

-----

## 💻 Usage Guide

The dashboard is organized into four main tabs to facilitate a structured data science workflow:

1. **📊 Deep Dive Analytics:** Explore data distributions, correlations, and feature relationships (`RQ1`).
2. **🧹 Data Quality Check:** Review missing data report and understand the automated imputation/scaling strategy (`RQ2`).
3. **🧠 AI Model Engine:** This is the core modeling section (`RQ3`).
      * **Configure:** Select **Gradient Boosting Regressor** and **Prediction Interval (90% Quantile)** for the most sophisticated results.
      * **Optimize:** Check **"Perform Hyperparameter Optimization (Slow)"** to execute `RandomizedSearchCV`.
      * **Train:** Click **"Train Model(s) on Current Data"**.
      * **Evaluate:** Review performance metrics (R², MAE, **PICP**), Residual Plot, and the **SHAP Summary Plot** for XAI.
4. **💼 Decision Support System:** Input new project parameters to get real-time effort predictions, including the critical 90% confidence range, and assess workforce planning (`RQ4`).

-----

## 📌 Methodology Highlights

The most advanced and thesis-relevant methodology is the use of **Gradient Boosting Quantile Regression** (specifically using the `loss='quantile'` parameter in `GradientBoostingRegressor`).

The application trains three distinct GBR models:

  **Q05 (5th Percentile):** The lower bound of the prediction interval.
  **Q50 (50th Percentile / Median):** The best single point estimate (used for central prediction and overall evaluation).
  **Q95 (95th Percentile):** The upper bound of the prediction interval, representing the worst-case, risk-managed effort.

This approach provides a statistically rigorous framework for uncertainty quantification, directly translating machine learning output into actionable business intelligence for project managers.
