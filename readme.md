
# 🧠 Daskan Structural Project Intelligence Dashboard: AI Effort Estimator

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

This project delivers a sophisticated, AI-driven dashboard for predicting the effort (in engineering hours) required for structural projects. Built upon the principles of advanced machine learning and explainability, it addresses the need for accurate resource planning, risk management, and transparent decision-making within engineering firms.

The application is developed using **Streamlit** and employs an **Ensemble Quantile Regression** approach to provide not just a single point estimate, but a **Prediction Interval** crucial for risk assessment and confident quotation. It also incorporates **Model Stacking** and MLOps principles like **Data Drift Monitoring**.

-----

## 📚 Table of Contents

- [Key Sophistication & Features](#-key-sophistication--features)
- [Installation and Setup](#️-installation-and-setup)
- [Usage Guide](#-usage-guide)
- [Methodology Highlights](#-methodology-highlights)
- [Support](#support)
- [License](#license)
- [Contact](#contact)

-----

## ✨ Key Sophistication & Features

This model and application incorporate several advanced techniques suitable for an academic thesis and enterprise deployment:

| Feature | Technical Focus | Business Value |
| :--- | :--- | :--- |
| **Quantile Regression** | Uses **Gradient Boosting Regressors (GBR)** to train three models (Q05, Q50, Q95). | Provides a **90% Prediction Interval** (Confidence Range) for risk-managed project quoting. |
| **Model Stacking** | Implements the **`StackingRegressor`** combining GBR/RF with a Ridge meta-model. | Improves predictive accuracy and robustness by leveraging the strengths of multiple base models. |
| **SHAP Explainability (XAI)** | Integrates **SHapley Additive exPlanations (SHAP)** plots (Beeswarm, Dependence). | Offers transparent, model-agnostic insight into feature influence, facilitating stakeholder trust. |
| **Data Drift Monitoring** | Compares new project inputs against the historical training data's mean/std. | Flags when a new project is **out-of-distribution**, preventing unreliable predictions in MLOps and maintaining model integrity. |
| **Automated HPO** | Implements **`RandomizedSearchCV`** within the training pipeline. | Automatically finds optimal hyperparameters, ensuring the model is not relying on arbitrary settings and maximizing predictive accuracy. |
| **Advanced Feature Engineering** | Calculates a custom **Complexity Index** based on levels, area, and material type. | Encodes essential domain knowledge into a single powerful feature, increasing model performance and interpretability. |
| **Deployment Simulation** | MLOps simulation for **Model Approval/Versioning** based on R² thresholds. | Creates a professional workflow for model promotion to a "live production" environment. |

-----

## ⚙️ Installation and Setup

To run the application locally, you will need Python 3.8+ and the following packages.

### 1. Prerequisites

Ensure you have Python installed.

### 2. Create and Activate Environment

```bash
# Create a new virtual environment (recommended)
python -m venv venv
# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
````

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

The dashboard is organized into four main tabs to facilitate a structured data science and planning workflow:

1.  **📊 Deep Dive Analytics:** Explore data distributions, correlations, and feature relationships. Includes the **Feature Correlation Heatmap** and granular **S-Curve** analysis.
2.  **🧠 AI Model Engine:** This is the core modeling section.
      * **Configure:** Select your algorithm (**Gradient Boosting**, **Random Forest**, or **Stacking Regressor**). Check **"Use Quantile Regression"** for risk intervals.
      * **Train:** Click **"Train & Tune Model"** to run the pipeline, including `RandomizedSearchCV` for optimization.
      * **Evaluation:** Review the **Test Set Error Analysis** (Predicted vs. Actual plot) and the **MLOps Simulation** for Model Approval.
3.  **📈 Model Explainability (XAI):** Dive into **SHAP values** for global and local transparency.
      * Review the **Global Feature Summary (Beeswarm)** to see which features impact predictions the most.
      * Use the **Feature Dependence Plot** to visualize feature interactions (e.g., non-linear effects of complexity).
4.  **💼 Smart Quotation:** The final decision support system.
      * Input new project parameters and financial settings (Rate, Markup).
      * Run the AI model to get the central prediction and the **90% Confidence Interval (Q05-Q95)**.
      * Check the **Data Drift & Reliability Check** for warnings.
      * Adjust the team size to generate a **Resource-Loaded Schedule** (conceptual Gantt chart).

-----

## 📌 Methodology Highlights

The application supports three powerful estimation methods, with a focus on **Quantile Regression** for uncertainty quantification:

1.  **Gradient Boosting Quantile Regression:** Trains three distinct GBR models using the `loss='quantile'` parameter:

      * **Q05 (5th Percentile):** The lower bound (best-case effort).
      * **Q50 (50th Percentile / Median):** The best single point estimate.
      * **Q95 (95th Percentile):** The upper bound (worst-case, risk-managed effort).
        This provides a statistically rigorous framework for uncertainty quantification, translating machine learning output into actionable business intelligence.

2.  **Stacking Regressor:** Combines different model types (e.g., Random Forest and Gradient Boosting) to leverage their respective strengths, with a meta-model (Ridge Regression) learning how to best combine their predictions.

3.  **Tuned Point Estimate:** Standard, highly accurate point prediction via hyperparameter-optimized Random Forest or Gradient Boosting Regressor.

-----

## 🛠️ Support

If you encounter any issues or have questions, please open an [issue](https://github.com/WilsonZangue/Structural-Engineering/issues) on GitHub.

## 📄 License

This project is licensed under the MIT License.

## 📬 Contact

For questions, suggestions, or collaboration, contact [Wilson Zangue](https://github.com/WilsonZangue) via GitHub.

```
```
