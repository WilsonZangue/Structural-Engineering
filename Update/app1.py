import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Daskan Inc. Estimator",
    page_icon="",
    layout="wide"
)

# --- SIDEBAR: THESIS CONTEXT ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2682/2682690.png", width=100)
    st.title("Thesis Project Info")
    st.info(
        """
        **Topic:** Predictive Modeling for Structural Engineering Effort
        **Author:** [Your Name]
        **Client:** Group Daskan Inc.
        **Methodology:** CRISP-DM
        """
    )
    st.header("User Controls")
    random_seed = st.slider("Random Seed (Simulation)", 1, 100, 42)

# --- 1. DATA GENERATION (Cached for Performance) ---
@st.cache_data
def generate_data(seed):
    np.random.seed(seed)
    num_projects = 200
    
    project_types = ['Residential', 'Commercial', 'Institutional', 'Industrial']
    materials = ['Wood', 'Steel', 'Concrete', 'Mixed']
    
    data = []
    for i in range(num_projects):
        p_type = np.random.choice(project_types, p=[0.5, 0.3, 0.1, 0.1])
        
        # Logic: Size depends on type
        if p_type == 'Residential':
            area = np.random.randint(100, 600)
            levels = np.random.randint(1, 4)
        elif p_type == 'Commercial':
            area = np.random.randint(500, 2000)
            levels = np.random.randint(2, 8)
        else:
            area = np.random.randint(1000, 5000)
            levels = np.random.randint(1, 15)
            
        is_winter = np.random.choice([0, 1], p=[0.7, 0.3])
        
        # Engineering the Target (Effort) with some noise
        base_effort = (area * 0.12) + (levels * 25)
        if is_winter: base_effort *= 1.2  # Winter penalty
        if p_type in ['Institutional', 'Industrial']: base_effort *= 1.5 # Complexity penalty
        
        actual_effort = abs(np.random.normal(base_effort, base_effort * 0.15))
        
        data.append({
            'project_type': p_type,
            'material_type': np.random.choice(materials),
            'surface_area_m2': area,
            'num_levels': levels,
            'is_winter': is_winter,
            'floor_area_ratio': area / levels,
            'total_project_effort': round(actual_effort, 2)
        })
        
    return pd.DataFrame(data)

df = generate_data(random_seed)

# --- MAIN DASHBOARD ---
st.title(" Daskan Inc. - Intelligent Effort Estimation System")
st.markdown("This system uses **Machine Learning** to predict project hours based on historical structural data.")

# Create Tabs for the Workflow
tab1, tab2, tab3 = st.tabs([" Data Exploration (EDA)", " Model Training", " Prediction Interface"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Overview")
        st.dataframe(df.head(10))
        st.caption(f"Total Records: {len(df)} Projects")
        
    with col2:
        st.subheader("Correlation Analysis (RQ1)")
        fig, ax = plt.subplots()
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.markdown("**Insight:** `surface_area_m2` has the strongest correlation with effort.")

    st.subheader("Effort vs. Surface Area (by Project Type)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.scatterplot(data=df, x='surface_area_m2', y='total_project_effort', hue='project_type', palette='viridis', ax=ax2)
    st.pyplot(fig2)

# --- TAB 2: MODELING ---
with tab2:
    st.header("Model Training & Evaluation")
    
    # Split Data
    X = df.drop('total_project_effort', axis=1)
    y = df['total_project_effort']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define Pipelines
    numeric_features = ['surface_area_m2', 'num_levels', 'floor_area_ratio']
    categorical_features = ['project_type', 'material_type']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Model Options
    model_choice = st.radio("Select Model Algorithm:", ["Linear Regression (Baseline)", "Random Forest (Advanced)"], horizontal=True)
    
    if model_choice == "Linear Regression (Baseline)":
        model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    else:
        model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100))])
    
    if st.button("Train Model"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE (Mean Absolute Error)", f"{mae:.1f} hrs", delta_color="inverse")
        col2.metric("R² Score (Accuracy)", f"{r2:.3f}")
        col3.metric("Success Criteria (>0.70)", "Passed " if r2 > 0.70 else "Failed ")
        
        # Plot Real vs Predicted
        fig3, ax3 = plt.subplots()
        sns.regplot(x=y_test, y=preds, ax=ax3, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        ax3.set_xlabel("Actual Hours")
        ax3.set_ylabel("Predicted Hours")
        st.pyplot(fig3)
        
        # Save model to session state for Tab 3
        st.session_state['trained_model'] = model
        st.success("Model Trained Successfully!")

# --- TAB 3: PREDICTION ---
with tab3:
    st.header("New Project Effort Estimator")
    
    if 'trained_model' not in st.session_state:
        st.warning(" Please train a model in the 'Model Training' tab first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            input_area = st.number_input("Surface Area (m²)", min_value=50, max_value=10000, value=500)
            input_levels = st.number_input("Number of Levels", min_value=1, max_value=50, value=2)
            input_type = st.selectbox("Project Type", df['project_type'].unique())
            
        with col2:
            input_material = st.selectbox("Material Type", df['material_type'].unique())
            input_season = st.radio("Is this a Winter Project?", ["No", "Yes"])
            is_winter_val = 1 if input_season == "Yes" else 0
            
        # Create Input DataFrame
        input_data = pd.DataFrame({
            'project_type': [input_type],
            'material_type': [input_material],
            'surface_area_m2': [input_area],
            'num_levels': [input_levels],
            'is_winter': [is_winter_val],
            'floor_area_ratio': [input_area / input_levels]
        })
        
        if st.button("Estimate Effort"):
            prediction = st.session_state['trained_model'].predict(input_data)[0]
            
            st.divider()
            st.subheader(f" Estimated Effort: {prediction:.1f} Hours")
            
            # Risk Classification (Logic: If > 200 hours, it's 'High Risk')
            if prediction > 200:
                st.error(" High Risk Project: Requires Senior Supervision")
            else:
                st.success(" Standard Project: Routine Supervision")
            
            # Explanation
            st.caption("Estimation based on trained Random Forest model parameters.")
