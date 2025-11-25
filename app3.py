import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge 
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor 
from sklearn.metrics import mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Daskan Intelligence | AI Project Estimator",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #049449;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: #EAEAEA;
        border-radius: 6px 6px 0px 0px;
        gap: 1px;
        padding-top: 15px;
        padding-bottom: 15px;
        font-weight: 600;
        color: #333333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 4px solid #049449;
        color: #049449;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. ADVANCED DATA GENERATOR (Granular + Summary) ---
@st.cache_data
def generate_granular_synthetic_data(num_projects=100, seed=42):
    np.random.seed(seed)
    
    # 1. Project Metadata
    project_types = ['Residential', 'Commercial', 'Institutional', 'Industrial']
    materials = ['Wood', 'Steel', 'Concrete', 'Mixed']
    projects = []
    project_effort_map = {} 
    
    for i in range(num_projects):
        p_id = f"P-2022-{str(i+1).zfill(3)}"
        p_type = np.random.choice(project_types, p=[0.25, 0.35, 0.15, 0.25])
        
        if p_type == 'Residential': area = np.random.randint(100, 600); levels = np.random.randint(1, 4)
        elif p_type == 'Commercial': area = np.random.randint(500, 2000); levels = np.random.randint(2, 8)
        else: area = np.random.randint(1000, 5000); levels = np.random.randint(1, 15)
            
        start_month = np.random.randint(1, 12)
        start_date = datetime(2022, start_month, 1) + timedelta(days=np.random.randint(0, 28))
        
        # Calc Effort (Hidden Truth)
        base_effort = (area * 0.1) + (levels * 20)
        if p_type in ['Institutional', 'Industrial']: base_effort *= 1.5
        total_hours_est = int(np.random.normal(base_effort, base_effort * 0.15))
        total_hours_est = max(10, total_hours_est)
        project_effort_map[p_id] = total_hours_est
        
        estimated_weeks = max(1, total_hours_est / 30) 
        end_date = start_date + timedelta(days=int(estimated_weeks * 7) + np.random.randint(5, 20))

        projects.append({
            'project_id': p_id,
            'project_type': p_type,
            'material_type': np.random.choice(materials),
            'surface_area_m2': area,
            'num_levels': levels,
            'start_date': start_date,
            'end_date': end_date
        })
        
    df_projects = pd.DataFrame(projects)
    
    # 2. Detailed Timesheets
    timesheet_entries = []
    log_id_counter = 1
    
    for _, proj in df_projects.iterrows():
        p_id = proj['project_id']
        total_hours = project_effort_map[p_id]
        num_entries = np.random.randint(5, 60)
        avg_entry_hours = total_hours / num_entries
        date_range_days = (proj['end_date'] - proj['start_date']).days
        
        for _ in range(num_entries):
            hours = round(np.random.normal(avg_entry_hours, 1.5), 2)
            if hours <= 0.25: hours = 0.5
            random_days = np.random.randint(0, max(1, date_range_days))
            log_date = proj['start_date'] + timedelta(days=random_days)
            
            timesheet_entries.append({
                'log_id': log_id_counter,
                'project_id': p_id,
                'employee_id': f"EMP-{np.random.randint(1, 12)}",
                'date_logged': log_date,
                'task_category': np.random.choice(['Design', 'Calculation', 'Drafting', 'Meeting', 'Site Visit'], p=[0.25, 0.25, 0.3, 0.1, 0.1]),
                'hours_worked': hours
            })
            log_id_counter += 1
            
    df_timesheets = pd.DataFrame(timesheet_entries)
    
    # 3. Merge
    df_master = pd.merge(df_timesheets, df_projects, on='project_id', how='left')
    df_master['date_logged'] = pd.to_datetime(df_master['date_logged'])
    return df_master

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2682/2682690.png", width=80) 
    st.markdown("### 🏗️ Daskan Intelligence")
    st.caption("v4.0 - Full Granular Analysis")
    st.divider()
    
    st.subheader("Data Source")
    uploaded_file = st.file_uploader("Upload 'daskan_full_training_data.csv' (Granular)", type=['csv'])
    
    st.divider()
    st.subheader("⚙️ Simulation")
    random_seed = st.slider("Seed", 1, 100, 42)
    sample_size = st.slider("Projects to Generate", 20, 200, 87)

# --- DATA LOADING & PROCESSING ---
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        if 'date_logged' in raw_df.columns:
            raw_df['date_logged'] = pd.to_datetime(raw_df['date_logged'])
        data_source = "Uploaded Granular Data"
    except Exception as e:
        st.error(f"Error: {e}")
        raw_df = generate_granular_synthetic_data(sample_size, random_seed)
        data_source = "Synthetic Data (Fallback)"
else:
    raw_df = generate_granular_synthetic_data(sample_size, random_seed)
    data_source = f"Synthetic Data ({sample_size} Projects)"

# --- SEPARATE ANALYTICS (LOGS) VS MODELING (SUMMARY) ---
# 1. df_analytics: The full granular dataset (Logs)
df_analytics = raw_df.copy()

# 2. df_modeling: Aggregated to one row per project
if 'log_id' in df_analytics.columns or 'date_logged' in df_analytics.columns:
    # It is granular, need to aggregate
    df_modeling = df_analytics.groupby('project_id').agg({
        'surface_area_m2': 'first',
        'num_levels': 'first',
        'project_type': 'first',
        'material_type': 'first',
        'hours_worked': 'sum' # The Target Variable
    }).rename(columns={'hours_worked': 'total_project_effort'}).reset_index()
    
    # Re-calculate derived features for modeling
    df_modeling['floor_area_ratio'] = df_modeling['surface_area_m2'] / df_modeling['num_levels']
    
    # Calculate Complexity Index
    def get_complexity(row):
        mat_factor = {'Mixed':2.0, 'Steel':1.5, 'Concrete':1.2}.get(row['material_type'], 1.0)
        c_idx = (row['num_levels']**1.2) * (row['surface_area_m2']/row['num_levels']) / 500 * mat_factor
        return max(0.5, c_idx)
    
    df_modeling['complexity_index'] = df_modeling.apply(get_complexity, axis=1)
    
    # Winter flag (Approximation: if project has logs in Jan/Feb)
    winter_projs = df_analytics[df_analytics['date_logged'].dt.month.isin([12,1,2,3])]['project_id'].unique()
    df_modeling['is_winter'] = df_modeling['project_id'].isin(winter_projs).astype(int)
    
else:
    # It is already summary data
    df_modeling = raw_df.copy()
    df_analytics = None # Cannot do deep dive

# --- MAIN APP ---
st.title("Structural Project Analytics Dashboard")
st.markdown(f"**Current Data Source:** `{data_source}` | **Mode:** `{'Granular Analysis Available' if df_analytics is not None else 'Summary Only'}`")

tabs = st.tabs(["📊 Deep Dive Analytics", "🧹 Data Quality", "🧠 AI Model Engine", "💼 Smart Quotation"])

# ----------------------------------------------------------------------
# TAB 1: ANALYTICS (Updated with S-Curve and Task Breakdown)
# ----------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Historical Project Insights")
    
    # Row 1: High Level
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Macro View: Effort vs. Scale")
        fig = px.scatter(df_modeling, x='surface_area_m2', y='total_project_effort', color='project_type', size='num_levels',
                         hover_data=['complexity_index'], title="Project Effort vs Area")
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Project Mix")
        fig_pie = px.pie(df_modeling, names='project_type', title="Distribution by Type")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Row 2: Granular Deep Dive (Only if data exists)
    if df_analytics is not None:
        st.divider()
        st.markdown("### 🕵️ Granular 'Deep Dive' (Logs & Timesheets)")
        
        c_g1, c_g2 = st.columns([2, 1])
        
        with c_g1:
            st.subheader("Project 'Burn Rate' (S-Curve)")
            # Selector for project
            selected_proj = st.selectbox("Select Project to Analyze Timeline", df_modeling['project_id'].unique())
            
            # Filter and Process
            proj_data = df_analytics[df_analytics['project_id'] == selected_proj].copy()
            daily_df = proj_data.groupby('date_logged')['hours_worked'].sum().reset_index().sort_values('date_logged')
            daily_df['cumulative'] = daily_df['hours_worked'].cumsum()
            
            # Dual Axis Plot
            fig_burn = go.Figure()
            fig_burn.add_trace(go.Bar(x=daily_df['date_logged'], y=daily_df['hours_worked'], name='Daily Hours', marker_color='#e74c3c', opacity=0.4))
            fig_burn.add_trace(go.Scatter(x=daily_df['date_logged'], y=daily_df['cumulative'], name='Cumulative (S-Curve)', line=dict(color='#2c3e50', width=3)))
            
            fig_burn.update_layout(title=f"Progress Timeline: {selected_proj}", xaxis_title="Date", yaxis_title="Hours", hovermode="x unified")
            st.plotly_chart(fig_burn, use_container_width=True)
            
        with c_g2:
            st.subheader("Task Breakdown")
            # Filter by the SAME project type as the selected project to show relevant averages
            current_type = df_modeling[df_modeling['project_id'] == selected_proj]['project_type'].iloc[0]
            st.caption(f"Task Distribution for {current_type} projects")
            
            # Task dist for this project
            fig_task = px.pie(proj_data, names='task_category', values='hours_worked', hole=0.4, title=f"Tasks: {selected_proj}")
            st.plotly_chart(fig_task, use_container_width=True)
            
            # Employee Count
            st.metric("Unique Employees on Project", proj_data['employee_id'].nunique())


# ----------------------------------------------------------------------
# TAB 2: DATA QUALITY
# ----------------------------------------------------------------------
with tabs[1]:
    st.subheader("Dataset Statistics (Modeling Level)")
    st.dataframe(df_modeling.describe().T, use_container_width=True)
    
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.write("**Missing Values (Modeling DF)**")
        st.dataframe(df_modeling.isnull().sum(), use_container_width=True)
    with col_q2:
        if df_analytics is not None:
            st.write("**Granular Stats**")
            st.info(f"Total Log Entries: {len(df_analytics)}")
            st.info(f"Unique Employees: {df_analytics['employee_id'].nunique()}")
            st.info(f"Date Range: {df_analytics['date_logged'].min().date()} to {df_analytics['date_logged'].max().date()}")

# ----------------------------------------------------------------------
# TAB 3: AI ENGINE
# ----------------------------------------------------------------------
with tabs[2]:
    st.subheader("Train Prediction Model")
    
    # Setup
    X = df_modeling[['surface_area_m2', 'num_levels', 'floor_area_ratio', 'complexity_index', 'is_winter', 'project_type', 'material_type']]
    y = df_modeling['total_project_effort']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Preprocessor
    num_features = ['surface_area_m2', 'num_levels', 'floor_area_ratio', 'complexity_index', 'is_winter']
    cat_features = ['project_type', 'material_type']
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_features)
    ])
    
    # Controls
    col_m1, col_m2 = st.columns([1, 3])
    with col_m1:
        model_choice = st.selectbox("Algorithm", ["Gradient Boosting", "Random Forest"])
        use_quantile = st.checkbox("Use Quantile Regression (Risk Intervals)", value=True)
    
    with col_m2:
        if st.button("Train Model"):
            with st.spinner("Training..."):
                if model_choice == "Gradient Boosting" and use_quantile:
                    # Train Quantile (Low, Median, High)
                    common_params = {'n_estimators': 150, 'max_depth': 5, 'random_state': 42}
                    models = {}
                    for name, alpha in [('Q05', 0.05), ('Q50', 0.5), ('Q95', 0.95)]:
                        gbr = GradientBoostingRegressor(loss='quantile', alpha=alpha, **common_params)
                        models[name] = Pipeline([('prep', preprocessor), ('model', gbr)])
                        models[name].fit(X_train, y_train)
                    
                    st.session_state['models'] = models
                    st.session_state['mode'] = 'quantile'
                    
                    # Eval Median
                    preds = models['Q50'].predict(X_test)
                    r2 = r2_score(y_test, preds)
                    st.metric("Model Accuracy (R² - Median)", f"{r2:.3f}")
                    
                else:
                    # Train Point
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    pipe = Pipeline([('prep', preprocessor), ('model', model)])
                    pipe.fit(X_train, y_train)
                    st.session_state['models'] = pipe
                    st.session_state['mode'] = 'point'
                    
                    preds = pipe.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    st.metric("Model Accuracy (R²)", f"{r2:.3f}")
            
            st.success("Model Trained!")

# ----------------------------------------------------------------------
# TAB 4: SMART QUOTATION (With Granular Insights)
# ----------------------------------------------------------------------
with tabs[3]:
    st.subheader("Project Quotation & Resource Planner")
    
    if 'models' not in st.session_state:
        st.warning("Train model in Tab 3 first.")
    else:
        # Inputs
        c1, c2, c3, c4 = st.columns(4)
        i_area = c1.number_input("Area (m2)", 100, 5000, 800)
        i_levels = c2.number_input("Levels", 1, 20, 2)
        i_type = c3.selectbox("Type", df_modeling['project_type'].unique())
        i_mat = c4.selectbox("Material", df_modeling['material_type'].unique())
        
        # Calc Features
        c_idx = (i_levels**1.2)*(i_area/i_levels)/500 * ({'Mixed':2.0,'Steel':1.5}.get(i_mat,1.0))
        input_data = pd.DataFrame({
            'surface_area_m2':[i_area], 'num_levels':[i_levels], 'floor_area_ratio':[i_area/i_levels],
            'complexity_index':[max(0.5, c_idx)], 'is_winter':[0], 'project_type':[i_type], 'material_type':[i_mat]
        })
        
        if st.button("Generate Quote"):
            # 1. Prediction
            if st.session_state['mode'] == 'quantile':
                q05 = st.session_state['models']['Q05'].predict(input_data)[0]
                q50 = st.session_state['models']['Q50'].predict(input_data)[0]
                q95 = st.session_state['models']['Q95'].predict(input_data)[0]
                pred_val = q50
                risk_txt = f"90% Confidence Interval: {q05:.0f}h - {q95:.0f}h"
            else:
                pred_val = st.session_state['models'].predict(input_data)[0]
                risk_txt = "Point Estimate Only"
            
            # Display Main Prediction
            st.markdown(f"""
            <div class="metric-card">
                <h1 style="margin:0; color:#1A1A1A;">{pred_val:.0f} Hours</h1>
                <p style="margin:0; color:#049449; font-weight:bold;">Estimated Effort ({i_type})</p>
                <small>{risk_txt}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. Granular Breakdown (The "All Data" Feature)
            if df_analytics is not None:
                st.markdown("#### 📋 Recommended Task Breakdown")
                st.caption(f"Based on historical data for '{i_type}' projects, here is how you should allocate these {pred_val:.0f} hours:")
                
                # Get historical ratios for this project type
                type_logs = df_analytics[df_analytics['project_type'] == i_type]
                if not type_logs.empty:
                    dist = type_logs['task_category'].value_counts(normalize=True)
                    
                    # Create breakdown columns
                    cols = st.columns(len(dist))
                    for idx, (task, pct) in enumerate(dist.items()):
                        allocated_hours = pred_val * pct
                        with cols[idx]:
                            st.metric(task, f"{allocated_hours:.0f} h", f"{pct*100:.0f}%")
                    
                    # Visual Bar
                    breakdown_df = pd.DataFrame({'Task': dist.index, 'Hours': dist.values * pred_val})
                    fig_bd = px.bar(breakdown_df, x='Hours', y='Task', orientation='h', color='Task', text_auto='.0f')
                    fig_bd.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0), height=200)
                    st.plotly_chart(fig_bd, use_container_width=True)
                else:
                    st.warning("No historical granular data for this project type.")