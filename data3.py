import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(87)

def generate_synthetic_data(num_projects=87):
    """
    Generates synthetic construction project data and timesheet logs.
    The FINAL training data CSV includes ALL requested Project-Level and 
    Log-Level features by aggregating and then joining back to the raw logs.
    """
    print("👷 Generating Daskan Inc. FULL ExplorationData (All Features)...")
    
    # --- 1. PROJECT METADATA, DURATION LOGIC, AND TASK MAPPING ---
    project_types = ['Residential', 'Commercial', 'Institutional', 'Industrial']
    materials = ['Wood', 'Steel', 'Concrete', 'Mixed']
    
    # Define task probability weights based on Project Type
    task_weights = {
        'Residential':     {'Design': 0.30, 'Calculation': 0.10, 'Drafting': 0.40, 'Meeting': 0.10, 'Site Visit': 0.10},
        'Commercial':      {'Design': 0.25, 'Calculation': 0.20, 'Drafting': 0.30, 'Meeting': 0.15, 'Site Visit': 0.10},
        'Institutional':   {'Design': 0.20, 'Calculation': 0.30, 'Drafting': 0.20, 'Meeting': 0.20, 'Site Visit': 0.10},
        'Industrial':      {'Design': 0.15, 'Calculation': 0.35, 'Drafting': 0.15, 'Meeting': 0.25, 'Site Visit': 0.10},
    }
    task_categories = list(task_weights['Residential'].keys())

    # Define Subtasks for each Main Task Category
    subtask_map = {
        'Design': [
            'Architectural Layout', 'Structural Concept', 'MEP System Planning', 
            'Facade/Exterior Detailing', 'Material Specification Review'
        ],
        'Calculation': [
            'Structural Analysis (Beam/Column Sizing)', 'Load Calculations (Wind/Seismic)', 
            'Energy Modeling', 'Cost Estimation Review', 'Permit Fee Calculation'
        ],
        'Drafting': [
            'Foundation Plan Updates', 'Section Drawings', 'Detail Sheet Generation', 
            'Markup Cleanup', '3D Model Adjustments'
        ],
        'Meeting': [
            'Client Design Review', 'Subcontractor Coordination', 'Internal Team Sync', 
            'Permitting Authority Review', 'BIM Clash Detection Session'
        ],
        'Site Visit': [
            'Existing Conditions Survey', 'Progress Inspection', 'Quality Check (QA/QC)', 
            'RFI Clarification', 'Punch List Generation'
        ]
    }
    
    projects = []
    project_effort_map = {} 
    
    project_counter = {2022: 0, 2023: 0, 2024: 0} 
    start_years = np.random.choice([2022, 2023, 2024], size=num_projects, p=[0.3, 0.4, 0.3])
    
    for start_year in start_years:
        
        # 1.1 Determine Start Date
        start_month = np.random.randint(1, 13) 
        start_date = datetime(start_year, start_month, 1) + timedelta(days=np.random.randint(0, 28))
        
        # 1.2 Generate Project ID based on Year and Sequence
        project_counter[start_year] += 1
        p_id = f"P-{start_year}-{str(project_counter[start_year]).zfill(3)}"
        
        # 1.3 Determine Physical Size & Complexity
        p_type = np.random.choice(project_types, p=[0.25, 0.35, 0.15, 0.25])
        
        if p_type == 'Residential':
            area = np.random.randint(100, 600)
            levels = np.random.randint(1, 4)
            num_units = np.random.randint(1, 10)
            height = levels * np.random.normal(3.2, 0.3)
            scope_cat = 'New Build - Low Density'
        elif p_type == 'Commercial':
            area = np.random.randint(500, 2000)
            levels = np.random.randint(2, 8)
            num_units = 1
            height = levels * np.random.normal(4.0, 0.5)
            scope_cat = 'Tenant Improvement' if np.random.rand() < 0.3 else 'New Build - Mid Density'
        else: # Institutional/Industrial
            area = np.random.randint(1000, 5000)
            levels = np.random.randint(1, 15)
            num_units = 1
            height = levels * np.random.normal(4.5, 0.7)
            scope_cat = 'Complex Infrastructure'
            
        # 1.4 Calculate Target Effort (Hours)
        base_effort = (area * 0.1) + (levels * 20)
        if p_type in ['Institutional', 'Industrial']:
            base_effort *= 1.5
        
        total_hours_est = int(np.random.normal(base_effort, base_effort * 0.15))
        total_hours_est = max(10, total_hours_est)
        project_effort_map[p_id] = total_hours_est
        
        # 1.5 Calculate Planned End Date
        estimated_weeks = max(1, total_hours_est / 30) 
        expected_duration_days = int(estimated_weeks * 7) + np.random.randint(5, 20)
        planned_end_date = start_date + timedelta(days=expected_duration_days)

        # 1.6 Revision Data
        num_revisions = np.random.poisson(1.5) 
        revision_reason = 'None'
        if num_revisions > 0:
            revision_reason = np.random.choice(['Client Change', 'Code Conflict', 'Scope Creep', 'Site Condition'])

        projects.append({
            'project_id': p_id,
            'project_type': p_type,
            'material_type': np.random.choice(materials),
            'surface_area_m2': area,
            'num_levels': levels,
            'num_units': num_units,
            'building_height_m': round(height, 2),
            'num_revisions': num_revisions,
            'revision_reason': revision_reason,
            'planned_start_date': start_date, 
            'planned_end_date': planned_end_date,
            'expected_duration_days': expected_duration_days,
            'scope_category': scope_cat
        })
        
    df_projects_metadata = pd.DataFrame(projects)
    
    # --- 2. TIMESHEETS (Detailed Logs with Subtasks) ---
    timesheet_entries = []
    log_id_counter = 1
    
    for _, proj in df_projects_metadata.iterrows():
        p_id = proj['project_id']
        p_type = proj['project_type']
        
        total_hours = project_effort_map[p_id]
        
        actual_total_hours = total_hours * np.random.normal(1.05, 0.1) 
        actual_total_hours = max(10, actual_total_hours)
        
        num_entries = np.random.randint(int(actual_total_hours/5), int(actual_total_hours/0.5)) 
        
        start_dt_planned = proj['planned_start_date']
        actual_duration_days = int(proj['expected_duration_days'] * np.random.normal(1.1, 0.1))
        actual_end_dt = start_dt_planned + timedelta(days=actual_duration_days)
        date_range_days = (actual_end_dt - start_dt_planned).days
        
        # Structured Task Logic
        target_task_hours = {
            task: actual_total_hours * task_weights[p_type][task] 
            for task in task_categories
        }
        
        total_logs_for_project = 0
        logs_per_task = {}
        for task, target_hours in target_task_hours.items():
            num_logs = max(1, int(target_hours / np.random.normal(2.5, 0.5)))
            logs_per_task[task] = num_logs
            total_logs_for_project += num_logs

        final_logs_per_task = {}
        if total_logs_for_project > 0:
            for task, logs in logs_per_task.items():
                final_logs_per_task[task] = int(logs / total_logs_for_project * num_entries)
        
        logs_difference = num_entries - sum(final_logs_per_task.values())
        if logs_difference > 0:
            highest_task = max(final_logs_per_task, key=final_logs_per_task.get)
            final_logs_per_task[highest_task] += logs_difference

        # 4. Generate the log entries
        for task, num_logs in final_logs_per_task.items():
            
            if logs_per_task.get(task, 0) > 0:
                 avg_task_hours = target_task_hours.get(task, 0) / logs_per_task[task]
            else:
                 avg_task_hours = 2.0 
            
            for _ in range(num_logs):
                hours = round(np.random.normal(avg_task_hours, 1.0), 2)
                if hours <= 0.25: hours = 0.5
                
                random_days = np.random.randint(0, max(1, date_range_days))
                log_date = start_dt_planned + timedelta(days=random_days)
                
                subtask = np.random.choice(subtask_map[task])
                
                timesheet_entries.append({
                    'log_id': log_id_counter,
                    'project_id': p_id,
                    'employee_id': f"EMP-{np.random.randint(1, 12)}", 
                    'date_logged': log_date,
                    'task_category': task, 
                    'subtask_description': subtask, 
                    'hours_worked': hours
                })
                log_id_counter += 1
            
    df_timesheets = pd.DataFrame(timesheet_entries)
    
    # --- 3. AGGREGATE PROJECT-LEVEL METRICS FROM TIMESHEETS ---
    
    # Convert date_logged to datetime for calculation
    df_timesheets['date_logged'] = pd.to_datetime(df_timesheets['date_logged'])
    
    # 3.1 Calculate all project-level metrics requested by the user
    df_project_metrics = df_timesheets.groupby('project_id').agg(
        # Duration & Dates (Requested Variables)
        corrected_start_date=('date_logged', 'min'),
        corrected_end_date=('date_logged', 'max'),
        total_project_hours=('hours_worked', 'sum'),
        
        # Employee Metrics
        num_employees=('employee_id', 'nunique'),
        
        # Task Metrics (Requested Variables)
        design_hours_total=('hours_worked', lambda x: x[df_timesheets.loc[x.index, 'task_category'] == 'Design'].sum()),
        num_site_visits=('task_category', lambda x: (x == 'Site Visit').sum())
    ).reset_index()

    # 3.2 Calculate Duration and Employee Averages
    df_project_metrics['project_duration_days'] = (df_project_metrics['corrected_end_date'] - df_project_metrics['corrected_start_date']).dt.days
    df_project_metrics['actual_duration_days'] = df_project_metrics['project_duration_days'] # Using actual for raw duration
    df_project_metrics['avg_hours_per_employee'] = df_project_metrics['total_project_hours'] / df_project_metrics['num_employees']
    
    # 3.3 Time-based features extracted from the corrected start date
    df_project_metrics['month_started'] = df_project_metrics['corrected_start_date'].dt.month
    df_project_metrics['quarter'] = df_project_metrics['corrected_start_date'].dt.quarter
    
    def get_season_flag(month):
        if month in [12, 1, 2]: return 'Winter'
        if month in [3, 4, 5]: return 'Spring'
        if month in [6, 7, 8]: return 'Summer'
        return 'Autumn'
        
    df_project_metrics['season_flag'] = df_project_metrics['corrected_start_date'].dt.month.apply(get_season_flag)
    
    def check_holiday_overlap(start, end):
        # Checks if project spans the Nov 15th to Jan 15th holiday window
        span_start = datetime(start.year, 11, 15)
        span_end = datetime(start.year + 1, 1, 15) 
        if (start < span_end) and (end > span_start):
            return 1
        span_start_prev = datetime(start.year - 1, 11, 15)
        span_end_prev = datetime(start.year, 1, 15)
        if (start < span_end_prev) and (end > span_start_prev):
             return 1
        return 0

    df_project_metrics['holiday_period_flag'] = df_project_metrics.apply(
        lambda row: check_holiday_overlap(row['corrected_start_date'], row['corrected_end_date']), axis=1
    )


    # --- 4. CREATE FULL GRANULAR TRAINING DATASET (Final Merge) ---
    
    # 4.1 Join Project Metadata (V1) to Project Metrics (V2)
    df_context = pd.merge(df_projects_metadata, df_project_metrics, on='project_id', how='left')
    
    # 4.2 Calculate Floor Area Ratio (Planned Metric)
    df_context['floor_area_ratio'] = df_context['surface_area_m2'] / df_context['num_levels']
    
    # 4.3 Final Merge: Join Timesheets (Left) with the full Project Context (Right)
    # This results in the final dataset where every row is a log entry + all context.
    df_full_training = pd.merge(df_timesheets, df_context, on='project_id', how='left')
    
    # 4.4 Calculate Log-Level Supporting Variables
    df_full_training['is_winter_day'] = df_full_training['date_logged'].dt.month.isin([12, 1, 2, 3]).astype(int)
    # 'week' as supporting variable: week number of the year
    df_full_training['week'] = df_full_training['date_logged'].dt.isocalendar().week.astype(int)

    # --- C. Final Clean-up and Save ---
    
    # Convert all datetime objects to string format for CSV
    for col in df_full_training.columns:
        if df_full_training[col].dtype == 'datetime64[ns]':
            df_full_training[col] = df_full_training[col].dt.strftime('%Y-%m-%d')
    
    # Define the final column order based on the requested variables
    final_cols = [
        # Log-Level Metrics
        'log_id', 'project_id', 'employee_id', 
        'date_logged', 'task_category', 'subtask_description', 'hours_worked',
        'week', 'is_winter_day', 

        # Project Metadata (Planned/Static)
        'project_type', 'material_type', 'scope_category', 
        'surface_area_m2', 'num_levels', 'num_units', 'building_height_m', 
        'floor_area_ratio', 
        
        # Planned Duration
        'planned_start_date', 'planned_end_date', 'expected_duration_days',
        
        # Actual/Calculated Metrics (Joined from Aggregation)
        'corrected_start_date', 'corrected_end_date', 
        'project_duration_days', 'actual_duration_days', # Both derived from corrected dates
        'month_started', 'quarter', 'season_flag', 'holiday_period_flag',
        'total_project_hours', 'design_hours_total', 'num_site_visits', 
        'num_revisions', 'revision_reason',
        'avg_hours_per_employee',
    ]

    final_cols = [col for col in final_cols if col in df_full_training.columns]
    df_final_training = df_full_training[final_cols]
    
    print(f"\nGenerated {len(df_final_training)} full log entries.")
    print(f"Full Training Data Shape: {df_final_training.shape}")
    
    # Save to CSV
    df_final_training.to_csv('daskan_full_exploration_data_complete.csv', index=False)
    
    print("\nFile saved:")
    print("1. 'daskan_full_exploration_data_complete.csv' (Contains ALL 26 requested variables per log entry)")

# Run the generator
generate_synthetic_data()