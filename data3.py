import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(87)

def generate_synthetic_data(num_projects=87):
    """
    Generates synthetic construction project data and timesheet logs.
    It produces three CSV files: metadata, raw logs, and a master project summary.
    """
    print("👷 Generating Daskan Inc. Synthetic Data...")
    
    # --- 1. PROJECT METADATA & DURATION LOGIC ---
    project_types = ['Residential', 'Commercial', 'Institutional', 'Industrial']
    materials = ['Wood', 'Steel', 'Concrete', 'Mixed']
    
    projects = []
    project_effort_map = {} 
    
    for i in range(num_projects):
        p_id = f"P-2022-{str(i+1).zfill(3)}"
        p_type = np.random.choice(project_types, p=[0.25, 0.35, 0.15, 0.25])
        
        # 1.1 Determine Physical Size & Complexity
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
            
        # 1.2 Determine Planned Dates
        start_month = np.random.randint(1, 12) 
        planned_start_date = datetime(2022, start_month, 1) + timedelta(days=np.random.randint(0, 28))
        
        # 1.3 Calculate Target Effort (Hours)
        base_effort = (area * 0.1) + (levels * 20)
        if p_type in ['Institutional', 'Industrial']:
            base_effort *= 1.5
        
        total_hours_est = int(np.random.normal(base_effort, base_effort * 0.15))
        total_hours_est = max(10, total_hours_est)
        project_effort_map[p_id] = total_hours_est
        
        # 1.4 Calculate Planned End Date
        estimated_weeks = max(1, total_hours_est / 30) 
        expected_duration_days = int(estimated_weeks * 7) + np.random.randint(5, 20) # Add buffer
        planned_end_date = planned_start_date + timedelta(days=expected_duration_days)

        # 1.5 Revision Data (New Features)
        num_revisions = np.random.poisson(1.5) # Average 1.5 revisions
        if num_revisions > 0:
            revision_reason = np.random.choice(['Client Change', 'Code Conflict', 'Scope Creep', 'Site Condition'], 
                                               p=[0.4, 0.3, 0.2, 0.1])
        else:
            revision_reason = 'None'

        projects.append({
            'project_id': p_id,
            'project_type': p_type,
            'material_type': np.random.choice(materials),
            'surface_area_m2': area,
            'num_levels': levels,
            # New variables
            'num_units': num_units,
            'building_height_m': round(height, 2),
            'num_revisions': num_revisions,
            'revision_reason': revision_reason,
            'planned_start_date': planned_start_date, 
            'planned_end_date': planned_end_date,
            'expected_duration_days': expected_duration_days,
            'scope_category': scope_cat
        })
        
    df_projects_metadata = pd.DataFrame(projects)
    
    # --- 2. TIMESHEETS (Detailed Logs) ---
    timesheet_entries = []
    log_id_counter = 1
    
    for _, proj in df_projects_metadata.iterrows():
        p_id = proj['project_id']
        total_hours = project_effort_map[p_id]
        
        # Simulate final hours differing from estimate (Actual vs Planned)
        actual_total_hours = total_hours * np.random.normal(1.05, 0.1) # Actual effort ~5% more
        actual_total_hours = max(10, actual_total_hours)
        
        num_entries = np.random.randint(int(actual_total_hours/5), int(actual_total_hours/0.5)) 
        avg_entry_hours = actual_total_hours / num_entries
        
        start_dt_planned = proj['planned_start_date']
        end_dt_planned = proj['planned_end_date']
        
        # Simulate project completion taking a bit longer/shorter than planned duration
        actual_duration_days = int(proj['expected_duration_days'] * np.random.normal(1.1, 0.1))
        
        # Actual end date used for log generation
        actual_end_dt = start_dt_planned + timedelta(days=actual_duration_days)
        date_range_days = (actual_end_dt - start_dt_planned).days
        
        for _ in range(num_entries):
            # Hours worked for this entry
            hours = round(np.random.normal(avg_entry_hours, 1.5), 2)
            if hours <= 0.25: hours = 0.5
            
            # Log Date within Project Window
            random_days = np.random.randint(0, max(1, date_range_days))
            log_date = start_dt_planned + timedelta(days=random_days)
            
            # Task categories
            task = np.random.choice(['Design', 'Calculation', 'Drafting', 'Meeting', 'Site Visit'])
            
            timesheet_entries.append({
                'log_id': log_id_counter,
                'project_id': p_id,
                'employee_id': f"EMP-{np.random.randint(1, 12)}", # 12 Employees
                'date_logged': log_date,
                'task_category': task,
                'hours_worked': hours
            })
            log_id_counter += 1
            
    df_timesheets = pd.DataFrame(timesheet_entries)
    
    # --- 3. CREATE PROJECT SUMMARY DATASET (Feature Engineering) ---
    
    # Create the base merged dataframe
    df_master_raw = pd.merge(df_timesheets, df_projects_metadata, on='project_id', how='left')
    df_master_raw['date_logged'] = pd.to_datetime(df_master_raw['date_logged'])
    
    # --- A. Aggregation Step ---
    df_summary = df_master_raw.groupby('project_id').agg(
        # Duration & Dates
        corrected_start_date=('date_logged', 'min'),
        corrected_end_date=('date_logged', 'max'),
        total_project_hours=('hours_worked', 'sum'),
        
        # Employee Metrics
        num_employees=('employee_id', 'nunique'),
        
        # Task Metrics
        design_hours_total=('hours_worked', lambda x: x[df_master_raw.loc[x.index, 'task_category'] == 'Design'].sum()),
        num_site_visits=('task_category', lambda x: (x == 'Site Visit').sum())
    ).reset_index()

    # Merge aggregated data back with metadata
    df_master = pd.merge(df_projects_metadata, df_summary, on='project_id', how='left')

    # --- B. Feature Calculation Step ---

    # 1. Duration and Ratio Calculations
    df_master['project_duration_days'] = (df_master['corrected_end_date'] - df_master['corrected_start_date']).dt.days
    df_master['actual_duration_days'] = (df_master['planned_end_date'] - df_master['planned_start_date']).dt.days # This uses the *planned* duration logic for consistency in this synthetic dataset
    
    # 2. Geometry
    df_master['floor_area_ratio'] = df_master['surface_area_m2'] / df_master['num_levels']

    # 3. Time/Date Features
    df_master['month_started'] = df_master['corrected_start_date'].dt.month
    df_master['quarter'] = df_master['corrected_start_date'].dt.quarter
    
    # Simple Season/Holiday Flags
    def get_season_flag(month):
        if month in [12, 1, 2]: return 'Winter'
        if month in [3, 4, 5]: return 'Spring'
        if month in [6, 7, 8]: return 'Summer'
        return 'Autumn'
        
    df_master['season_flag'] = df_master['corrected_start_date'].dt.month.apply(get_season_flag)
    
    # Simple Holiday Overlap Check (e.g., if project spans Thanksgiving/Christmas/New Year)
    # Checks if corrected_start_date is before Nov 15th and corrected_end_date is after Jan 15th
    holiday_overlap_check = lambda start, end: (
        (start.month < 12 and end.month >= 12) or 
        (start.month == 12) or
        (end.month == 1) or
        (start < datetime(2022, 12, 20) and end > datetime(2023, 1, 5))
    )
    df_master['holiday_period_flag'] = df_master.apply(
        lambda row: int(holiday_overlap_check(row['corrected_start_date'], row['corrected_end_date'])), axis=1
    )

    # 4. Employee/Effort Features
    df_master['avg_hours_per_employee'] = df_master['total_project_hours'] / df_master['num_employees']
    
    # 5. Supporting Variables (for training data, but might not be explicitly in the final output table)
    # The actual master table is project-level, so these log-level supporting variables are better kept in the raw log file.
    # The 'is_winter' column generated in the original code is already a log-level feature.
    
    
    # --- C. Final Clean-up and Save ---
    
    # Format Dates for CSV
    df_master['corrected_start_date'] = df_master['corrected_start_date'].dt.strftime('%Y-%m-%d')
    df_master['corrected_end_date'] = df_master['corrected_end_date'].dt.strftime('%Y-%m-%d')
    df_master['planned_start_date'] = df_master['planned_start_date'].dt.strftime('%Y-%m-%d')
    df_master['planned_end_date'] = df_master['planned_end_date'].dt.strftime('%Y-%m-%d')
    
    # Select and reorder columns for the final training set
    final_cols = [
        'project_id',
        'project_type',
        'material_type',
        'scope_category',
        
        'surface_area_m2',
        'num_levels',
        'num_units',
        'building_height_m',
        'floor_area_ratio',
        
        'planned_start_date',
        'planned_end_date',
        'expected_duration_days',
        
        'corrected_start_date',
        'corrected_end_date',
        'project_duration_days',
        'actual_duration_days', # Renamed from original for clarity
        
        'month_started',
        'quarter',
        'season_flag',
        'holiday_period_flag',
        
        'total_project_hours',
        'design_hours_total',
        'num_site_visits',
        'num_revisions',
        'revision_reason',
        
        'num_employees',
        'avg_hours_per_employee',
    ]

    # Handle missing columns if any project had no logs (shouldn't happen here)
    final_cols = [col for col in final_cols if col in df_master.columns]
    df_final_training = df_master[final_cols]
    
    # Also save the raw timesheets with cleaned dates for the supporting data
    df_timesheets['date_logged'] = df_timesheets['date_logged'].dt.strftime('%Y-%m-%d')

    print(f"\nGenerated {len(df_final_training)} projects.")
    print(f"Generated {len(df_timesheets)} granular timesheet logs.")
    print(f"Master Project Summary Shape: {df_final_training.shape}")
    
    # Save to CSV
    df_projects_metadata.to_csv('daskan_projects_metadata_v2.csv', index=False)
    df_timesheets.to_csv('daskan_timesheets_raw_v2.csv', index=False)
    df_final_training.to_csv('daskan_project_summary_training_data_v2.csv', index=False)
    
    print("\nFiles saved:")
    print("1. 'daskan_projects_metadata_v2.csv' (Initial Planned Info)")
    print("2. 'daskan_timesheets_raw_v2.csv' (Granular Logs - now log-level supporting data)")
    print("3. 'daskan_project_exploration_data_v2.csv' (The Project-Level Summary Set)")

# Run the generator
generate_synthetic_data()