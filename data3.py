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
    
    MODIFIED: Project IDs now correlate with start year, and dates span 2022-2024.
    """
    print("👷 Generating Daskan Inc. Synthetic Data with Multi-Year Dates...")
    
    # --- 1. PROJECT METADATA & DURATION LOGIC ---
    project_types = ['Residential', 'Commercial', 'Institutional', 'Industrial']
    materials = ['Wood', 'Steel', 'Concrete', 'Mixed']
    
    projects = []
    project_effort_map = {} 
    
    # Dictionary to keep track of project count per year for sequential ID generation
    project_counter = {2022: 0, 2023: 0, 2024: 0} 
    
    # Generate a list of start years and sort them so project IDs are sequential by year
    start_years = np.random.choice([2022, 2023, 2024], size=num_projects, p=[0.3, 0.4, 0.3])
    
    # We will iterate based on the start year list to ensure distribution
    for start_year in start_years:
        
        # 1.1 Determine Start Date
        # Start months are randomized for each year
        start_month = np.random.randint(1, 13) 
        start_date = datetime(start_year, start_month, 1) + timedelta(days=np.random.randint(0, 28))
        
        # 1.2 Generate Project ID based on Year and Sequence
        project_counter[start_year] += 1
        p_id = f"P-{start_year}-{str(project_counter[start_year]).zfill(3)}"
        
        # 1.3 Determine Physical Size & Complexity (Same logic as before)
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
            'planned_start_date': start_date, # Now linked to the new start year logic
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
        actual_total_hours = total_hours * np.random.normal(1.05, 0.1) 
        actual_total_hours = max(10, actual_total_hours)
        
        num_entries = np.random.randint(int(actual_total_hours/5), int(actual_total_hours/0.5)) 
        avg_entry_hours = actual_total_hours / num_entries
        
        start_dt_planned = proj['planned_start_date']
        
        # Simulate project completion taking longer/shorter than planned duration
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
                'employee_id': f"EMP-{np.random.randint(1, 12)}", 
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
    df_master['actual_duration_days'] = (df_master['corrected_end_date'] - df_master['corrected_start_date']).dt.days # Actual measured duration
    
    # 2. Geometry
    df_master['floor_area_ratio'] = df_master['surface_area_m2'] / df_master['num_levels']

    # 3. Time/Date Features
    df_master['month_started'] = df_master['corrected_start_date'].dt.month
    df_master['quarter'] = df_master['corrected_start_date'].dt.quarter
    
    def get_season_flag(month):
        if month in [12, 1, 2]: return 'Winter'
        if month in [3, 4, 5]: return 'Spring'
        if month in [6, 7, 8]: return 'Summer'
        return 'Autumn'
        
    df_master['season_flag'] = df_master['corrected_start_date'].dt.month.apply(get_season_flag)
    
    # Check for holiday overlap between start and end dates (Nov 15 - Jan 15)
    def check_holiday_overlap(start, end):
        holiday_start = datetime(start.year, 11, 15)
        holiday_end = datetime(start.year + 1, 1, 15) if start.month >= 11 else datetime(start.year, 1, 15)
        
        # Adjust holiday_start to previous year if project starts in Jan/Feb
        if start.month in [1, 2] and holiday_start.year == start.year:
            holiday_start = datetime(start.year - 1, 11, 15)
            
        return int((start <= holiday_end) and (end >= holiday_start))

    df_master['holiday_period_flag'] = df_master.apply(
        lambda row: check_holiday_overlap(row['corrected_start_date'], row['corrected_end_date']), axis=1
    )

    # 4. Employee/Effort Features
    df_master['avg_hours_per_employee'] = df_master['total_project_hours'] / df_master['num_employees']
    
    
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
        'actual_duration_days', 
        
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

    final_cols = [col for col in final_cols if col in df_master.columns]
    df_final_training = df_master[final_cols]
    
    # Save the raw timesheets with cleaned dates for the supporting data
    df_timesheets['date_logged'] = df_timesheets['date_logged'].dt.strftime('%Y-%m-%d')

    print(f"\nGenerated {len(df_final_training)} projects.")
    print(f"Generated {len(df_timesheets)} granular timesheet logs.")
    print(f"Master Project Summary Shape: {df_final_training.shape}")
    
    # Save to CSV
    # Using 'v3' suffix to denote this latest, multi-year, ID-correlated version
    df_projects_metadata.to_csv('daskan_projects_metadata_v3.csv', index=False)
    df_timesheets.to_csv('daskan_timesheets_raw_v3.csv', index=False)
    df_final_training.to_csv('daskan_project_exploration_data_v3.csv', index=False)
    
    print("\nFiles saved:")
    print("1. 'daskan_projects_metadata_v3.csv' (Initial Planned Info)")
    print("2. 'daskan_timesheets_raw_v3.csv' (Granular Logs)")
    print("3. 'daskan_project_exploration_data_v3.csv' (The Project Exploration Dataset)")

# Run the generator
generate_synthetic_data()