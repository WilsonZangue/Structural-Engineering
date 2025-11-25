import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(87)

def generate_synthetic_data(num_projects=87):
    print("Generating Daskan Inc. Synthetic Data...")
    
    # --- 1. PROJECT METADATA & DURATION LOGIC ---
    project_types = ['Residential', 'Commercial', 'Institutional', 'Industrial']
    materials = ['Wood', 'Steel', 'Concrete', 'Mixed']
    
    projects = []
    
    # We will store the 'target effort' here to use it in step 2
    project_effort_map = {} 
    
    for i in range(num_projects):
        p_id = f"P-2022-{str(i+1).zfill(3)}"
        p_type = np.random.choice(project_types, p=[0.25, 0.35, 0.15, 0.25])
        
        # 1.1 Determine Physical Size
        if p_type == 'Residential':
            area = np.random.randint(100, 600)
            levels = np.random.randint(1, 4)
        elif p_type == 'Commercial':
            area = np.random.randint(500, 2000)
            levels = np.random.randint(2, 8)
        else:
            area = np.random.randint(1000, 5000)
            levels = np.random.randint(1, 15)
            
        # 1.2 Determine Start Date
        start_month = np.random.randint(1, 12) # Leaving room for Dec end dates
        start_date = datetime(2022, start_month, 1) + timedelta(days=np.random.randint(0, 28))
        
        # 1.3 Calculate Target Effort (Hours) to estimate Duration
        base_effort = (area * 0.1) + (levels * 20)
        if p_type in ['Institutional', 'Industrial']:
            base_effort *= 1.5
        
        # Add noise/variation
        total_hours_est = int(np.random.normal(base_effort, base_effort * 0.15))
        total_hours_est = max(10, total_hours_est)
        
        # Store for Step 2
        project_effort_map[p_id] = total_hours_est
        
        # 1.4 Calculate End Date based on Effort
        # Assumption: A team works ~30 effective hours a week on this specific project
        estimated_weeks = max(1, total_hours_est / 30) 
        duration_days = int(estimated_weeks * 7) + np.random.randint(5, 20) # Add buffer
        end_date = start_date + timedelta(days=duration_days)

        projects.append({
            'project_id': p_id,
            'project_type': p_type,
            'material_type': np.random.choice(materials),
            'surface_area_m2': area,
            'num_levels': levels,
            'start_date': start_date, # Keep as datetime for calc
            'end_date': end_date      # Keep as datetime for calc
        })
        
    df_projects = pd.DataFrame(projects)
    
    # --- 2. TIMESHEETS (Detailed Logs) ---
    timesheet_entries = []
    log_id_counter = 1
    
    for _, proj in df_projects.iterrows():
        p_id = proj['project_id']
        total_hours = project_effort_map[p_id]
        
        # Number of individual log entries
        num_entries = np.random.randint(5, 60)
        avg_entry_hours = total_hours / num_entries
        
        start_dt = proj['start_date']
        end_dt = proj['end_date']
        date_range_days = (end_dt - start_dt).days
        
        for _ in range(num_entries):
            # Randomize hours for this specific entry
            hours = round(np.random.normal(avg_entry_hours, 1.5), 2)
            if hours <= 0.25: hours = 0.5
            
            # Randomize Date within Project Window
            random_days = np.random.randint(0, max(1, date_range_days))
            log_date = start_dt + timedelta(days=random_days)
            
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
    
    # --- 3. CREATE BIG MASTER TRAINING SET (Full Merge) ---
    
    # Convert dates to string for clean CSV export
    df_projects['start_date'] = df_projects['start_date'].dt.strftime('%Y-%m-%d')
    df_projects['end_date'] = df_projects['end_date'].dt.strftime('%Y-%m-%d')
    df_timesheets['date_logged'] = df_timesheets['date_logged'].dt.strftime('%Y-%m-%d')

    # Merge Timesheets (Left) with Projects (Right)
    # This creates the "Big Data Set" where every row is a log entry + project context
    df_master = pd.merge(df_timesheets, df_projects, on='project_id', how='left')
    
    # Feature Engineering: Seasonality (Based on Log Date, not just Project Start)
    df_master['date_logged'] = pd.to_datetime(df_master['date_logged'])
    df_master['is_winter'] = df_master['date_logged'].dt.month.isin([12, 1, 2, 3]).astype(int)
    
    # Feature Engineering: Floor Area Ratio
    df_master['floor_area_ratio'] = df_master['surface_area_m2'] / df_master['num_levels']
    
    print(f"Generated {len(df_projects)} projects.")
    print(f"Generated {len(df_timesheets)} granular timesheet logs.")
    print(f"Master Dataset Shape: {df_master.shape}")
    
    # Save to CSV
    df_projects.to_csv('daskan_projects_metadata.csv', index=False)
    df_timesheets.to_csv('daskan_timesheets_raw.csv', index=False)
    df_master.to_csv('daskan_full_training_data.csv', index=False)
    
    print("Files saved:")
    print("1. 'daskan_projects_metadata.csv' (Project info + Start/End Dates)")
    print("2. 'daskan_timesheets_raw.csv' (Logs only)")
    print("3. 'daskan_full_training_data.csv' (The Big Data Set)")

# Run the generator
generate_synthetic_data()