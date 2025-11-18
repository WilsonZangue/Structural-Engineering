import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(num_projects=87):
    print("Generating Daskan Inc. Synthetic Data...")
    
    # --- 1. PROJECT METADATA (The "Plans" & "Calculation Notes") ---
    project_types = ['Residential', 'Commercial', 'Institutional', 'Industrial']
    materials = ['Wood', 'Steel', 'Concrete', 'Mixed']
    
    projects = []
    
    for i in range(num_projects):
        p_id = f"P-2024-{str(i+1).zfill(3)}"
        p_type = np.random.choice(project_types, p=[0.25, 0.35, 0.15, 0.25]) # 25% Residential
        
        # Logic: Commercial/Industrial are usually larger
        if p_type == 'Residential':
            area = np.random.randint(100, 600) # m2
            levels = np.random.randint(1, 4)
        elif p_type == 'Commercial':
            area = np.random.randint(500, 2000)
            levels = np.random.randint(2, 8)
        else:
            area = np.random.randint(1000, 5000)
            levels = np.random.randint(1, 15)
            
        # Seasonality Logic
        start_month = np.random.randint(1, 13)
        start_date = datetime(2022, start_month, 1) + timedelta(days=np.random.randint(0, 28))
        
        projects.append({
            'project_id': p_id,
            'project_type': p_type,
            'material_type': np.random.choice(materials),
            'surface_area_m2': area,
            'num_levels': levels,
            'start_date': start_date.strftime('%Y-%m-%d')
        })
        
    df_projects = pd.DataFrame(projects)
    
    # --- 2. TIMESHEETS (The "Feuilles de Temps") ---
    # We need to generate hours that generally correlate with size (so your model works) 
    # but have noise (randomness) to make it realistic.
    
    timesheet_entries = []
    log_id_counter = 1
    
    for _, proj in df_projects.iterrows():
        # Base effort calculation (Ground truth logic + Noise)
        base_effort = (proj['surface_area_m2'] * 0.1) + (proj['num_levels'] * 20)
        
        # Complex types take longer
        if proj['project_type'] in ['Institutional', 'Industrial']:
            base_effort *= 1.5
            
        # Random variation (Standard Deviation)
        actual_total_effort = int(np.random.normal(base_effort, base_effort * 0.15))
        actual_total_effort = max(8, actual_total_effort) # Minimum 8 hours
        
        # Distribute this effort across multiple log entries
        # A project might have 5 to 50 entries depending on size
        num_entries = np.random.randint(5, 50)
        avg_entry = actual_total_effort / num_entries
        
        for _ in range(num_entries):
            hours = round(np.random.normal(avg_entry, 1.0), 2)
            if hours <= 0: hours = 0.5
            
            # Task categories
            task = np.random.choice(['Design', 'Calculation', 'Drafting', 'Meeting'])
            
            timesheet_entries.append({
                'log_id': log_id_counter,
                'project_id': proj['project_id'],
                'employee_id': f"EMP-{np.random.randint(1, 10)}",
                'date_logged': proj['start_date'], # Simplified for this example
                'task_category': task,
                'hours_worked': hours
            })
            log_id_counter += 1
            
    df_timesheets = pd.DataFrame(timesheet_entries)
    
    # --- 3. CREATE MASTER TRAINING SET (Aggregation) ---
    
    # Aggregate hours per project
    total_hours = df_timesheets.groupby('project_id')['hours_worked'].sum().reset_index()
    total_hours.rename(columns={'hours_worked': 'total_project_effort'}, inplace=True)
    
    # Merge with Metadata
    df_master = pd.merge(df_projects, total_hours, on='project_id', how='inner')
    
    # Feature Engineering: Seasonality Flag
    # (Winter in Quebec: Dec, Jan, Feb, Mar)
    df_master['start_date'] = pd.to_datetime(df_master['start_date'])
    df_master['is_winter'] = df_master['start_date'].dt.month.isin([12, 1, 2, 3]).astype(int)
    
    # Feature Engineering: Floor Area Ratio
    df_master['floor_area_ratio'] = df_master['surface_area_m2'] / df_master['num_levels']
    
    print(f"Successfully generated {len(df_projects)} projects and {len(df_timesheets)} timesheet logs.")
    
    # Save to CSV
    df_projects.to_csv('daskan_projects_metadata.csv', index=False)
    df_timesheets.to_csv('daskan_timesheets_raw.csv', index=False)
    df_master.to_csv('daskan_master_training_set.csv', index=False)
    print("Files saved: 'daskan_projects_metadata.csv', 'daskan_timesheets_raw.csv', 'daskan_master_training_set.csv'")

# Run the generator
generate_synthetic_data()