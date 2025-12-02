import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(87)

def generate_synthetic_data_optimized(num_projects=87):
    print("👷 Generating Daskan Inc. FULL Exploration Data (Optimized)...")

    # --- 1. CONSTANTS / LOOKUPS ---
    project_types = np.array(['Residential', 'Commercial', 'Institutional', 'Industrial'])
    project_type_probs = np.array([0.25, 0.35, 0.15, 0.25])
    materials = np.array(['Wood', 'Steel', 'Concrete', 'Mixed'])

    task_weights = {
        'Residential':     {'Design': 0.30, 'Calculation': 0.10, 'Drafting': 0.40, 'Meeting': 0.10, 'Site Visit': 0.10},
        'Commercial':      {'Design': 0.25, 'Calculation': 0.20, 'Drafting': 0.30, 'Meeting': 0.15, 'Site Visit': 0.10},
        'Institutional':   {'Design': 0.20, 'Calculation': 0.30, 'Drafting': 0.20, 'Meeting': 0.20, 'Site Visit': 0.10},
        'Industrial':      {'Design': 0.15, 'Calculation': 0.35, 'Drafting': 0.15, 'Meeting': 0.25, 'Site Visit': 0.10},
    }
    task_categories = list(task_weights['Residential'].keys())

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

    # --- 2. VECTORIZED PROJECT METADATA GENERATION ---
    start_years = np.random.choice([2022, 2023, 2024], size=num_projects, p=[0.3, 0.4, 0.3])
    start_months = np.random.randint(1, 13, size=num_projects)
    start_day_offsets = np.random.randint(0, 28, size=num_projects)

    start_dates = pd.to_datetime({
        'year': start_years,
        'month': start_months,
        'day': np.ones(num_projects, dtype=int)
    }) + pd.to_timedelta(start_day_offsets, unit='D')

    # Project IDs (preserve zero-padding per year sequence)
    project_counters = {2022: 0, 2023: 0, 2024: 0}
    project_ids = []
    for yr in start_years:
        project_counters[yr] += 1
        project_ids.append(f"P-{yr}-{str(project_counters[yr]).zfill(3)}")
    project_ids = np.array(project_ids)

    # Project types and materials
    p_types = np.random.choice(project_types, size=num_projects, p=project_type_probs)
    p_materials = np.random.choice(materials, size=num_projects)

    # Physical attributes vectorized by project type
    areas = np.empty(num_projects, dtype=int)
    levels = np.empty(num_projects, dtype=int)
    num_units = np.ones(num_projects, dtype=int)
    heights = np.empty(num_projects, dtype=float)
    scope_cat = np.empty(num_projects, dtype=object)

    for i, pt in enumerate(p_types):
        if pt == 'Residential':
            areas[i] = np.random.randint(100, 600)
            levels[i] = np.random.randint(1, 4)
            num_units[i] = np.random.randint(1, 10)
            heights[i] = levels[i] * np.random.normal(3.2, 0.3)
            scope_cat[i] = 'New Build - Low Density'
        elif pt == 'Commercial':
            areas[i] = np.random.randint(500, 2000)
            levels[i] = np.random.randint(2, 8)
            heights[i] = levels[i] * np.random.normal(4.0, 0.5)
            scope_cat[i] = 'Tenant Improvement' if np.random.rand() < 0.3 else 'New Build - Mid Density'
        else:
            areas[i] = np.random.randint(1000, 5000)
            levels[i] = np.random.randint(1, 15)
            heights[i] = levels[i] * np.random.normal(4.5, 0.7)
            scope_cat[i] = 'Complex Infrastructure'

    # Effort estimate (vectorized)
    base_effort = (areas * 0.1) + (levels * 20)
    institutional_mask = np.isin(p_types, ['Institutional', 'Industrial'])
    base_effort[institutional_mask] *= 1.5

    total_hours_est = np.random.normal(base_effort, base_effort * 0.15)
    total_hours_est = np.maximum(10, (total_hours_est).astype(int))

    estimated_weeks = np.maximum(1, total_hours_est / 30)
    expected_duration_days = (estimated_weeks * 7).astype(int) + np.random.randint(5, 20, size=num_projects)
    planned_end_dates = start_dates + pd.to_timedelta(expected_duration_days, unit='D')

    # Revisions (vectorized-ish)
    num_revisions = np.random.poisson(1.5, size=num_projects)
    revision_reasons_list = np.array([
        'Client Change','Scope Creep','Value Engineering Request','Architectural Change',
        'Late Client Information','Stakeholder Requirement Change','End-User Requirement Update',
        'Design Optimization','Structural Recalculation','Clash Detection Issue','Design Error Correction',
        'Survey Data Update','Utility Conflict','Structural Detail Revision','MEP Coordination Revision',
        'Foundation Design Revision','Fire Protection Requirement Change','Site Condition',
        'Groundwater Issue','Access or Logistics Constraint','Field Adjustment Needed','Weather-Related Adjustment',
        'Contractor Request','Code Conflict','Permit Review Comments','Environmental Regulatory Update',
        'Accessibility Standard Update','Vendor/Manufacturer Change','Material Lead Time Issue',
        'Procurement Delay','Drawing Correction','Specification Clarification','Missing Detail Correction','RFI Response Integration'
    ])

    # Choose reason only when num_revisions > 0
    revision_reasons = np.where(
        num_revisions > 0,
        np.random.choice(revision_reasons_list, size=num_projects),
        'None'
    )

    # Assemble projects DataFrame
    df_projects_metadata = pd.DataFrame({
        'project_id': project_ids,
        'project_type': p_types,
        'material_type': p_materials,
        'surface_area_m2': areas,
        'num_levels': levels,
        'num_units': num_units,
        'building_height_m': np.round(heights, 2),
        'num_revisions': num_revisions,
        'revision_reason': revision_reasons,
        'planned_start_date': start_dates,
        'planned_end_date': planned_end_dates,
        'expected_duration_days': expected_duration_days,
        'scope_category': scope_cat
    })

    # Map project effort for quick lookup
    project_effort_map = dict(zip(project_ids, total_hours_est))

    # --- 3. TIMESHEETS (batched per project, vectorized inner draws) ---
    timesheet_rows = []  # will collect dicts for DataFrame construction but each dict will contain arrays
    log_id_counter = 1

    # Helper to generate employee ids as strings quickly
    def gen_emp_ids(n):
        return np.char.add('EMP-', np.random.randint(1, 12, size=n).astype(str))

    for _, proj in df_projects_metadata.iterrows():
        p_id = proj['project_id']
        p_type = proj['project_type']

        total_hours = project_effort_map[p_id]
        actual_total_hours = max(10, total_hours * np.random.normal(1.05, 0.1))

        # number of entries sampled in wider band but deterministic
        num_entries = int(np.random.randint(max(1, int(actual_total_hours/5)), max(2, int(actual_total_hours/0.5))))

        start_dt_planned = proj['planned_start_date']
        actual_duration_days = int(proj['expected_duration_days'] * np.random.normal(1.1, 0.1))
        actual_end_dt = start_dt_planned + timedelta(days=actual_duration_days)
        date_range_days = max(1, (actual_end_dt - start_dt_planned).days)

        # target task hours for this project
        weights = task_weights[p_type]
        target_task_hours = {task: actual_total_hours * weights[task] for task in task_categories}

        # estimate logs per task using a reproducible divisor
        logs_per_task = {task: max(1, int(target_task_hours[task] / max(1.0, np.random.normal(2.5, 0.5)))) for task in task_categories}

        # allocate final logs by proportional distribution
        total_logs_est = sum(logs_per_task.values())
        if total_logs_est == 0:
            final_logs = {task: 1 for task in task_categories}
        else:
            # proportional split to match requested num_entries
            proportions = np.array(list(logs_per_task.values())) / total_logs_est
            allocated = np.floor(proportions * num_entries).astype(int)
            # ensure at least one per task
            allocated = np.maximum(allocated, 1)
            # correct rounding differences
            diff = num_entries - allocated.sum()
            if diff > 0:
                allocated[np.argmax(allocated)] += diff
            final_logs = dict(zip(task_categories, allocated))

        # For each task, generate arrays of attributes in batch
        for task, nlogs in final_logs.items():
            if nlogs <= 0:
                continue

            # average hours computed against original logs_per_task to mimic original logic
            denom = logs_per_task.get(task, 1)
            avg_task_hours = target_task_hours.get(task, 0) / max(1, denom)
            hours = np.round(np.random.normal(loc=avg_task_hours, scale=1.0, size=nlogs), 2)
            hours[hours <= 0.25] = 0.5

            # random days offset from planned start
            day_offsets = np.random.randint(0, date_range_days, size=nlogs)
            log_dates = pd.to_datetime(start_dt_planned) + pd.to_timedelta(day_offsets, unit='D')

            # employee ids and subtasks
            emp_ids = gen_emp_ids(nlogs)
            subtasks = np.random.choice(subtask_map[task], size=nlogs)

            # build rows
            for i in range(nlogs):
                timesheet_rows.append({
                    'log_id': log_id_counter,
                    'project_id': p_id,
                    'employee_id': emp_ids[i],
                    'date_logged': log_dates[i],
                    'task_category': task,
                    'subtask_description': subtasks[i],
                    'hours_worked': float(hours[i])
                })
                log_id_counter += 1

    df_timesheets = pd.DataFrame(timesheet_rows)

    # --- 4. AGGREGATE PROJECT METRICS (fast, avoid lambdas) ---
    df_timesheets['date_logged'] = pd.to_datetime(df_timesheets['date_logged'])

    # Basic aggregations
    agg_basic = df_timesheets.groupby('project_id').agg(
        corrected_start_date=('date_logged', 'min'),
        corrected_end_date=('date_logged', 'max'),
        total_project_hours=('hours_worked', 'sum'),
        num_employees=('employee_id', 'nunique')
    ).reset_index()

    # Task-specific aggregations computed by filtering then grouping
    design_hours = (
        df_timesheets[df_timesheets['task_category'] == 'Design']
        .groupby('project_id')['hours_worked'].sum()
        .rename('design_hours_total')
        .reset_index()
    )

    site_visits = (
        df_timesheets[df_timesheets['task_category'] == 'Site Visit']
        .groupby('project_id')['task_category']
        .count()
        .rename('num_site_visits')
        .reset_index()
    )

    df_project_metrics = agg_basic.merge(design_hours, on='project_id', how='left')
    df_project_metrics = df_project_metrics.merge(site_visits, on='project_id', how='left')
    df_project_metrics['design_hours_total'] = df_project_metrics['design_hours_total'].fillna(0.0)
    df_project_metrics['num_site_visits'] = df_project_metrics['num_site_visits'].fillna(0).astype(int)

    # duration and averages
    df_project_metrics['project_duration_days'] = (df_project_metrics['corrected_end_date'] - df_project_metrics['corrected_start_date']).dt.days
    df_project_metrics['actual_duration_days'] = df_project_metrics['project_duration_days']
    df_project_metrics['avg_hours_per_employee'] = df_project_metrics['total_project_hours'] / df_project_metrics['num_employees']

    # time features
    df_project_metrics['month_started'] = df_project_metrics['corrected_start_date'].dt.month
    df_project_metrics['quarter'] = df_project_metrics['corrected_start_date'].dt.quarter

    def get_season_flag_series(months):
        res = np.full(len(months), 'Autumn', dtype=object)
        res[np.isin(months, [12,1,2])] = 'Winter'
        res[np.isin(months, [3,4,5])] = 'Spring'
        res[np.isin(months, [6,7,8])] = 'Summer'
        return res

    df_project_metrics['season_flag'] = get_season_flag_series(df_project_metrics['month_started'].values)

    # Vectorized holiday overlap check (Nov 15 - Jan 15 window)
    start = df_project_metrics['corrected_start_date']
    end = df_project_metrics['corrected_end_date']
    holiday_start = pd.to_datetime(dict(year=start.dt.year, month=11, day=15))
    holiday_end = pd.to_datetime(dict(year=start.dt.year + 1, month=1, day=15))
    holiday_start_prev = pd.to_datetime(dict(year=start.dt.year - 1, month=11, day=15))
    holiday_end_prev = pd.to_datetime(dict(year=start.dt.year, month=1, day=15))

    cond1 = (start < holiday_end) & (end > holiday_start)
    cond2 = (start < holiday_end_prev) & (end > holiday_start_prev)
    df_project_metrics['holiday_period_flag'] = (cond1 | cond2).astype(int)

    # --- 5. MERGE CONTEXT AND FINAL LOG-LEVEL FRAME ---
    df_context = df_projects_metadata.merge(df_project_metrics, on='project_id', how='left')
    df_context['floor_area_ratio'] = df_context['surface_area_m2'] / df_context['num_levels']

    df_full_training = df_timesheets.merge(df_context, on='project_id', how='left')

    # Efficient datetime formatting for export
    df_full_training['is_winter_day'] = df_full_training['date_logged'].dt.month.isin([12,1,2,3]).astype(int)
    df_full_training['week'] = df_full_training['date_logged'].dt.isocalendar().week.astype(int)

    # Prepare export copies and format dates in batch
    df_projects_metadata_export = df_projects_metadata.copy()
    df_projects_metadata_export['planned_start_date'] = df_projects_metadata_export['planned_start_date'].dt.strftime('%Y-%m-%d')
    df_projects_metadata_export['planned_end_date'] = df_projects_metadata_export['planned_end_date'].dt.strftime('%Y-%m-%d')

    df_timesheets_export = df_timesheets.copy()
    df_timesheets_export['date_logged'] = df_timesheets_export['date_logged'].dt.strftime('%Y-%m-%d')

    # format datetime columns in df_full_training
    datetime_cols = df_full_training.select_dtypes(include=['datetime64[ns]']).columns
    df_full_training[datetime_cols] = df_full_training[datetime_cols].apply(lambda s: s.dt.strftime('%Y-%m-%d'))

    final_cols = [
        'log_id', 'project_id', 'employee_id',
        'date_logged', 'task_category', 'subtask_description', 'hours_worked',
        'week', 'is_winter_day',
        'project_type', 'material_type', 'scope_category',
        'surface_area_m2', 'num_levels', 'num_units', 'building_height_m',
        'floor_area_ratio',
        'planned_start_date', 'planned_end_date', 'expected_duration_days',
        'corrected_start_date', 'corrected_end_date',
        'project_duration_days', 'actual_duration_days',
        'month_started', 'quarter', 'season_flag', 'holiday_period_flag',
        'total_project_hours', 'design_hours_total', 'num_site_visits',
        'num_revisions', 'revision_reason',
        'avg_hours_per_employee',
    ]

    final_cols = [c for c in final_cols if c in df_full_training.columns]
    df_final_training = df_full_training[final_cols]

    print(f"\nGenerated {len(df_final_training)} total log entries across {num_projects} projects.")

    # exports (same filenames as original)
    df_final_training.to_csv('daskan_full_exploration_data_v8.csv', index=False)
    df_projects_metadata_export.to_csv('daskan_projects_metadata_v8.csv', index=False)
    raw_cols = ['log_id', 'project_id', 'employee_id', 'date_logged', 'task_category', 'subtask_description', 'hours_worked']
    df_timesheets_export[raw_cols].to_csv('daskan_timesheets_raw_v8.csv', index=False)

    print("\nAll files successfully saved (optimized):")
    print("1. 'daskan_full_exploration_data_v8.csv'")
    print("2. 'daskan_projects_metadata_v8.csv'")
    print("3. 'daskan_timesheets_raw_v8.csv'")

# Entry point
if __name__ == '__main__':
    generate_synthetic_data_optimized()