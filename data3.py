#!/usr/bin/env python3
"""
daskan_synth_generator.py
Optimized synthetic data generator for Daskan (project + timesheets).
Produces three CSVs:
 - daskan_full_exploration_data_v15.csv
 - daskan_projects_metadata_v15.csv
 - daskan_timesheets_raw_v15.csv
"""

from __future__ import annotations
import os
import math
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ----------------------------
# Configuration / Constants
# ----------------------------
NUM_PROJECTS = 87
OUT_DIR = "./"
CSV_PREFIX = "daskan"
SEED = 87
MAX_LOGS_PER_PROJECT = 2000  # safety cap
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"

# Project configuration - tuned for structural engineering workflows
ENGINEER_TYPES = {
    'Junior engineer': 0.35,
    'Mid-level engineer': 0.30,
    'Senior engineer': 0.25,
    'Lead engineer': 0.10
}
PROJECT_TYPES = ['Residential', 'Commercial', 'Institutional', 'Industrial']
MATERIALS = ['Wood', 'Steel', 'Concrete', 'Mixed']
TASK_WEIGHTS = {
    'Residential':     {'Design': 0.30, 'Calculation': 0.10, 'Drafting': 0.40, 'Meeting': 0.10, 'Site Visit': 0.10},
    'Commercial':      {'Design': 0.25, 'Calculation': 0.20, 'Drafting': 0.30, 'Meeting': 0.15, 'Site Visit': 0.10},
    'Institutional':   {'Design': 0.20, 'Calculation': 0.30, 'Drafting': 0.20, 'Meeting': 0.20, 'Site Visit': 0.10},
    'Industrial':      {'Design': 0.15, 'Calculation': 0.35, 'Drafting': 0.15, 'Meeting': 0.25, 'Site Visit': 0.10},
}
SUBTASK_MAP = {
    'Design': ['Architectural Layout', 'Structural Concept', 'MEP System Planning', 'Facade/Exterior Detailing', 'Material Specification Review'],
    'Calculation': ['Structural Analysis (Beam/Column Sizing)', 'Load Calculations (Wind/Seismic)', 'Energy Modeling', 'Cost Estimation Review', 'Permit Fee Calculation'],
    'Drafting': ['Foundation Plan Updates', 'Section Drawings', 'Detail Sheet Generation', 'Markup Cleanup', '3D Model Adjustments'],
    'Meeting': ['Client Design Review', 'Subcontractor Coordination', 'Internal Team Sync', 'Permitting Authority Review', 'BIM Clash Detection Session'],
    'Site Visit': ['Existing Conditions Survey', 'Progress Inspection', 'Quality Check (QA/QC)', 'RFI Clarification', 'Punch List Generation']
}
TASK_CATEGORIES = list(TASK_WEIGHTS['Residential'].keys())
ENGINEER_LABELS = list(ENGINEER_TYPES.keys())
ENGINEER_PROBABILITIES = list(ENGINEER_TYPES.values())
YEARS = [2022, 2023, 2024]
START_YEAR_PROBS = [0.3, 0.4, 0.3]

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("daskan_generator")

# ----------------------------
# Dataclasses
# ----------------------------
@dataclass
class ProjectSpec:
    project_id: str
    project_type: str
    material_type: str
    surface_area_m2: int
    num_levels: int
    num_units: int
    building_height_m: float
    engineer_assigned: str
    project_notes: str
    num_revisions: int
    revision_reason: str
    planned_start_date: datetime
    planned_end_date: datetime
    expected_duration_days: int
    scope_category: str


# ----------------------------
# Helpers (small, well-tested)
# ----------------------------
def safe_randint_array(low: int, high: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Return an integer array with safe bounds (low <= result < high)."""
    if low >= high:
        return np.full(size, low, dtype=int)
    return rng.integers(low, high, size=size)

def get_project_dimensions_vectorized(types: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized generation for area, levels, units, height, and scope."""
    n = types.shape[0]
    area = np.empty(n, dtype=int)
    levels = np.empty(n, dtype=int)
    num_units = np.ones(n, dtype=int)
    height = np.empty(n, dtype=float)
    scope_cat = np.empty(n, dtype=object)

    for i, t in enumerate(types):
        if t == 'Residential':
            area[i] = int(rng.integers(100, 600))
            levels[i] = int(rng.integers(1, 4))
            num_units[i] = int(rng.integers(1, 10))
            h = levels[i] * rng.normal(3.2, 0.3)
            scope_cat[i] = 'New Build - Low Density'
        elif t == 'Commercial':
            area[i] = int(rng.integers(500, 2000))
            levels[i] = int(rng.integers(2, 8))
            num_units[i] = 1
            h = levels[i] * rng.normal(4.0, 0.5)
            scope_cat[i] = 'Tenant Improvement' if rng.random() < 0.3 else 'New Build - Mid Density'
        else:
            area[i] = int(rng.integers(1000, 5000))
            levels[i] = int(rng.integers(1, 15))
            num_units[i] = 1
            h = levels[i] * rng.normal(4.5, 0.7)
            scope_cat[i] = 'Complex Infrastructure'
        height[i] = round(float(h), 2)
    return area, levels, num_units, height, scope_cat

def generate_project_note(p_type: str, num_rev: int, rng: np.random.Generator) -> str:
    if num_rev > 2:
        return f"High revision count due to client changes. Requires extra QA effort for {p_type} compliance."
    if p_type == 'Residential' and rng.random() < 0.2:
        return "Project uses custom materials; coordination with specialty supplier required."
    if p_type == 'Institutional' and rng.random() < 0.3:
        return "Complex permitting process expected; started pre-application phase early."
    return f"Standard {p_type} project scope, focusing on efficient {rng.choice(['Drafting','Calculation'])} phase."

def check_holiday_overlap_pd(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if pd.isna(start) or pd.isna(end):
        return 0
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    # Nov 15 to Jan 15 window (check current and previous year)
    def overlaps_year(year:int):
        s = pd.Timestamp(year=year, month=11, day=15)
        e = pd.Timestamp(year=year+1, month=1, day=15)
        return (start < e) and (end > s)
    return int(overlaps_year(start.year) or overlaps_year(start.year - 1))


# ----------------------------
# Main generator
# ----------------------------
def generate_synthetic_data(num_projects:int=NUM_PROJECTS,
                            out_dir:str=OUT_DIR,
                            csv_prefix:str=CSV_PREFIX,
                            seed:int=SEED,
                            max_logs_per_project:int=MAX_LOGS_PER_PROJECT,
                            save_csv: bool=True) -> Dict[str, pd.DataFrame]:
    """
    Main entry. Returns dict with DataFrames:
      - df_full_training (log-level merged)
      - df_projects_metadata
      - df_timesheets (raw)
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # keep compatibility with any other np.random calls

    logger.info("Starting synthetic generation (num_projects=%d, seed=%d)", num_projects, seed)

    # 1) Build project metadata (vectorized where possible)
    start_years = rng.choice(YEARS, size=num_projects, p=START_YEAR_PROBS)
    project_types = rng.choice(PROJECT_TYPES, size=num_projects, p=[0.25, 0.35, 0.15, 0.25])
    materials = rng.choice(MATERIALS, size=num_projects)
    engineers = rng.choice(ENGINEER_LABELS, size=num_projects, p=ENGINEER_PROBABILITIES)

    areas, levels_arr, num_units_arr, heights, scope_cats = get_project_dimensions_vectorized(project_types, rng)

    # safe start dates: choose day 1-28 to avoid month-day issues, then random month
    months = rng.integers(1, 13, size=num_projects)
    days = rng.integers(1, 29, size=num_projects)
    start_dates = np.array([datetime(int(y), int(m), int(d)) for y,m,d in zip(start_years, months, days)], dtype='datetime64[s]')

    # base effort + adjustments
    base_effort = (areas * 0.1) + (levels_arr * 20.0)
    is_complex = np.isin(project_types, ['Institutional', 'Industrial'])
    base_effort = base_effort * np.where(is_complex, 1.5, 1.0)
    total_hours_est = np.maximum(10, np.round(rng.normal(base_effort, base_effort * 0.15)).astype(int))

    expected_duration_days = (np.maximum(1, (total_hours_est / 30).astype(int)) * 7) + rng.integers(5, 20, size=num_projects)
    planned_end_dates = np.array([pd.Timestamp(s) + pd.Timedelta(days=int(d)) for s,d in zip(start_dates, expected_duration_days)], dtype='datetime64[s]')

    # revisions
    num_revisions = rng.poisson(1.5, size=num_projects)
    revision_reason = np.array([ 'None' if r==0 else rng.choice(['Client Change','Code Conflict','Scope Creep','Site Condition']) for r in num_revisions ])

    # create ProjectSpec list
    projects: List[ProjectSpec] = []
    counter_map = {y:0 for y in YEARS}
    for i in range(num_projects):
        y = int(start_years[i])
        counter_map[y] += 1
        pid = f"P-{y}-{str(counter_map[y]).zfill(3)}"
        note = generate_project_note(project_types[i], int(num_revisions[i]), rng)
        spec = ProjectSpec(
            project_id=pid,
            project_type=str(project_types[i]),
            material_type=str(materials[i]),
            surface_area_m2=int(areas[i]),
            num_levels=int(levels_arr[i]),
            num_units=int(num_units_arr[i]),
            building_height_m=float(heights[i]),
            engineer_assigned=str(engineers[i]),
            project_notes=note,
            num_revisions=int(num_revisions[i]),
            revision_reason=str(revision_reason[i]),
            planned_start_date=pd.Timestamp(start_dates[i]),
            planned_end_date=pd.Timestamp(planned_end_dates[i]),
            expected_duration_days=int(expected_duration_days[i]),
            scope_category=str(scope_cats[i])
        )
        projects.append(spec)

    df_projects_metadata = pd.DataFrame([asdict(p) for p in projects])

    # 2) Timesheet generation (controlled and efficient)
    timesheet_rows: List[Dict] = []
    next_log_id = 1

    # Build project_effort_map
    project_effort_map = {p.project_id: int(np.maximum(10, np.round(((p.surface_area_m2 * 0.1) + (p.num_levels * 20.0)) * (1.5 if p.project_type in ['Institutional','Industrial'] else 1.0)))) for p in projects}

    # We iterate projects and create logs — still O(total_logs), but optimized inner logic
    for proj in projects:
        pid = proj.project_id
        p_type = proj.project_type
        total_hours = project_effort_map[pid]
        actual_total_hours = max(10.0, total_hours * float(rng.normal(1.05, 0.1)))
        # Determine reasonable #entries bounds and sample
        min_entries = max(1, int(math.ceil(actual_total_hours / 5.0)))
        max_entries = int(max(min_entries, actual_total_hours / 0.5))
        max_entries = min(max_entries, min_entries + max_logs_per_project)
        num_entries = int(rng.integers(min_entries, max_entries + 1)) if max_entries > min_entries else min_entries

        # determine date range
        planned_start = pd.Timestamp(proj.planned_start_date)
        actual_duration_days = max(1, int(proj.expected_duration_days * float(rng.normal(1.1, 0.1))))
        date_range_days = max(1, actual_duration_days)

        # target hours per task
        weights = TASK_WEIGHTS[p_type]
        target_task_hours = {task: actual_total_hours * w for task, w in weights.items()}

        # derive logs per task proportional to target hours but keep at least one entry per task
        # compute proportion once
        total_target = sum(target_task_hours.values())
        proportions = {task: (h / total_target if total_target > 0 else 1.0/len(target_task_hours)) for task,h in target_task_hours.items()}
        final_logs_per_task = {task: max(1, int(round(proportions[task] * num_entries))) for task in proportions}

        # adjust rounding difference
        diff = num_entries - sum(final_logs_per_task.values())
        if diff > 0:
            # add to largest proportions
            for task in sorted(proportions, key=proportions.get, reverse=True):
                if diff == 0:
                    break
                final_logs_per_task[task] += 1
                diff -= 1
        elif diff < 0:
            for task in sorted(proportions, key=proportions.get):
                if diff == 0:
                    break
                if final_logs_per_task[task] > 1:
                    final_logs_per_task[task] -= 1
                    diff += 1

        # produce logs
        for task, nlogs in final_logs_per_task.items():
            # estimate avg task hours to center log sampling
            logs_expected = max(1, int(max(0.1, target_task_hours[task]) / 2.5))
            avg_task_hours = (target_task_hours[task] / logs_expected) if logs_expected > 0 else 2.0
            for _ in range(nlogs):
                hours = float(max(0.25, round(rng.normal(avg_task_hours, 1.0), 2)))
                log_date = planned_start + pd.Timedelta(days=int(rng.integers(0, date_range_days)))
                subtask = str(rng.choice(SUBTASK_MAP[task]))
                employee_id = f"EMP-{int(rng.integers(1, 12)):02d}"
                timesheet_rows.append({
                    "log_id": next_log_id,
                    "project_id": pid,
                    "employee_id": employee_id,
                    "date_logged": pd.Timestamp(log_date),
                    "task_category": task,
                    "subtask_description": subtask,
                    "hours_worked": hours
                })
                next_log_id += 1

    df_timesheets = pd.DataFrame(timesheet_rows)
    if not df_timesheets.empty:
        df_timesheets['date_logged'] = pd.to_datetime(df_timesheets['date_logged'])

    # 3) Aggregations and derived features
    if not df_timesheets.empty:
        agg = df_timesheets.groupby('project_id').agg(
            corrected_start_date = ('date_logged', 'min'),
            corrected_end_date   = ('date_logged', 'max'),
            total_project_hours  = ('hours_worked', 'sum'),
            num_employees        = ('employee_id', 'nunique'),
            num_site_visits      = ('task_category', lambda x: (x=='Site Visit').sum())
        ).reset_index()

        # design hours (explicit mask)
        design_hours = (df_timesheets[df_timesheets['task_category']=='Design']
                        .groupby('project_id')['hours_worked'].sum().reset_index().rename(columns={'hours_worked':'design_hours_total'}))
        agg = agg.merge(design_hours, on='project_id', how='left')
        agg['design_hours_total'] = agg['design_hours_total'].fillna(0.0)

        agg['project_duration_days'] = (agg['corrected_end_date'] - agg['corrected_start_date']).dt.days.fillna(0).astype(int)
        agg['actual_duration_days'] = agg['project_duration_days']
        agg['avg_hours_per_employee'] = (agg['total_project_hours'] / agg['num_employees']).replace([np.inf, -np.inf], 0).fillna(0)
        agg['month_started'] = pd.to_datetime(agg['corrected_start_date']).dt.month.fillna(0).astype(int)
        agg['quarter'] = pd.to_datetime(agg['corrected_start_date']).dt.quarter.fillna(0).astype(int)
        agg['season_flag'] = agg['month_started'].apply(lambda m: 'Winter' if m in [12,1,2] else ('Spring' if m in [3,4,5] else ('Summer' if m in [6,7,8] else 'Autumn')))
        agg['holiday_period_flag'] = agg.apply(lambda row: check_holiday_overlap_pd(row['corrected_start_date'], row['corrected_end_date']), axis=1)
    else:
        agg = pd.DataFrame(columns=[
            'project_id','corrected_start_date','corrected_end_date','total_project_hours','num_employees',
            'num_site_visits','design_hours_total','project_duration_days','actual_duration_days',
            'avg_hours_per_employee','month_started','quarter','season_flag','holiday_period_flag'
        ])

    # 4) merge context + create df_full_training
    df_context = df_projects_metadata.merge(agg, on='project_id', how='left')
    df_context['floor_area_ratio'] = df_context['surface_area_m2'] / df_context['num_levels'].replace(0, np.nan)
    df_context['is_winter_day'] = df_context['month_started'].isin([12, 1, 2]).astype(int)
    df_full_training = df_timesheets.merge(df_context, on='project_id', how='left')

    # add final log-level columns
    if not df_full_training.empty:
        df_full_training['is_winter_day'] = df_full_training['date_logged'].dt.month.isin([12,1,2]).astype(int)
        df_full_training['week'] = df_full_training['date_logged'].dt.isocalendar().week.astype(int)

    # 5) Export (format dates consistently)
    final_cols_order = [
        # Log-Level Metrics (log_id removed)
        'project_id', 'employee_id', 'date_logged', 'task_category', 'subtask_description', 'hours_worked',
        'week', 'is_winter_day',
        # Project Metadata (Static)
        'project_type', 'material_type', 'scope_category', 'surface_area_m2', 'num_levels', 'num_units', 'building_height_m', 
        'engineer_assigned', 'project_notes', 'floor_area_ratio',
        # Planned Duration
        'planned_start_date', 'planned_end_date', 'expected_duration_days',
        # Actual/Calculated Metrics
        'corrected_start_date', 'corrected_end_date', 'num_site_visits', 'num_revisions', 'revision_reason',
        'project_duration_days', 'actual_duration_days', 'month_started', 'quarter', 'season_flag',
        'holiday_period_flag', 'total_project_hours', 'design_hours_total', 'avg_hours_per_employee'
    ]

    def format_dates_for_export(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
        df_export = df.copy()
        for c in date_cols:
            if c in df_export.columns:
                df_export[c] = pd.to_datetime(df_export[c], errors='coerce').dt.strftime('%Y-%m-%d')
        return df_export

    # ensure out_dir
    os.makedirs(out_dir, exist_ok=True)

    csv1 = os.path.join(out_dir, f"{csv_prefix}_full_exploration_data_v15.csv")
    csv2 = os.path.join(out_dir, f"{csv_prefix}_projects_metadata_v15.csv")
    csv3 = os.path.join(out_dir, f"{csv_prefix}_timesheets_raw_v15.csv")

    if save_csv:
        # 1. Full Exploration Data - pick columns that exist and fill missing columns if necessary
        df_export_full = df_full_training.copy()
        # ensure all final_cols_order exist:
        for c in final_cols_order:
            if c not in df_export_full.columns:
                df_export_full[c] = np.nan
        df_export_full = df_export_full[final_cols_order]
        df_export_full = format_dates_for_export(df_export_full, ['date_logged','planned_start_date','planned_end_date','corrected_start_date','corrected_end_date'])
        df_export_full.to_csv(csv1, index=False)

        # 2. Project Metadata Export (Comprehensive project-level data, excluding notes)
        metadata_cols = [
            'project_id','is_winter_day','project_type','material_type','scope_category','surface_area_m2',
            'num_levels','num_units','building_height_m','floor_area_ratio', # project_notes removed
            'planned_start_date','planned_end_date','expected_duration_days','corrected_start_date',
            'corrected_end_date','num_site_visits','project_duration_days','actual_duration_days',
            'month_started','quarter','season_flag','holiday_period_flag','total_project_hours',
            'design_hours_total','avg_hours_per_employee'
        ]
        # Use df_context (which has all merged metrics) as the source
        df_projects_metadata_export = df_context.copy()
        for c in metadata_cols:
            if c not in df_projects_metadata_export.columns:
                df_projects_metadata_export[c] = np.nan
        df_projects_metadata_export = df_projects_metadata_export[metadata_cols]
        date_cols = ['planned_start_date','planned_end_date', 'corrected_start_date', 'corrected_end_date']
        df_projects_metadata_export = format_dates_for_export(df_projects_metadata_export, date_cols)
        df_projects_metadata_export.to_csv(csv2, index=False)

        # 3. Raw Timesheets Export
        raw_cols = ['log_id','project_id','employee_id','date_logged','task_category','subtask_description','hours_worked']
        df_timesheets_export = df_timesheets.copy()
        for c in raw_cols:
            if c not in df_timesheets_export.columns:
                df_timesheets_export[c] = np.nan
        df_timesheets_export = df_timesheets_export[raw_cols]
        df_timesheets_export = format_dates_for_export(df_timesheets_export, ['date_logged'])
        df_timesheets_export.to_csv(csv3, index=False)
        
        logger.info("Saved CSVs: %s, %s, %s", csv1, csv2, csv3)
        logger.info("Note: projects_metadata_v15.csv now contains aggregated metrics, excluding project notes.")

    return {
        "df_full_training": df_full_training,
        "df_projects_metadata": df_projects_metadata,
        "df_timesheets": df_timesheets
    }


# ----------------------------
# CLI Run
# ----------------------------
if __name__ == "__main__":
    outputs = generate_synthetic_data(num_projects=NUM_PROJECTS, out_dir=OUT_DIR, csv_prefix=CSV_PREFIX, seed=SEED)
    logger.info("Done. Generated %d log entries across %d projects.",
                len(outputs['df_timesheets']), NUM_PROJECTS)