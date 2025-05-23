from src.data.loader import load_data
from src.features.calendar_features import add_calendar_columns
from src.features.time_activity_features import extract_features
import pandas as pd

# === Load Data ===
submissions, students = load_data(
    "data/raw/task_submit_2021_2024.csv",
    "data/raw/users_hexad_2021_2024.csv"
)

# === Merge and preprocess ===
data = submissions.merge(students, on="user_id", how="inner")
data = add_calendar_columns(data)
features_df = extract_features(data)

# === Merge with profiles ===
final_df = features_df.merge(students, on="user_id", how="right")
fill_cols = ["solve_rate", "n_solved_tasks", "n_attempted_tasks", "n_days_active", "n_weeks_active"]
final_df[fill_cols] = final_df[fill_cols].fillna(0)

# === Save ===
final_df.to_csv("data/preprocessed/student_time_features_2021_2024.csv", index=False)
