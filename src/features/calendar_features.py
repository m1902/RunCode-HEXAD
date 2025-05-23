import pandas as pd

COHORT_START_DATES = {
    2021: pd.Timestamp("2021-10-06"),
    2022: pd.Timestamp("2022-10-13"),
    2023: pd.Timestamp("2023-10-13"),
    2024: pd.Timestamp("2024-10-06"),
}

def add_calendar_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["cohort_start"] = data["year_x"].map(COHORT_START_DATES)
    data["day"] = (data["upload_date"] - data["cohort_start"]).dt.days
    data["week"] = (data["upload_date"] - data["cohort_start"]).dt.days // 7
    return data
