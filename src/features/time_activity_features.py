import pandas as pd
import numpy as np
from scipy.stats import entropy

def calculate_entropy(series):
    counts = series.value_counts()
    probabilities = counts / counts.sum()
    return entropy(probabilities, base=2)

def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    features = []

    for user_id, group in data.groupby("user_id"):
        days = sorted(group["day"].unique())
        counts = group["day"].value_counts().sort_index()

        first_day = days[0]
        last_day = days[-1]
        timespan = last_day - first_day

        cumsum_counts = counts.cumsum()
        total_subs = cumsum_counts.iloc[-1]
        median_day_of_activity = cumsum_counts.index[cumsum_counts >= total_subs / 2][0]

        n_solved_tasks = group.loc[group["score"] == 1, "task_id"].nunique()
        n_attempted_tasks = group["task_id"].nunique()

        # Solve rate (guard against division by 0)
        solve_rate = n_solved_tasks / n_attempted_tasks if n_attempted_tasks > 0 else 0
        # %LateWork
        late_work = group[group["day"] >= 97]["day"].count()
        percent_late_work = late_work / len(group)

        # %LateWork_25: last 25% (days 90–120)
        late_work_25 = group[group["day"] >= 90]["day"].count()
        percent_late_work_25 = late_work_25 / len(group)

        # %LateWork_10: last 10% (days 108–120)
        late_work_10 = group[group["day"] >= 108]["day"].count()
        percent_late_work_10 = late_work_10 / len(group)

        # === Add submission type labels for entropy
        group = group.copy()  # avoid SettingWithCopyWarning
        group["non_compilable"] = (group["error_count"] > 0) & (group["test_count"] == 0)
        group["correct"] = group["score"] > 0

        # === Entropy features
        def get_entropy(df, time_col):
            return calculate_entropy(df[time_col]) if not df.empty else 0

        entropy_day_all = get_entropy(group, "day")
        entropy_week_all = get_entropy(group, "week")
        entropy_day_noncomp = get_entropy(group[group["non_compilable"]], "day")
        entropy_week_noncomp = get_entropy(group[group["non_compilable"]], "week")
        entropy_day_correct = get_entropy(group[group["correct"]], "day")
        entropy_week_correct = get_entropy(group[group["correct"]], "week")

        # Gini_day
        daily_counts = counts.values
        sorted_counts = np.sort(daily_counts)
        n = len(sorted_counts)
        gini_day = (2 * np.arange(1, n + 1) - n - 1) @ sorted_counts / (
                n * sorted_counts.sum()) if sorted_counts.sum() > 0 else 0

        # Inter-quartile activity window
        cumsum = counts.cumsum() / counts.sum()
        q25 = counts.index[cumsum >= 0.25][0]
        q75 = counts.index[cumsum >= 0.75][0]
        iqr_window = q75 - q25

        # Weeks active
        weeks = group["week"]
        n_weeks_active = weeks.nunique()
        n_days_active = len(days)
        # MeanGap & SDGap
        if len(days) > 1:
            day_gaps = np.diff(days)
            mean_gap = np.mean(day_gaps)
            sd_gap = np.std(day_gaps)
        else:
            mean_gap = sd_gap = 0

        # Burstiness
        mu = daily_counts.mean()
        sigma = daily_counts.std()
        burstiness_day = (sigma - mu) / (sigma + mu) if (sigma + mu) > 0 else 0

        # Weekly burstiness_day
        weekly_counts = group.groupby("week").size()
        mu_week = weekly_counts.mean()
        sigma_week = weekly_counts.std()
        burstiness_week = (sigma_week - mu_week) / (sigma_week + mu_week) if (sigma_week + mu_week) > 0 else 0

        # Gini week
        weekly_counts = group.groupby("week").size().values
        sorted_weekly = np.sort(weekly_counts)
        n = len(sorted_weekly)

        # Gini coefficient for weekly activity
        gini_week = (
                (2 * np.arange(1, n + 1) - n - 1) @ sorted_weekly
                / (n * sorted_weekly.sum())
        ) if sorted_weekly.sum() > 0 else 0

        # Entropy_days
        probs = daily_counts / daily_counts.sum() if daily_counts.sum() > 0 else np.array([1])
        entropy_day = entropy(probs, base=np.e)

        # Entropy_weeks
        weekly_probs = weekly_counts / weekly_counts.sum() if weekly_counts.sum() > 0 else np.array([1])
        entropy_week = entropy(weekly_probs, base=np.e)

        # Acceleration/Deceleration
        cum_counts = group.groupby("day").size().sort_index().cumsum()
        early = cum_counts[cum_counts.index < 40]
        late = cum_counts[cum_counts.index >= 80]

        # Early slope
        if len(early) > 1:
            early_slope = (early.iloc[-1] - early.iloc[0]) / (early.index[-1] - early.index[0])
        else:
            early_slope = 0

        # Late slope
        if len(late) > 1:
            late_slope = (late.iloc[-1] - late.iloc[0]) / (late.index[-1] - late.index[0])
        else:
            late_slope = 0

        # Slope ratio
        if early_slope == 0:
            if late_slope == 0:
                slope_ratio = 0  # no change at all
            else:
                slope_ratio = np.inf  # fully backloaded
        else:
            slope_ratio = late_slope / early_slope

        slope_ratio = min(slope_ratio, 10)

        features.append({
            "user_id": user_id,
            "first_day": first_day,
            "last_day": last_day,
            "time_span": timespan,
            "percent_late_work": percent_late_work,
            "percent_late_work_25": percent_late_work_25,
            "percent_late_work_10": percent_late_work_10,
            "entropy_day_all": entropy_day_all,
            "entropy_week_all": entropy_week_all,
            "entropy_day_noncomp": entropy_day_noncomp,
            "entropy_week_noncomp": entropy_week_noncomp,
            "entropy_day_correct": entropy_day_correct,
            "entropy_week_correct": entropy_week_correct,
            "gini_day": gini_day,
            "gini_week": gini_week,
            "burstiness_day": burstiness_day,
            "burstiness_week": burstiness_week,
            "entropy_day": entropy_day,
            "entropy_week": entropy_week,
            "iqr_window": iqr_window,
            "n_weeks_active": n_weeks_active,
            "n_days_active": n_days_active,
            "mean_gap": mean_gap,
            "sd_gap": sd_gap,
            "early_slope": early_slope,
            "late_slope": late_slope,
            "slope_ratio": slope_ratio,
            "median_day_of_activity": median_day_of_activity,
            "n_solved_tasks": n_solved_tasks,
            "n_attempted_tasks": n_attempted_tasks,
            "solve_rate": solve_rate,
        })

    return pd.DataFrame(features)
