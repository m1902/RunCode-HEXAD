import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

def standardize_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy[columns] = StandardScaler().fit_transform(df[columns])
    return df_copy

def fit_ols_model(df: pd.DataFrame, formula: str):
    model = smf.ols(formula=formula, data=df).fit()
    return model

def get_default_ols_formulas():
    return {
        "base_model": "test ~ pre_test + n_solved_tasks",
        "hexad_main": "test ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",
        "hexad_interactions": """
            test ~ pre_test + n_solved_tasks
            + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R
            + HEXAD_F:HEXAD_D + HEXAD_S:HEXAD_P + HEXAD_A:HEXAD_R
        """,
    }

def get_ols_formula_by_name(name: str) -> str:
    formulas = {
        "hexad_outcomes_main": "test ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",
        "hexad_outcomes_interactions": """
            test ~ pre_test
            + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R
            + HEXAD_F:HEXAD_D + HEXAD_S:HEXAD_P + HEXAD_A:HEXAD_R
            """,
        "hexad_tasks_main": """
            test ~ pre_test + n_solved_tasks
            + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R
            """,
        "hexad_tasks_interactions": """
            test ~ pre_test + n_solved_tasks
            + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R
            + HEXAD_F:HEXAD_D + HEXAD_S:HEXAD_P + HEXAD_A:HEXAD_R
            """,
        "regularity_day_main": "test ~ pre_test + entropy_day + burstiness_day + gini_day",
        "regularity_week_main": "test ~ pre_test + entropy_week + burstiness_week + gini_week",
        "entropy_day_main": "entropy_day ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",
        "gini_day_main": "gini_day ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",
        "burstiness_day_main": "burstiness_day ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",
        "entropy_week_main": "entropy_week ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",
        "gini_week_main": "gini_week ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",
        "burstiness_week_main": "burstiness_week ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",

    }

    if name not in formulas:
        raise ValueError(f"Model '{name}' is not defined. Choose from: {list(formulas.keys())}")

    return formulas[name]
