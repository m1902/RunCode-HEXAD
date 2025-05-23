import pandas as pd
import statsmodels.formula.api as smf

def check_overdispersion(df: pd.DataFrame, count_col: str = "n_solved_tasks"):
    mean_val = df[count_col].mean()
    var_val = df[count_col].var()
    is_overdispersed = var_val > mean_val
    return is_overdispersed, mean_val, var_val

def fit_negative_binomial(df: pd.DataFrame, formula: str):
    return smf.negativebinomial(formula, data=df).fit()

def get_nb_formula_by_name(name: str) -> str:
    formulas = {
        "solved_tasks_main": "n_solved_tasks ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",
        "solved_tasks_interactions": (
            "n_solved_tasks ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R "
            "+ HEXAD_F:HEXAD_D + HEXAD_S:HEXAD_P + HEXAD_A:HEXAD_R"
        ),
        "first_day_main" : "first_day ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + "
                   "HEXAD_A + HEXAD_D + HEXAD_R",
        "first_day_interactions": "first_day ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + "
                   "HEXAD_A + HEXAD_D + HEXAD_R"
                                "+ HEXAD_F:HEXAD_D + HEXAD_S:HEXAD_P + HEXAD_A:HEXAD_R",
        "median_day_main": "median_day_of_activity ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + "
                   "HEXAD_A + HEXAD_D + HEXAD_R",
        "median_day_interactions": "median_day_of_activity ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + "
                   "HEXAD_A + HEXAD_D + HEXAD_R"
                                "+ HEXAD_F:HEXAD_D + HEXAD_S:HEXAD_P + HEXAD_A:HEXAD_R",
        "days_of_activity_main": "n_days_active ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + "
                   "HEXAD_A + HEXAD_D + HEXAD_R",
        "days_of_activity_interactions": "n_days_active ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + "
                   "HEXAD_A + HEXAD_D + HEXAD_R"
                                "+ HEXAD_F:HEXAD_D + HEXAD_S:HEXAD_P + HEXAD_A:HEXAD_R",
        "weeks_of_activity_main": "n_weeks_active ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + "
                   "HEXAD_A + HEXAD_D + HEXAD_R",
        "weeks_of_activity_interactions": "n_weeks_active ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + "
                   "HEXAD_A + HEXAD_D + HEXAD_R"
                                "+ HEXAD_F:HEXAD_D + HEXAD_S:HEXAD_P + HEXAD_A:HEXAD_R",


    }

    if name not in formulas:
        raise ValueError(f"Model '{name}' is not defined. Choose from: {list(formulas.keys())}")

    return formulas[name]
