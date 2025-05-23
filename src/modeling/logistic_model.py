import pandas as pd
import statsmodels.formula.api as smf

def add_participation_variable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["participated"] = (df["n_days_active"] >= 1).astype(int)
    return df

def add_shallow_engagement_variable(df: pd.DataFrame, quantile: float = 0.25) -> pd.DataFrame:
    """Add binary variable: 1 if user is in bottom quantile of n_days_active"""
    df = df.copy()
    threshold = df["n_days_active"].quantile(quantile)
    df["shallow_engagement"] = (df["n_days_active"] <= threshold).astype(int)
    return df

def add_late_worker_variable(df: pd.DataFrame, quantile: float = 0.9) -> pd.DataFrame:
    """Add binary variable: 1 if user is in the top quantile of percent_late_work"""
    df = df.copy()
    threshold = df["percent_late_work"].quantile(quantile)
    df["late_worker"] = (df["percent_late_work"] > threshold).astype(int)
    return df

def fit_logit_model(df: pd.DataFrame, formula: str):
    return smf.logit(formula=formula, data=df).fit()

def get_logit_formula_by_name(name: str) -> str:
    formulas = {
        "participated_main": "participated ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",
        "participated_interactions": "participated ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R"
            "+ HEXAD_F:HEXAD_D + HEXAD_S:HEXAD_P + HEXAD_A:HEXAD_R",
        "shallow_engagement_main": "shallow_engagement ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",
        "shallow_engagement_interactions": "shallow_engagement ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R"
                                           "+ HEXAD_F:HEXAD_D + HEXAD_S:HEXAD_P + HEXAD_A:HEXAD_R",
        "late_work_main": "late_worker ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",
        "late_work_interactions": "late_worker ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R"
                                  "+ HEXAD_F:HEXAD_D + HEXAD_S:HEXAD_P + HEXAD_A:HEXAD_R",
    }

    if name not in formulas:
        raise ValueError(f"Model '{name}' is not defined. Choose from: {list(formulas.keys())}")

    return formulas[name]
