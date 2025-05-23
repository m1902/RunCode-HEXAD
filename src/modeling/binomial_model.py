import pandas as pd
import statsmodels.api as sm

def prepare_solve_rate_data(df: pd.DataFrame):
    """Filter and prepare data for binomial modeling: successes vs. failures"""
    df_valid = df[df["n_attempted_tasks"] > 0].copy()
    df_valid["failures"] = df_valid["n_attempted_tasks"] - df_valid["n_solved_tasks"]

    assert (df_valid["failures"] >= 0).all(), "Some failures < 0"

    endog = df_valid[["n_solved_tasks", "failures"]]  # successes, failures
    exog = df_valid[[
        "pre_test", "HEXAD_P", "HEXAD_S", "HEXAD_F", "HEXAD_A", "HEXAD_D", "HEXAD_R"
    ]]
    exog = sm.add_constant(exog)

    return endog, exog

def fit_binomial_model(endog, exog):
    """Fit binomial GLM on solve rate data"""
    model = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()
    return model
