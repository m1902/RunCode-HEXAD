import pandas as pd
from src.modeling.ols_model import standardize_columns, fit_ols_model

# Load data
df = pd.read_csv("data/preprocessed/student_time_features_2021_2024.csv")

# Columns to scale
scale_cols = ['pre_test', 'HEXAD_P', 'HEXAD_S', 'HEXAD_F', 'HEXAD_A', 'HEXAD_D', 'HEXAD_R']
df_scaled = standardize_columns(df, scale_cols)

# Define models

ols_formulas = {
    "base": "test ~ pre_test + n_solved_tasks",
    "hexad_main": "test ~ pre_test + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R",
    "hexad_interactions": """
        test ~ pre_test + n_solved_tasks
        + HEXAD_P + HEXAD_S + HEXAD_F + HEXAD_A + HEXAD_D + HEXAD_R
        + HEXAD_F:HEXAD_D + HEXAD_S:HEXAD_P + HEXAD_A:HEXAD_R
    """,
}

# Fit models and store in a dictionary
ols_models = {
    name: fit_ols_model(df_scaled, formula)
    for name, formula in ols_formulas.items()
}
