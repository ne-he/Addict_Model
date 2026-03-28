"""
src/preprocessing.py
Preprocessing pipeline — identik dengan notebook AOL_Machine_Learning.ipynb
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ---------------------------------------------------------------------------
# 2.1  clean_sleep_hours
# ---------------------------------------------------------------------------

def clean_sleep_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Strip surrounding quotes from Sleep_Hours and convert to float.

    Notebook cell 43:
        df["Sleep_Hours"] = df["Sleep_Hours"].astype(str).str.strip('"').astype(float)
    """
    df = df.copy()
    df["Sleep_Hours"] = df["Sleep_Hours"].astype(str).str.strip('"').astype(float)
    return df


# ---------------------------------------------------------------------------
# 2.2  handle_missing_values
# ---------------------------------------------------------------------------

def handle_missing_values(
    df: pd.DataFrame,
    num_medians: dict,
    cat_modes: dict,
) -> pd.DataFrame:
    """Impute missing values using statistics computed from the training set.

    Notebook cells 74 & 80:
        - Numerical: fillna(median from X_train)
        - Categorical: fillna(mode from X_train)

    Args:
        df: Input DataFrame.
        num_medians: {column_name: median_value} from training set.
        cat_modes:   {column_name: mode_value}   from training set.
    """
    df = df.copy()
    for col, val in num_medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    for col, val in cat_modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


# ---------------------------------------------------------------------------
# 2.3  encode_categorical
# ---------------------------------------------------------------------------

CAT_COLS = ["Gender", "Phone_Usage_Purpose"]


def encode_categorical(df: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    """Apply fitted OneHotEncoder to Gender and Phone_Usage_Purpose.

    Notebook cell 94:
        ohe = OneHotEncoder(drop=["Other", "Other"], sparse_output=False,
                            handle_unknown="ignore")
        ohe.fit(X_train[cat_cols])

    The encoded columns replace the original categorical columns.
    """
    df = df.copy()
    encoded = ohe.transform(df[CAT_COLS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(CAT_COLS),
        index=df.index,
    )
    df = df.drop(columns=CAT_COLS)
    return pd.concat([df, encoded_df], axis=1)
