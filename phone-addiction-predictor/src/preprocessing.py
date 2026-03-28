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
