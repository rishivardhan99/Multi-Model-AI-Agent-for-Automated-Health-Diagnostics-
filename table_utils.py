# table_utils.py
"""
Creates pivot views and summary tables for interpreted parameter rows.
"""

import pandas as pd


def pivot_params_to_wide(df):
    df = df.copy()

    df["param_col"] = df["canonical"]

    return df.pivot_table(
        index=["patient_id", "filename", "age", "gender"],
        columns="param_col",
        values="value_num",   # use numeric values for downstream processing
        aggfunc="first"
    ).reset_index()


def summary_counts_by_interpretation(df):
    return df.groupby(["canonical", "interpretation"]).size().unstack(fill_value=0)


def patient_level_flag(df):
    df = df.copy()
    df["abn"] = df["interpretation"].isin(["low", "high", "borderline_low", "borderline_high"])
    out = df.groupby("patient_id").agg(
        n_params=("canonical", "count"),
        n_abnormal=("abn", "sum")
    )
    out["any_abnormal"] = out["n_abnormal"] > 0
    return out.reset_index()
