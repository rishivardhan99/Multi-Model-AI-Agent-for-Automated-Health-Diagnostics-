#!/usr/bin/env python3
"""
extractor/run_model1_on_csv.py
Run Model-1 on per-report structured CSVs and produce per-report final CSVs
matching Streamlit UI final CSV shape.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_and_standardization import read_structured_csv, standardize_dataframe, _norm_key_simple
from model1_interpretation import interpret_dataframe

STRUCTURED_DIR = ROOT / "outputs" / "structured_per_report"
FINAL_DIR = ROOT / "outputs" / "model1_per_report"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = ROOT / "outputs"   # used by interpret_dataframe save


def _infer_param_column(df):
    candidates = ['canonical', 'canonical_name', 'parameter', 'param', 'test_name', 'test', 'name']
    for kw in candidates:
        for c in df.columns:
            if kw in c.lower():
                return c
    # fallback: first object column with reasonable cardinality
    for c in df.columns:
        if df[c].dtype == object:
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= min(500, len(df)//1):
                return c
    return None


def _infer_status_column(df):
    candidates = ['status', 'interpret', 'flag', 'classification']
    for kw in candidates:
        for c in df.columns:
            if kw in c.lower():
                return c
    return None


def _infer_note_column(df):
    candidates = ['note', 'explain', 'comment', 'interpretation', 'message']
    for kw in candidates:
        for c in df.columns:
            if kw in c.lower():
                return c
    return None


def process_one(struct_path: Path):
    stem = struct_path.stem.replace(".structured", "") if struct_path.stem.endswith(".structured") else struct_path.stem
    # 1) read long form and standardize
    df_long = read_structured_csv(struct_path)
    df_std = standardize_dataframe(df_long)

    # --------------------------------------------------------------
    # DETECT implausible parameters (collect canonical names) BEFORE dropping rows
    # --------------------------------------------------------------
    invalid_params = set()
    if "valid" in df_std.columns and "invalid_reason" in df_std.columns:
        bad_rows = df_std[
            (df_std["valid"] == False) &
            (df_std["invalid_reason"] == "implausible_value")
        ]
        # collect canonical keys (normalized) for later blanking of wide CSV columns
        for p in bad_rows["canonical"].dropna().unique():
            invalid_params.add(_norm_key_simple(str(p)))

    # --------------------------------------------------------------
    # HARD DROP: remove implausible rows BEFORE interpretation
    # --------------------------------------------------------------
    if "valid" in df_std.columns and "invalid_reason" in df_std.columns:
        before = len(df_std)
        df_std = df_std[
            ~(
                (df_std["valid"] == False) &
                (df_std["invalid_reason"] == "implausible_value")
            )
        ].copy()
        after = len(df_std)
        print(f"[Model-1] HARD DROP removed {before - after} implausible rows")
        # Debug artifact (one-time): shows what's left after enforcement
        df_std.to_csv(ROOT / "outputs" / f"{stem}.debug_after_hard_drop.csv", index=False)

    # 2) interpret using model1 (same as Streamlit)
    try:
        df_interp, units_used = interpret_dataframe(
            df_std,
            save_outputs=True,
            out_dir=OUT_DIR,
            basename=stem,
            save_json=True,
            border_frac=0.10,
            units_tracking=True,
        )
    except Exception as e:
        print("Interpretation error for", struct_path.name, ":", e)
        df_interp = pd.DataFrame()
        units_used = {}

    # 3) start from base (wide numeric) — keep patient metadata
    base = pd.read_csv(struct_path, dtype=str).fillna("")
    # --------------------------------------------------------------
    # Remove implausible params in the wide CSV (value + status + note)
    # This blanks the original numeric cells so final CSV no longer contains
    # implausible numeric values that were dropped earlier.
    # --------------------------------------------------------------
    def _norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    if invalid_params:
        for col in list(base.columns):
            if col in ("filename", "patient_id", "age", "gender"):
                continue

            root = col
            if root.endswith("_status"):
                root = root[:-7]
            elif root.endswith("_note"):
                root = root[:-5]

            if _norm(root) in invalid_params:
                base[col] = ""

    # compute which numeric parameters are actually present in this structured CSV
    keep_core = {'filename', 'patient_id', 'age', 'gender'}
    numeric_cols = set()
    for c in base.columns:
        if c in keep_core:
            continue
        # treat blanks/empty strings as missing
        ser = base[c].dropna().astype(str).map(lambda x: x.strip()).replace('', pd.NA)
        if ser.notna().any():
            numeric_cols.add(c)

    if not df_interp.empty and numeric_cols:
        param_col = _infer_param_column(df_interp) or "canonical"
        status_col = _infer_status_column(df_interp)
        note_col = _infer_note_column(df_interp)
        index_col = 'filename' if 'filename' in df_interp.columns else ('patient_id' if 'patient_id' in df_interp.columns else None)

        if index_col is None:
            print("Interpretation output lacks filename/patient_id — writing numeric only final CSV for", struct_path.name)
        else:
            # Filter interpretation rows to only those parameters present in this report
            # df_interp canonical names should match the column names in the structured CSV
            if param_col not in df_interp.columns:
                print(f"Warning: expected parameter column '{param_col}' not found in interpretation output; skipping status/note pivot for {struct_path.name}")
            else:
                def _canon(s):
                    return str(s).lower().replace(" ", "").replace("_", "").replace(".", "")

                canon_numeric = {_canon(c) for c in numeric_cols}

                df_interp_filtered = df_interp[
                    df_interp[param_col].map(_canon).isin(canon_numeric)
                ].copy()

                if df_interp_filtered.empty:
                    # nothing to merge for this file
                    pass
                else:
                    # pivot status columns (only for filtered params)
                    if status_col and status_col in df_interp_filtered.columns:
                        try:
                            status_pivot = df_interp_filtered.pivot_table(
                                index=index_col,
                                columns=param_col,
                                values=status_col,
                                aggfunc=lambda x: '; '.join(x.dropna().astype(str).unique())
                            )
                            status_pivot.columns = [f"{str(c)}_status" for c in status_pivot.columns]
                            base = base.merge(status_pivot, left_on='filename', right_index=True, how='left')
                        except Exception as e:
                            print("Failed to pivot status for", struct_path.name, ":", e)
                    # pivot note columns
                    if note_col and note_col in df_interp_filtered.columns:
                        try:
                            note_pivot = df_interp_filtered.pivot_table(
                                index=index_col,
                                columns=param_col,
                                values=note_col,
                                aggfunc=lambda x: '; '.join(x.dropna().astype(str).unique())
                            )
                            note_pivot.columns = [f"{str(c)}_note" for c in note_pivot.columns]
                            base = base.merge(note_pivot, left_on='filename', right_index=True, how='left')
                        except Exception as e:
                            print("Failed to pivot note for", struct_path.name, ":", e)
    # else: no interp rows or no numeric cols — leave base as numeric-only

    # 4) Remove columns that are entirely empty (treat blanks as missing)
    import numpy as np
    cols_to_drop = []
    for c in list(base.columns):
        if c in keep_core:
            continue
        series = base[c]
        non_null = series.dropna().astype(str).map(lambda x: x.strip()).replace('', np.nan).dropna()
        if non_null.empty:
            cols_to_drop.append(c)

    if cols_to_drop:
        base.drop(columns=cols_to_drop, inplace=True)

    # 5) reset index and save final CSV (per report)
    out_path = FINAL_DIR / f"{stem}.model1_final.csv"
    base.reset_index(drop=True, inplace=True)
    base.to_csv(out_path, index=False)
    print("Final →", out_path.name)


def main():
    files = sorted(STRUCTURED_DIR.glob("*.structured.csv"))
    if not files:
        print("No structured CSVs found in", STRUCTURED_DIR)
        return
    for f in files:
        process_one(f)


if __name__ == "__main__":
    main()
