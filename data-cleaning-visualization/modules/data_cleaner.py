"""Cleaning operations. Every function returns a NEW DataFrame (originals are never mutated)."""
import re
import numpy as np
import pandas as pd
from .data_quality import iqr_bounds


def _std_name(c):
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip().lower())).strip("_")


def standardize_columns(df):
    d = df.copy()
    d.columns = [_std_name(c) or f"col_{i}" for i, c in enumerate(d.columns)]
    return d


def drop_columns(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns])


def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


def drop_empty_rows(df):
    return df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all").reset_index(drop=True)


def convert_dtype(df, col, target):
    d = df.copy()
    if target == "numeric":
        d[col] = pd.to_numeric(d[col].astype(str).str.replace(r"[,$%]", "", regex=True), errors="coerce")
    elif target == "datetime":
        d[col] = pd.to_datetime(d[col], errors="coerce")
    elif target == "category":
        d[col] = d[col].astype("category")
    elif target == "string":
        d[col] = d[col].astype("string")
    return d


def standardize_text(df, cols=None, case="title"):
    """Trim, collapse spaces and unify case so 'north ', 'NORTH' -> 'North'."""
    d = df.copy()
    for c in (cols if cols is not None else d.select_dtypes("object").columns):
        s = d[c].where(d[c].isna(), d[c].astype(str).str.strip().str.replace(r"\s+", " ", regex=True))
        s = {"title": s.str.title, "lower": s.str.lower, "upper": s.str.upper}.get(case, lambda: s)()
        d[c] = s.replace("", np.nan)
    return d


def handle_missing(df, strategy="median", cols=None):
    """strategy: mean | median | mode | remove. Mean/median fall back to mode for text columns."""
    d = df.copy()
    cols = [c for c in (cols if cols is not None else d.columns) if d[c].isna().any()]
    if strategy == "remove":
        return d.dropna(subset=cols).reset_index(drop=True)
    for c in cols:
        if strategy in ("mean", "median") and pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].fillna(d[c].mean() if strategy == "mean" else d[c].median())
        else:
            m = d[c].mode(dropna=True)
            if not m.empty:
                d[c] = d[c].fillna(m.iloc[0])
    return d


def handle_outliers(df, method="cap", cols=None, k=1.5):
    """method: cap (clip to IQR bounds) | remove (drop rows)."""
    d = df.copy()
    cols = cols if cols is not None else d.select_dtypes(include=np.number).columns
    mask = pd.Series(False, index=d.index)
    for c in cols:
        s = d[c].dropna()
        if s.empty:
            continue
        lo, hi = iqr_bounds(s, k)
        if method == "cap":
            d[c] = d[c].clip(lo, hi)
        else:
            mask |= (d[c] < lo) | (d[c] > hi)
    return d[~mask].reset_index(drop=True) if method == "remove" else d


def apply_pipeline(df, opts):
    """Run the selected steps in a sensible order. Returns (clean_df, log)."""
    d, log = df.copy(), []

    def step(name, new):
        nonlocal d
        log.append(f"{name}: {d.shape[0]}x{d.shape[1]} -> {new.shape[0]}x{new.shape[1]}")
        d = new

    if opts.get("drop_cols"): step("Drop columns", drop_columns(d, opts["drop_cols"]))
    if opts.get("empty_rows"): step("Remove empty rows", drop_empty_rows(d))
    for col, tgt in (opts.get("dtypes") or {}).items():
        if col in d.columns: step(f"Convert '{col}' to {tgt}", convert_dtype(d, col, tgt))
    if opts.get("text"): step("Standardize text", standardize_text(d, case=opts.get("case", "title")))
    if opts.get("dups"): step("Remove duplicates", remove_duplicates(d))
    if opts.get("outliers") in ("cap", "remove"): step(f"Outliers ({opts['outliers']})", handle_outliers(d, opts["outliers"]))
    if opts.get("missing"): step(f"Missing values ({opts['missing']})", handle_missing(d, opts["missing"]))
    if opts.get("std_cols"): step("Standardize column names", standardize_columns(d))  # last, so user-picked names still match
    return d, log
