"""Detect data quality issues without assuming any column names."""
import re
import warnings
import numpy as np
import pandas as pd


def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def missing_report(df):
    m = df.isna().sum()
    return pd.DataFrame({"Column": m.index, "Missing": m.values,
                         "Missing %": (m.values / len(df) * 100).round(2)})


def outlier_report(df):
    rows = []
    for c in df.select_dtypes(include=np.number).columns:
        s = df[c].dropna()
        if s.empty:
            continue
        lo, hi = iqr_bounds(s)
        n = int(((s < lo) | (s > hi)).sum())
        rows.append({"Column": c, "Outliers": n, "Outlier %": round(n / len(s) * 100, 2),
                     "Lower Bound": round(lo, 3), "Upper Bound": round(hi, 3)})
    return pd.DataFrame(rows, columns=["Column", "Outliers", "Outlier %", "Lower Bound", "Upper Bound"])


def suggest_dtypes(df):
    """Text columns that are mostly numeric or dates -> suggest conversion."""
    out = []
    for c in df.select_dtypes(include="object").columns:
        s = df[c].dropna().astype(str).str.strip()
        s = s[s != ""]
        if s.empty:
            continue
        num = pd.to_numeric(s.str.replace(r"[,$%]", "", regex=True), errors="coerce").notna().mean()
        if num >= 0.8:
            out.append({"Column": c, "Current": "object", "Suggested": "numeric"})
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt = pd.to_datetime(s, errors="coerce").notna().mean()
        if dt >= 0.8:
            out.append({"Column": c, "Current": "object", "Suggested": "datetime"})
    return pd.DataFrame(out, columns=["Column", "Current", "Suggested"])


def _norm(v):
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def inconsistent_categories(df, max_unique=50):
    """Categories that differ only by case/whitespace (e.g. 'North', ' north ', 'NORTH')."""
    rows = []
    for c in df.select_dtypes(include="object").columns:
        vals = df[c].dropna().astype(str).unique()
        if len(vals) > max_unique or len(vals) == 0:
            continue
        groups = {}
        for v in vals:
            groups.setdefault(_norm(v), []).append(v)
        bad = {k: v for k, v in groups.items() if len(v) > 1}
        if bad:
            rows.append({"Column": c, "Groups": len(bad),
                         "Examples": "; ".join(" | ".join(map(repr, v)) for v in list(bad.values())[:3])})
    return pd.DataFrame(rows, columns=["Column", "Groups", "Examples"])


def quality_report(df):
    """Full report as a dict of DataFrames + headline numbers."""
    n = max(len(df), 1)
    empty_rows = int(df.isna().all(axis=1).sum())
    dups = int(df.duplicated().sum())
    blank = int(sum((df[c].astype(str).str.strip() == "").sum() for c in df.select_dtypes("object")))
    miss, outl = missing_report(df), outlier_report(df)
    types, cats = suggest_dtypes(df), inconsistent_categories(df)
    cells = max(df.size, 1)
    miss_total = int(df.isna().sum().sum())
    out_total = int(outl["Outliers"].sum()) if len(outl) else 0
    summary = pd.DataFrame([
        ["Total rows", len(df), ""],
        ["Total columns", df.shape[1], ""],
        ["Missing cells", miss_total, f"{miss_total / cells * 100:.2f}%"],
        ["Duplicate rows", dups, f"{dups / n * 100:.2f}%"],
        ["Fully empty rows", empty_rows, f"{empty_rows / n * 100:.2f}%"],
        ["Blank-string cells", blank, f"{blank / cells * 100:.2f}%"],
        ["Total outliers (IQR)", out_total, ""],
        ["Columns with wrong dtype", len(types), ""],
        ["Columns with inconsistent text", len(cats), ""],
    ], columns=["Metric", "Count", "Percentage"])
    # Simple 0-100 quality score: penalise missing cells, duplicates and outliers
    penalty = miss_total / cells * 200 + dups / n * 100 + (out_total / (n * max(1, len(outl))) * 100 if len(outl) else 0)
    return {"summary": summary, "missing": miss, "outliers": outl, "dtypes": types,
            "categories": cats, "duplicates": dups, "empty_rows": empty_rows,
            "score": round(max(100 - penalty, 0), 1)}
