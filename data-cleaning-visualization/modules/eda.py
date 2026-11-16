"""Exploratory analysis and automatic insights."""
import numpy as np
import pandas as pd
from .data_quality import outlier_report


def numeric_cols(df): return list(df.select_dtypes(include=np.number).columns)
def datetime_cols(df): return list(df.select_dtypes(include="datetime").columns)
def categorical_cols(df, max_unique=30):
    return [c for c in df.select_dtypes(include=["object", "category", "string"]).columns if df[c].nunique() <= max_unique]


def descriptive_stats(df):
    nums = df[numeric_cols(df)]
    if nums.empty:
        return pd.DataFrame()
    t = nums.describe().T
    t["median"] = nums.median()
    t["mode"] = [nums[c].mode().iloc[0] if not nums[c].mode().empty else np.nan for c in nums]
    t["skew"] = nums.skew()
    return t[["count", "mean", "median", "mode", "std", "min", "25%", "75%", "max", "skew"]].round(3)


def correlation(df):
    nums = df[numeric_cols(df)]
    return nums.corr() if nums.shape[1] > 1 else pd.DataFrame()


def category_summary(df, cat, num=None):
    """Counts per category; with a numeric column also count/mean/sum."""
    if num is None:
        return df[cat].value_counts().rename_axis(cat).reset_index(name="count")
    return df.groupby(cat, observed=True)[num].agg(["count", "mean", "sum"]).round(2).sort_values("sum", ascending=False).reset_index()


def strong_correlations(df, threshold=0.6):
    c = correlation(df)
    pairs = [(a, b, round(c.loc[a, b], 3)) for i, a in enumerate(c.columns) for b in c.columns[i + 1:]
             if abs(c.loc[a, b]) >= threshold]
    return sorted(pairs, key=lambda x: -abs(x[2]))


def _measures(df):
    """Numeric columns that look like real measures (skip ID-like columns)."""
    return [c for c in numeric_cols(df) if not (str(c).lower().endswith("id") or str(c).lower().startswith("id")
            or (df[c].nunique() == len(df) and pd.api.types.is_integer_dtype(df[c]) or df[c].nunique() == len(df) and (df[c].dropna() % 1 == 0).all()))]


def generate_insights(df):
    """Plain-English findings derived from whatever columns exist."""
    n = len(df)
    mp = df.isna().sum().sum() / max(df.size, 1) * 100
    out = [f"The dataset has {n:,} rows and {df.shape[1]} columns with {mp:.2f}% missing values overall."]
    worst = df.isna().mean().sort_values(ascending=False)
    if worst.iloc[0] > 0:
        out.append(f"Column '{worst.index[0]}' has the most missing data ({worst.iloc[0] * 100:.1f}%).")
    cats, nums = categorical_cols(df), _measures(df)
    for c in cats[:3]:
        vc = df[c].value_counts()
        if len(vc):
            out.append(f"Most frequent '{c}': '{vc.index[0]}' ({vc.iloc[0] / vc.sum() * 100:.1f}% of non-null records).")
    if cats and nums:
        c, v = cats[0], nums[0]
        g = df.groupby(c, observed=True)[v].sum().sort_values(ascending=False)
        if len(g) > 1:
            out.append(f"Highest-performing '{c}' by total '{v}': '{g.index[0]}' ({g.iloc[0]:,.2f}); lowest: '{g.index[-1]}' ({g.iloc[-1]:,.2f}).")
    for a, b, r in strong_correlations(df)[:3]:
        out.append(f"{'Strong positive' if r > 0 else 'Strong negative'} correlation between '{a}' and '{b}' (r = {r}).")
    o = outlier_report(df)
    for _, r in o[o["Outliers"] > 0].sort_values("Outliers", ascending=False).head(3).iterrows():
        out.append(f"'{r['Column']}' has {int(r['Outliers'])} outliers ({r['Outlier %']}%) outside [{r['Lower Bound']}, {r['Upper Bound']}].")
    for c in nums:
        sk = df[c].skew()
        if abs(sk) > 1:
            out.append(f"'{c}' is {'right' if sk > 0 else 'left'}-skewed (skew {sk:.2f}); median {df[c].median():.2f} vs mean {df[c].mean():.2f}.")
            break
    for dc in datetime_cols(df)[:1]:
        for v in nums[:1]:
            t = df[[dc, v]].dropna().sort_values(dc)
            if len(t) > 3:
                slope = np.polyfit(np.arange(len(t)), t[v].values, 1)[0]
                out.append(f"'{v}' shows an overall {'upward' if slope > 0 else 'downward'} trend over '{dc}'.")
    if df.duplicated().sum():
        out.append(f"{int(df.duplicated().sum())} duplicate rows are present.")
    return out
