"""Data Cleaning & Visualization Dashboard (Streamlit). Run: streamlit run app.py"""
import os
import streamlit as st
import pandas as pd

from modules.data_loader import load_dataset, dataset_overview
from modules.data_quality import quality_report
from modules.data_cleaner import apply_pipeline
from modules.eda import (descriptive_stats, correlation, category_summary, numeric_cols,
                         categorical_cols, generate_insights)
from modules.visualization import plotly_chart, static_chart, fig_to_png
from modules.report_generator import (generate_pdf_report, quality_report_excel, df_to_csv, df_to_excel)

st.set_page_config(page_title="Data Cleaning Dashboard", page_icon="📊", layout="wide")

# ---------- Blue & white light theme ----------
st.markdown("""
<style>
.stApp {background:#f5f8ff;}
section[data-testid="stSidebar"] {background:#fff; border-right:1px solid #dbe6ff;}
h1,h2,h3 {color:#0b3d91; font-family:'Segoe UI',sans-serif;}
.card {background:#fff; border-radius:12px; padding:16px 20px; border-left:5px solid #1f6feb;
       box-shadow:0 2px 8px rgba(31,111,235,.12); margin-bottom:8px;}
.card .v {font-size:1.7rem; font-weight:700; color:#1f6feb;} .card .l {color:#5b6b8c; font-size:.85rem;}
.stButton>button, .stDownloadButton>button {background:#1f6feb; color:#fff; border-radius:8px; border:0;}
</style>""", unsafe_allow_html=True)


def card(col, label, value):
    col.markdown(f'<div class="card"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)


# ---------- Session state ----------
for k in ("raw", "clean", "filename", "log"):
    st.session_state.setdefault(k, None)


def current():
    """Cleaned data if available, else raw."""
    return st.session_state.clean if st.session_state.clean is not None else st.session_state.raw


PAGES = ["Overview", "Upload Dataset", "Data Quality Report", "Data Cleaning",
         "Exploratory Data Analysis", "Visualizations", "Insights", "Download Reports"]
st.sidebar.title("📊 Data Dashboard")
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
if st.session_state.filename:
    st.sidebar.success(f"Loaded: {st.session_state.filename}")
    st.sidebar.caption("Using cleaned data" if st.session_state.clean is not None else "Using raw data")

st.title(page)

# Every page except Overview/Upload needs data
if page not in ("Overview", "Upload Dataset") and st.session_state.raw is None:
    st.info("Please upload a dataset first (sidebar → Upload Dataset).")
    st.stop()

# ================= Overview =================
if page == "Overview":
    if st.session_state.raw is None:
        st.write("Upload a CSV or Excel file to clean it, explore it, visualize it and export a PDF report.")
        st.markdown("**Workflow:** Upload → Quality check → Clean → EDA → Visualize → Insights → Export")
    else:
        df, q = current(), quality_report(current())
        c = st.columns(5)
        card(c[0], "Rows", f"{len(df):,}"); card(c[1], "Columns", df.shape[1])
        card(c[2], "Missing cells", f"{int(df.isna().sum().sum()):,}")
        card(c[3], "Duplicates", q["duplicates"]); card(c[4], "Quality score", f"{q['score']}/100")
        st.subheader("Preview"); st.dataframe(df.head(10), use_container_width=True)

# ================= Upload =================
elif page == "Upload Dataset":
    f = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if st.button("Load bundled sample dataset"):
        path = os.path.join("dataset", "sample_dataset.csv")
        st.session_state.update(raw=load_dataset(path), clean=None, filename="sample_dataset.csv", log=None)
    if f is not None and f.name != st.session_state.filename:
        try:
            st.session_state.update(raw=load_dataset(f, f.name), clean=None, filename=f.name, log=None)
        except ValueError as e:
            st.error(str(e))
    if st.session_state.raw is not None:
        o = dataset_overview(st.session_state.raw)
        c = st.columns(2); card(c[0], "Total rows", f"{o['rows']:,}"); card(c[1], "Total columns", o["cols"])
        st.subheader("Columns & data types"); st.dataframe(o["info"], use_container_width=True)
        st.subheader("First 5 records"); st.dataframe(o["head"], use_container_width=True)
        st.subheader("Last 5 records"); st.dataframe(o["tail"], use_container_width=True)

# ================= Quality =================
elif page == "Data Quality Report":
    q = quality_report(st.session_state.raw if st.session_state.clean is None else st.session_state.clean)
    st.metric("Quality score", f"{q['score']} / 100")
    st.subheader("Summary"); st.dataframe(q["summary"], use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    c1.subheader("Missing values"); c1.dataframe(q["missing"], use_container_width=True, hide_index=True)
    c2.subheader("Outliers (IQR)"); c2.dataframe(q["outliers"], use_container_width=True, hide_index=True)
    st.subheader("Columns that look like the wrong data type"); st.dataframe(q["dtypes"], use_container_width=True, hide_index=True)
    st.subheader("Inconsistent categorical values"); st.dataframe(q["categories"], use_container_width=True, hide_index=True)

# ================= Cleaning =================
elif page == "Data Cleaning":
    raw = st.session_state.raw
    q = quality_report(raw)
    with st.form("clean"):
        c1, c2 = st.columns(2)
        dups = c1.checkbox("Remove duplicate records", True)
        empty = c1.checkbox("Remove empty rows", True)
        std = c1.checkbox("Standardize column names (snake_case)", False)
        text = c1.checkbox("Standardize text values (trim + case)", True)
        case = c1.selectbox("Text case", ["title", "lower", "upper"])
        miss = c2.selectbox("Missing values", ["(leave as is)", "mean", "median", "mode", "remove"], index=2)
        out = c2.selectbox("Outliers (IQR)", ["(leave as is)", "cap", "remove"])
        drop = c2.multiselect("Drop columns", list(raw.columns))
        sug = {r.Column: r.Suggested for r in q["dtypes"].itertuples()}
        dt = c2.multiselect("Convert detected columns to suggested type", list(sug), default=list(sug),
                            format_func=lambda c: f"{c} → {sug[c]}")
        go = st.form_submit_button("Preview changes")
    if go:
        opts = dict(dups=dups, empty_rows=empty, std_cols=std, text=text, case=case, drop_cols=drop,
                    missing=None if miss.startswith("(") else miss, outliers=None if out.startswith("(") else out,
                    dtypes={c: sug[c] for c in dt})
        st.session_state.preview = apply_pipeline(raw, opts)
    if st.session_state.get("preview"):
        prev, log = st.session_state.preview
        st.subheader("Preview")
        c = st.columns(3); card(c[0], "Rows before → after", f"{len(raw)} → {len(prev)}")
        card(c[1], "Missing before → after", f"{int(raw.isna().sum().sum())} → {int(prev.isna().sum().sum())}")
        card(c[2], "Duplicates after", int(prev.duplicated().sum()))
        st.code("\n".join(log) or "No steps selected"); st.dataframe(prev.head(20), use_container_width=True)
        a, b = st.columns(2)
        if a.button("✅ Apply changes"):
            st.session_state.clean, st.session_state.log = prev, log
            st.session_state.preview = None; st.success("Applied. All pages now use the cleaned data."); st.rerun()
        if b.button("↩ Reset to raw data"):
            st.session_state.clean = None; st.session_state.preview = None; st.rerun()

# ================= EDA =================
elif page == "Exploratory Data Analysis":
    df = current()
    st.subheader("Descriptive statistics (mean, median, mode, min, max, std)")
    st.dataframe(descriptive_stats(df), use_container_width=True)
    corr = correlation(df)
    if not corr.empty:
        st.subheader("Correlation matrix"); st.dataframe(corr.style.background_gradient(cmap="Blues").format("{:.2f}"), use_container_width=True)
    nums, cats = numeric_cols(df), categorical_cols(df)
    if nums:
        st.subheader("Distribution analysis")
        col = st.selectbox("Numeric column", nums)
        st.plotly_chart(plotly_chart(df, "histogram", x=col), use_container_width=True)
        st.caption(f"Skewness: {df[col].skew():.2f} · Kurtosis: {df[col].kurt():.2f}")
    if cats:
        st.subheader("Category-wise analysis")
        cat = st.selectbox("Category column", cats)
        num = st.selectbox("Numeric column to aggregate", ["(count only)"] + nums)
        st.dataframe(category_summary(df, cat, None if num.startswith("(") else num), use_container_width=True, hide_index=True)

# ================= Visualizations =================
elif page == "Visualizations":
    df = current(); nums, cats = numeric_cols(df), categorical_cols(df)
    all_cols = list(df.columns)
    kind = st.selectbox("Chart type", ["bar", "line", "pie", "histogram", "box", "scatter", "heatmap"])
    x = y = color = None
    if kind in ("bar", "pie"):
        x = st.selectbox("Category column", cats or all_cols)
        y = st.selectbox("Value column (optional)", [None] + nums)
    elif kind == "line":
        x = st.selectbox("X axis", all_cols); y = st.selectbox("Y axis", nums)
    elif kind == "histogram":
        x = st.selectbox("Numeric column", nums)
    elif kind == "box":
        x = st.selectbox("Numeric column", nums); color = st.selectbox("Group by (optional)", [None] + cats)
    elif kind == "scatter":
        x = st.selectbox("X", nums); y = st.selectbox("Y", nums, index=min(1, len(nums) - 1)); color = st.selectbox("Color by (optional)", [None] + cats)
    try:
        st.subheader("Interactive (Plotly)"); st.plotly_chart(plotly_chart(df, kind, x, y, color), use_container_width=True)
        st.subheader("Static (Matplotlib / Seaborn)")
        png = fig_to_png(static_chart(df, kind, x, y, color)); st.image(png)
        st.download_button("⬇ Download chart image (PNG)", png, f"{kind}.png", "image/png")
    except Exception as e:
        st.error(f"Could not draw this chart with the selected columns: {e}")

# ================= Insights =================
elif page == "Insights":
    for i in generate_insights(current()):
        st.markdown(f'<div class="card" style="font-size:1rem">💡 {i}</div>', unsafe_allow_html=True)

# ================= Downloads =================
elif page == "Download Reports":
    df = current(); q = quality_report(df); base = os.path.splitext(st.session_state.filename)[0]
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Cleaned dataset (CSV)", df_to_csv(df), f"{base}_cleaned.csv", "text/csv")
    c2.download_button("⬇ Cleaned dataset (Excel)", df_to_excel(df), f"{base}_cleaned.xlsx")
    c1.download_button("⬇ Data quality report (Excel)", quality_report_excel(q), f"{base}_quality_report.xlsx")
    if c2.button("Generate PDF analysis report"):
        with st.spinner("Building PDF…"):
            st.session_state.pdf = generate_pdf_report(df, q, f"Analysis Report – {st.session_state.filename}")
    if st.session_state.get("pdf"):
        st.download_button("⬇ Download PDF report", st.session_state.pdf, f"{base}_report.pdf", "application/pdf")
    st.caption("Chart PNGs can be downloaded on the Visualizations page.")
