"""Smoke test of every non-UI feature. Run: python test_modules.py"""
from modules.data_loader import load_dataset, dataset_overview
from modules.data_quality import quality_report
from modules.data_cleaner import apply_pipeline
from modules.eda import descriptive_stats, generate_insights, category_summary, correlation
from modules.visualization import static_chart, fig_to_png
from modules.report_generator import generate_pdf_report, quality_report_excel, df_to_csv, df_to_excel
import io, pandas as pd

df = load_dataset("dataset/sample_dataset.csv")
o = dataset_overview(df); q = quality_report(df)
print("loaded", o["rows"], o["cols"], "| score", q["score"])
print(q["summary"].to_string(index=False)); print(q["dtypes"]); print(q["categories"])
sug = {r.Column: r.Suggested for r in q["dtypes"].itertuples()}
opts = dict(dups=True, empty_rows=True, std_cols=True, text=True, missing="median", outliers="cap", dtypes=sug)
c, log = apply_pipeline(df, opts); print("\n".join(log))
q2 = quality_report(c); print("after: score", q2["score"], "dups", q2["duplicates"], "missing", int(c.isna().sum().sum()))
print(descriptive_stats(c).head(3)); print(*generate_insights(c), sep="\n")
for k, x, y, col in [("bar","region","sales",None),("line","order_date","sales",None),("pie","category","sales",None),
                     ("histogram","sales",None,None),("box","sales",None,"region"),("scatter","quantity","sales",None),("heatmap",None,None,None)]:
    assert len(fig_to_png(static_chart(c, k, x, y, col))) > 1000, k
pdf = generate_pdf_report(c, q2); assert pdf[:4] == b"%PDF"; open("outputs/sample_report.pdf", "wb").write(pdf)
assert len(df_to_csv(c)) and len(df_to_excel(c)) and len(quality_report_excel(q2))
pd.read_excel(io.BytesIO(df_to_excel(c)))
# robustness: bad inputs
for bad in [("x.txt", io.BytesIO(b"a")), ("e.csv", io.BytesIO(b""))]:
    try: load_dataset(bad[1], bad[0]); print("FAIL")
    except ValueError as e: print("handled:", e)
print("ALL OK, pdf bytes:", len(pdf))
