"""PDF analysis report, data-quality report, and export helpers."""
import io
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

from .eda import descriptive_stats, generate_insights, numeric_cols, categorical_cols
from .visualization import static_chart, fig_to_png

BLUE = colors.HexColor("#1f6feb")


def _table(df, max_rows=25, max_cols=8):
    d = df.iloc[:max_rows, :max_cols].copy()
    data = [list(map(str, d.columns))] + [[str(v)[:22] for v in r] for r in d.values.tolist()]
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BLUE), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4ff")])]))
    return t


def generate_pdf_report(df, quality, title="Data Analysis Report"):
    """Build a PDF (bytes) with overview, quality, stats, charts and insights."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    ss = getSampleStyleSheet()
    h = ss["Heading2"]; h.textColor = BLUE
    el = [Paragraph(title, ss["Title"]),
          Paragraph(f"Rows: {len(df):,} &nbsp; Columns: {df.shape[1]} &nbsp; Quality score: {quality['score']}/100", ss["Normal"]),
          Spacer(1, 12), Paragraph("1. Data Quality Summary", h), _table(quality["summary"]), Spacer(1, 10),
          Paragraph("Missing values by column", ss["Heading4"]), _table(quality["missing"], 30)]
    if len(quality["outliers"]):
        el += [Spacer(1, 10), Paragraph("Outliers (IQR)", ss["Heading4"]), _table(quality["outliers"])]
    stats = descriptive_stats(df)
    if not stats.empty:
        el += [Paragraph("2. Descriptive Statistics", h), _table(stats.reset_index().rename(columns={"index": "column"}), 20, 9)]
    el += [PageBreak(), Paragraph("3. Visualizations", h)]
    nums, cats = numeric_cols(df), categorical_cols(df)
    charts = []
    if nums: charts += [("histogram", nums[0], None, None), ("box", nums[0], None, None)]
    if cats: charts += [("bar", cats[0], nums[0] if nums else None, None)]
    if cats and nums: charts += [("pie", cats[0], nums[0], None)]
    if len(nums) > 1: charts += [("scatter", nums[0], nums[1], None), ("heatmap", None, None, None)]
    for kind, x, y, c in charts:
        try:  # one failing chart should not break the whole report
            el += [Image(io.BytesIO(fig_to_png(static_chart(df, kind, x, y, c))), width=13 * cm, height=7.4 * cm), Spacer(1, 6)]
        except Exception:
            continue
    el += [Paragraph("4. Key Insights", h)] + [Paragraph(f"• {i}", ss["Normal"]) for i in generate_insights(df)]
    doc.build(el)
    return buf.getvalue()


def quality_report_excel(quality):
    """All quality tables as sheets in one Excel workbook (bytes)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for k in ("summary", "missing", "outliers", "dtypes", "categories"):
            quality[k].to_excel(w, sheet_name=k.title(), index=False)
    return buf.getvalue()


def df_to_csv(df): return df.to_csv(index=False).encode("utf-8")


def df_to_excel(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()
