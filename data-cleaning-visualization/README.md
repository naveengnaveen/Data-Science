# Data Cleaning & Visualization Dashboard

Upload a CSV/Excel file → find quality issues → clean → explore → visualize → export a PDF report.
Built with Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly and Streamlit.

## Setup (VS Code)
1. Install Python 3.10+ and open this folder in VS Code.
2. Terminal → create a virtual environment:
   - Windows: `python -m venv venv && venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `streamlit run app.py` (opens http://localhost:8501)
5. Click **Upload Dataset → Load bundled sample dataset** to try it, or upload your own file.

## Pages
Overview · Upload Dataset · Data Quality Report · Data Cleaning (preview before apply) · EDA · Visualizations · Insights · Download Reports

## Structure
- `app.py` – Streamlit UI (blue/white theme, sidebar navigation)
- `modules/data_loader.py` – CSV/Excel loading + validation
- `modules/data_quality.py` – missing, duplicates, IQR outliers, dtype & text-consistency checks
- `modules/data_cleaner.py` – cleaning operations + pipeline
- `modules/eda.py` – statistics, correlations, auto insights
- `modules/visualization.py` – Plotly (interactive) + Matplotlib/Seaborn (PNG)
- `modules/report_generator.py` – PDF, Excel/CSV exports
- `dataset/sample_dataset.csv` – sample with deliberate issues (`make_sample.py` regenerates it)
- `test_modules.py` – smoke test of all non-UI features: `python test_modules.py`

## Notes
No column names are hardcoded; everything is detected from the uploaded data.
