"""Load CSV / Excel files into a DataFrame with basic validation."""
import io
import pandas as pd

SUPPORTED = (".csv", ".xlsx", ".xls")


def load_dataset(file, name=None):
    """Read an uploaded file (path or file-like) into a DataFrame. Raises ValueError on bad input."""
    name = (name or getattr(file, "name", str(file))).lower()
    if not name.endswith(SUPPORTED):
        raise ValueError(f"Unsupported file type. Use one of: {', '.join(SUPPORTED)}")
    try:
        if name.endswith(".csv"):
            raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
            df = None
            for enc in ("utf-8", "utf-8-sig", "latin-1"):  # try common encodings
                try:
                    df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise ValueError("Could not decode the CSV file.")
        else:
            df = pd.read_excel(file)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")
    if df.empty or df.shape[1] == 0:
        raise ValueError("The dataset has no rows or columns.")
    return df


def dataset_overview(df):
    """Rows, columns and a column/dtype table."""
    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null": df.notna().sum().values,
        "Unique": df.nunique().values,
    })
    return {"rows": len(df), "cols": df.shape[1], "info": info,
            "head": df.head(5), "tail": df.tail(5)}
