"""Generates dataset/sample_dataset.csv containing realistic quality issues."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 300
df = pd.DataFrame({
    "Order ID": range(1001, 1001 + n),
    "Order Date": pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
    "Region": rng.choice(["North", "South", "East", "West"], n),
    "Category": rng.choice(["Electronics", "Furniture", "Clothing", "Grocery"], n),
    "Quantity": rng.integers(1, 20, n).astype(float),
    "Unit Price": rng.normal(120, 40, n).round(2),
})
df["Sales"] = (df["Quantity"] * df["Unit Price"]).round(2)
df["Customer Age"] = rng.integers(18, 70, n).astype(float)
df["Notes"] = "ok"
for c, k in [("Quantity", 15), ("Unit Price", 20), ("Customer Age", 25), ("Region", 12)]:
    df.loc[rng.choice(n, k, replace=False), c] = np.nan          # missing values
df.loc[rng.choice(n, 6, replace=False), "Sales"] = rng.integers(9000, 15000, 6)  # outliers
idx = rng.choice(n, 30, replace=False)
df.loc[idx[:10], "Region"] = df.loc[idx[:10], "Region"].str.upper()              # inconsistent text
df.loc[idx[10:20], "Category"] = " " + df.loc[idx[10:20], "Category"].str.lower() + " "
df["Unit Price"] = df["Unit Price"].astype(object)
df.loc[idx[20:25], "Unit Price"] = "$" + df.loc[idx[20:25], "Unit Price"].astype(str)  # wrong dtype
df = pd.concat([df, df.sample(10, random_state=1)], ignore_index=True)           # duplicates
df.loc[len(df)] = [np.nan] * df.shape[1]                                          # empty record
df.to_csv("dataset/sample_dataset.csv", index=False)
print(df.shape)
