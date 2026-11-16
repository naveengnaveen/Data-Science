"""Charts. Plotly = interactive (dashboard); Matplotlib/Seaborn = static PNGs (PDF + image download)."""
import io
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from .eda import correlation

BLUE = "#1f6feb"
sns.set_theme(style="whitegrid", palette="Blues_d")


# ---------- Plotly (interactive) ----------
def plotly_chart(df, kind, x=None, y=None, color=None):
    import plotly.express as px  # lazy import
    kw = dict(color_discrete_sequence=px.colors.sequential.Blues_r[:6] if kind == "pie" else [BLUE])
    if kind == "bar":
        d = df.groupby(x, observed=True)[y].sum().reset_index() if y else df[x].value_counts().rename_axis(x).reset_index(name="count")
        fig = px.bar(d, x=x, y=y or "count", **kw)
    elif kind == "line":
        fig = px.line(df.sort_values(x), x=x, y=y, **kw)
    elif kind == "pie":
        d = df.groupby(x, observed=True)[y].sum().reset_index() if y else df[x].value_counts().rename_axis(x).reset_index(name="count")
        fig = px.pie(d, names=x, values=y or "count", **kw)
    elif kind == "histogram":
        fig = px.histogram(df, x=x, nbins=30, **kw)
    elif kind == "box":
        fig = px.box(df, x=color, y=x, **kw)
    elif kind == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color, **({} if color else {"color_discrete_sequence": [BLUE]}))
    elif kind == "heatmap":
        fig = px.imshow(correlation(df), text_auto=".2f", color_continuous_scale="Blues", aspect="auto")
    else:
        raise ValueError(f"Unknown chart: {kind}")
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ---------- Matplotlib / Seaborn (static) ----------
def static_chart(df, kind, x=None, y=None, color=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    if kind == "bar":
        d = df.groupby(x, observed=True)[y].sum() if y else df[x].value_counts()
        d.head(20).plot.bar(ax=ax, color=BLUE)
    elif kind == "line":
        d = df.sort_values(x); ax.plot(d[x], d[y], color=BLUE)
    elif kind == "pie":
        d = (df.groupby(x, observed=True)[y].sum() if y else df[x].value_counts()).head(8)
        ax.pie(d, labels=d.index, autopct="%1.0f%%", colors=sns.color_palette("Blues_r", len(d))); ax.set_ylabel("")
    elif kind == "histogram":
        sns.histplot(df[x].dropna(), kde=True, ax=ax, color=BLUE)
    elif kind == "box":
        sns.boxplot(data=df, x=color, y=x, ax=ax, color=BLUE) if color else sns.boxplot(y=df[x], ax=ax, color=BLUE)
    elif kind == "scatter":
        sns.scatterplot(data=df, x=x, y=y, hue=color, ax=ax)
    elif kind == "heatmap":
        sns.heatmap(correlation(df), annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title(f"{kind.title()}: " + " vs ".join(str(v) for v in (x, y) if v))
    fig.tight_layout()
    return fig


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()
