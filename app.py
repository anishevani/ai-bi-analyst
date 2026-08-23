"""
AI BI Analyst — ask natural-language questions about tabular sales data.
Works with OpenAI when OPENAI_API_KEY is set; otherwise uses a local analyst.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = APP_DIR / "sample_data" / "sales_data.csv"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

st.set_page_config(
    page_title="AI BI Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');

      :root {
        --ink: #1a2332;
        --muted: #5a6a7a;
        --accent: #0d7a6f;
        --accent-soft: #e6f4f2;
        --surface: #f7f5f1;
        --line: #e2ddd4;
      }

      html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
      }

      .stApp {
        background:
          radial-gradient(1200px 600px at 10% -10%, #d8ebe7 0%, transparent 55%),
          radial-gradient(900px 500px at 100% 0%, #f0e6d8 0%, transparent 50%),
          linear-gradient(180deg, #f7f5f1 0%, #efebe4 100%);
      }

      h1, h2, h3 {
        font-family: 'Fraunces', Georgia, serif !important;
        letter-spacing: -0.02em;
      }

      .brand-mark {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--ink);
        margin: 0 0 0.25rem 0;
        line-height: 1.15;
      }

      .brand-sub {
        color: var(--muted);
        font-size: 1.05rem;
        margin: 0 0 1.5rem 0;
        max-width: 42rem;
      }

      .metric-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.85rem;
        margin: 0.5rem 0 1.25rem 0;
      }

      .metric-tile {
        background: rgba(255,255,255,0.72);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        backdrop-filter: blur(6px);
      }

      .metric-tile .label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
        margin-bottom: 0.35rem;
      }

      .metric-tile .value {
        font-family: 'Fraunces', Georgia, serif;
        font-size: 1.55rem;
        font-weight: 600;
        color: var(--ink);
      }

      .answer-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        margin: 0.75rem 0 1rem 0;
        line-height: 1.55;
      }

      .mode-pill {
        display: inline-block;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.3rem 0.65rem;
        border-radius: 6px;
        margin-bottom: 0.75rem;
      }

      @media (max-width: 900px) {
        .metric-strip { grid-template-columns: repeat(2, 1fr); }
        .brand-mark { font-size: 1.9rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return prepare_dataframe(df)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for col in out.columns:
        lower = col.lower()
        if "date" in lower:
            out[col] = pd.to_datetime(out[col], errors="coerce")
        elif lower in {"revenue", "unit_price", "units_sold", "quantity", "amount", "sales"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def detect_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = {c.lower(): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in cols:
                return cols[n]
        for key, original in cols.items():
            for n in names:
                if n in key:
                    return original
        return None

    return {
        "date": pick("date", "order_date", "sold_at"),
        "revenue": pick("revenue", "sales", "amount", "total"),
        "units": pick("units_sold", "units", "quantity", "qty"),
        "region": pick("region", "territory", "area"),
        "product": pick("product", "item", "sku_name"),
        "category": pick("category", "product_category"),
        "segment": pick("customer_segment", "segment"),
        "rep": pick("sales_rep", "rep", "owner"),
    }


def money(value: float) -> str:
    return f"${value:,.0f}"


def overview_metrics(df: pd.DataFrame, cols: dict[str, str | None]) -> dict[str, str]:
    rev_col = cols["revenue"]
    units_col = cols["units"]
    product_col = cols["product"]
    region_col = cols["region"]

    total_rev = float(df[rev_col].sum()) if rev_col else 0.0
    total_units = int(df[units_col].sum()) if units_col else len(df)
    n_products = int(df[product_col].nunique()) if product_col else int(df.shape[1])
    n_regions = int(df[region_col].nunique()) if region_col else 0

    return {
        "Total revenue": money(total_rev),
        "Units sold": f"{total_units:,}",
        "Products": str(n_products),
        "Regions": str(n_regions) if n_regions else "—",
    }


def dataframe_profile(df: pd.DataFrame, cols: dict[str, str | None]) -> str:
    lines = [
        f"Rows: {len(df):,}",
        f"Columns: {', '.join(df.columns.astype(str))}",
    ]
    rev = cols["revenue"]
    if rev:
        lines.append(f"Total revenue: {money(float(df[rev].sum()))}")
        lines.append(f"Average order revenue: {money(float(df[rev].mean()))}")
    if cols["region"] and rev:
        top = (
            df.groupby(cols["region"], dropna=False)[rev]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        lines.append("Revenue by region: " + ", ".join(f"{k}={money(v)}" for k, v in top.items()))
    if cols["product"] and rev:
        top = (
            df.groupby(cols["product"], dropna=False)[rev]
            .sum()
            .sort_values(ascending=False)
            .head(5)
        )
        lines.append("Top products by revenue: " + ", ".join(f"{k}={money(v)}" for k, v in top.items()))
    if cols["category"] and rev:
        top = (
            df.groupby(cols["category"], dropna=False)[rev]
            .sum()
            .sort_values(ascending=False)
        )
        lines.append("Revenue by category: " + ", ".join(f"{k}={money(v)}" for k, v in top.items()))
    if cols["date"] and rev:
        monthly = (
            df.assign(_m=df[cols["date"]].dt.to_period("M").astype(str))
            .groupby("_m")[rev]
            .sum()
        )
        if len(monthly):
            best = monthly.idxmax()
            lines.append(f"Best month: {best} ({money(float(monthly.max()))})")
            lines.append(f"Latest month: {monthly.index[-1]} ({money(float(monthly.iloc[-1]))})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Local (offline) analyst
# ---------------------------------------------------------------------------
def local_analyze(question: str, df: pd.DataFrame, cols: dict[str, str | None]) -> dict[str, Any]:
    q = question.lower().strip()
    rev = cols["revenue"]
    date_col = cols["date"]
    region = cols["region"]
    product = cols["product"]
    category = cols["category"]
    segment = cols["segment"]
    units = cols["units"]
    rep = cols["rep"]

    chart: dict[str, Any] | None = None
    answer_parts: list[str] = []

    def bar(series: pd.Series, title: str, xlabel: str) -> dict[str, Any]:
        return {
            "type": "bar",
            "title": title,
            "x": series.index.astype(str).tolist(),
            "y": series.values.tolist(),
            "xlabel": xlabel,
            "ylabel": "Revenue" if rev else "Value",
        }

    def line(series: pd.Series, title: str) -> dict[str, Any]:
        return {
            "type": "line",
            "title": title,
            "x": series.index.astype(str).tolist(),
            "y": series.values.tolist(),
            "xlabel": "Period",
            "ylabel": "Revenue" if rev else "Value",
        }

    # Trend / monthly
    if any(w in q for w in ("trend", "over time", "monthly", "month", "time series", "growth")):
        if date_col and rev:
            monthly = (
                df.dropna(subset=[date_col])
                .assign(_p=lambda d: d[date_col].dt.to_period("M").astype(str))
                .groupby("_p")[rev]
                .sum()
            )
            first, last = float(monthly.iloc[0]), float(monthly.iloc[-1])
            change = ((last - first) / first * 100) if first else 0.0
            answer_parts.append(
                f"Revenue moved from {money(first)} in {monthly.index[0]} to "
                f"{money(last)} in {monthly.index[-1]} ({change:+.1f}% overall)."
            )
            answer_parts.append(f"Peak month was {monthly.idxmax()} at {money(float(monthly.max()))}.")
            chart = line(monthly, "Monthly revenue")
        else:
            answer_parts.append("I need date and revenue columns to chart a trend.")

    # Region
    elif any(w in q for w in ("region", "territory", "geography", "where")):
        if region and rev:
            by_region = df.groupby(region)[rev].sum().sort_values(ascending=False)
            leader = by_region.index[0]
            answer_parts.append(
                f"{leader} leads with {money(float(by_region.iloc[0]))} "
                f"({float(by_region.iloc[0]) / float(by_region.sum()) * 100:.1f}% of total)."
            )
            answer_parts.append(
                "Breakdown: " + ", ".join(f"{k} {money(v)}" for k, v in by_region.items()) + "."
            )
            chart = bar(by_region, "Revenue by region", "Region")
        else:
            answer_parts.append("No region/revenue columns found in this dataset.")

    # Product
    elif any(w in q for w in ("product", "sku", "item", "best seller", "top seller")):
        if product and rev:
            by_prod = df.groupby(product)[rev].sum().sort_values(ascending=False)
            top_n = by_prod.head(5)
            answer_parts.append(
                f"Top product is {top_n.index[0]} at {money(float(top_n.iloc[0]))}."
            )
            answer_parts.append(
                "Top 5: " + ", ".join(f"{k} {money(v)}" for k, v in top_n.items()) + "."
            )
            chart = bar(top_n, "Top products by revenue", "Product")
        else:
            answer_parts.append("No product/revenue columns found in this dataset.")

    # Category
    elif "categor" in q:
        if category and rev:
            by_cat = df.groupby(category)[rev].sum().sort_values(ascending=False)
            answer_parts.append(
                "Category mix: " + ", ".join(f"{k} {money(v)}" for k, v in by_cat.items()) + "."
            )
            chart = bar(by_cat, "Revenue by category", "Category")
        else:
            answer_parts.append("No category/revenue columns found.")

    # Segment
    elif any(w in q for w in ("segment", "customer type", "enterprise", "consumer", "smb")):
        if segment and rev:
            by_seg = df.groupby(segment)[rev].sum().sort_values(ascending=False)
            answer_parts.append(
                "Revenue by segment: " + ", ".join(f"{k} {money(v)}" for k, v in by_seg.items()) + "."
            )
            chart = bar(by_seg, "Revenue by customer segment", "Segment")
        else:
            answer_parts.append("No customer segment column found.")

    # Rep / who
    elif any(w in q for w in ("rep", "salesperson", "who sold", "sales rep", "performer")):
        if rep and rev:
            by_rep = df.groupby(rep)[rev].sum().sort_values(ascending=False)
            answer_parts.append(
                f"{by_rep.index[0]} is the top rep with {money(float(by_rep.iloc[0]))}."
            )
            chart = bar(by_rep, "Revenue by sales rep", "Sales rep")
        else:
            answer_parts.append("No sales rep column found.")

    # Units / volume
    elif any(w in q for w in ("unit", "volume", "quantity")):
        if units and product:
            by_prod = df.groupby(product)[units].sum().sort_values(ascending=False).head(8)
            answer_parts.append(
                f"Highest volume: {by_prod.index[0]} with {int(by_prod.iloc[0]):,} units."
            )
            chart = {
                "type": "bar",
                "title": "Units sold by product",
                "x": by_prod.index.astype(str).tolist(),
                "y": by_prod.values.tolist(),
                "xlabel": "Product",
                "ylabel": "Units",
            }
        elif units:
            answer_parts.append(f"Total units sold: {int(df[units].sum()):,}.")
        else:
            answer_parts.append("No units column found.")

    # Summary / default
    else:
        answer_parts.append(dataframe_profile(df, cols).replace("\n", " "))
        if date_col and rev:
            monthly = (
                df.dropna(subset=[date_col])
                .assign(_p=lambda d: d[date_col].dt.to_period("M").astype(str))
                .groupby("_p")[rev]
                .sum()
            )
            chart = line(monthly, "Monthly revenue")
        elif region and rev:
            chart = bar(
                df.groupby(region)[rev].sum().sort_values(ascending=False),
                "Revenue by region",
                "Region",
            )

    # Optional month filter phrases like "in June" / "in 2024-06"
    month_match = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        q,
    )
    if month_match and date_col and rev and "trend" not in q:
        month_name = month_match.group(1).title()
        month_num = pd.to_datetime(month_name, format="%B").month
        subset = df[df[date_col].dt.month == month_num]
        if len(subset):
            answer_parts.append(
                f"In {month_name}: {money(float(subset[rev].sum()))} revenue across {len(subset)} orders."
            )

    return {
        "answer": " ".join(answer_parts) if answer_parts else "I could not derive an answer from this data.",
        "chart": chart,
        "mode": "local",
    }


# ---------------------------------------------------------------------------
# OpenAI analyst
# ---------------------------------------------------------------------------
def openai_analyze(question: str, df: pd.DataFrame, cols: dict[str, str | None]) -> dict[str, Any]:
    from openai import OpenAI

    profile = dataframe_profile(df, cols)
    sample = df.head(12).to_csv(index=False)

    system = (
        "You are a BI analyst. Answer using only the provided dataset profile and sample rows. "
        "Return strict JSON with keys: answer (string), chart (object or null). "
        "chart, when present, must be: "
        '{"type":"bar"|"line","title":str,"x":[str],"y":[number],"xlabel":str,"ylabel":str}. '
        "Prefer clear, concise business language. No markdown."
    )
    user = (
        f"Dataset profile:\n{profile}\n\n"
        f"Sample rows (CSV):\n{sample}\n\n"
        f"Question: {question}"
    )

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return {"answer": content, "chart": None, "mode": "openai"}

    chart = payload.get("chart")
    if chart is not None and not isinstance(chart, dict):
        chart = None
    return {
        "answer": str(payload.get("answer") or "No answer returned."),
        "chart": chart,
        "mode": "openai",
    }


def analyze(question: str, df: pd.DataFrame, cols: dict[str, str | None]) -> dict[str, Any]:
    if OPENAI_API_KEY:
        try:
            return openai_analyze(question, df, cols)
        except Exception as exc:  # noqa: BLE001 — surface to UI, fall back locally
            local = local_analyze(question, df, cols)
            local["answer"] = (
                f"(OpenAI unavailable: {exc}. Showing local analyst result.) {local['answer']}"
            )
            local["mode"] = "local-fallback"
            return local
    return local_analyze(question, df, cols)


def render_chart(chart: dict[str, Any] | None) -> None:
    if not chart:
        return
    chart_type = chart.get("type", "bar")
    x = chart.get("x") or []
    y = chart.get("y") or []
    if not x or not y or len(x) != len(y):
        return
    plot_df = pd.DataFrame({"x": x, "y": y})
    title = chart.get("title") or "Chart"
    xlabel = chart.get("xlabel") or ""
    ylabel = chart.get("ylabel") or ""

    if chart_type == "line":
        fig = px.line(plot_df, x="x", y="y", markers=True, title=title)
    else:
        fig = px.bar(plot_df, x="x", y="y", title=title)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        font_family="DM Sans",
        title_font_family="Fraunces",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=380,
    )
    fig.update_traces(marker_color="#0d7a6f")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.markdown('<p class="brand-mark">AI BI Analyst</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="brand-sub">Ask plain-English questions about your sales data. '
        "Charts and answers update from the active dataset.</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Data")
        source = st.radio(
            "Dataset",
            ["Sample sales data", "Upload CSV"],
            label_visibility="collapsed",
        )
        uploaded = None
        if source == "Upload CSV":
            uploaded = st.file_uploader("CSV file", type=["csv"])

        st.divider()
        st.caption(
            "OpenAI mode is on when `OPENAI_API_KEY` is set in `.env`. "
            "Otherwise the local analyst answers from pandas summaries."
        )
        if OPENAI_API_KEY:
            st.success(f"OpenAI connected · {OPENAI_MODEL}")
        else:
            st.info("Local analyst mode (no API key)")

        st.divider()
        st.markdown("**Try asking**")
        suggestions = [
            "What is the revenue trend over time?",
            "Which region performs best?",
            "What are the top products by revenue?",
            "How does revenue break down by category?",
            "Which sales rep leads?",
            "Show units sold by product",
        ]
        for s in suggestions:
            if st.button(s, use_container_width=True, key=f"sug_{s}"):
                st.session_state["question"] = s
                st.session_state["auto_run"] = True

    # Load data
    error: str | None = None
    df: pd.DataFrame | None = None
    if source == "Sample sales data":
        if SAMPLE_CSV.exists():
            df = load_csv(SAMPLE_CSV)
        else:
            error = f"Sample file missing: {SAMPLE_CSV}"
    elif uploaded is not None:
        try:
            df = prepare_dataframe(pd.read_csv(uploaded))
        except Exception as exc:  # noqa: BLE001
            error = f"Could not read CSV: {exc}"
    else:
        error = "Upload a CSV to get started, or switch back to the sample dataset."

    if error or df is None:
        st.warning(error or "No data loaded.")
        return

    if df.empty:
        st.error("The dataset is empty.")
        return

    cols = detect_columns(df)
    metrics = overview_metrics(df, cols)
    tiles = "".join(
        f'<div class="metric-tile"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in metrics.items()
    )
    st.markdown(f'<div class="metric-strip">{tiles}</div>', unsafe_allow_html=True)

    with st.expander("Preview data", expanded=False):
        st.dataframe(df, use_container_width=True, height=280)

    default_q = st.session_state.get("question", "What is the revenue trend over time?")
    question = st.text_input(
        "Your question",
        value=default_q,
        placeholder="e.g. Which region grew the most?",
    )
    run = st.button("Analyze", type="primary") or st.session_state.pop("auto_run", False)

    if run and question.strip():
        st.session_state["question"] = question.strip()
        with st.spinner("Analyzing…"):
            result = analyze(question.strip(), df, cols)
        mode_label = {
            "openai": "OpenAI",
            "local": "Local analyst",
            "local-fallback": "Local fallback",
        }.get(result["mode"], result["mode"])
        st.markdown(f'<div class="mode-pill">{mode_label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-card">{result["answer"]}</div>', unsafe_allow_html=True)
        render_chart(result.get("chart"))
    elif not question.strip():
        st.caption("Enter a question to generate an insight and chart.")


if __name__ == "__main__":
    main()
