# ============================================================
# LLM-Powered Automated EDA Agent
# Author: Ayush Kumar
# ============================================================

import os
import warnings
import tempfile
import pandas as pd
import numpy as np
import plotly.express as px
import gradio as gr
from openai import OpenAI
from fpdf import FPDF
from datetime import datetime

# Gradio >=6.0 moved the `theme` kwarg from the Blocks constructor
# to .launch(); it's still accepted in the constructor for backward
# compatibility, but emits a UserWarning. Harmless — silenced so it
# doesn't clutter the console for anyone running this locally.
warnings.filterwarnings(
    "ignore",
    message=".*moved from the Blocks constructor.*"
)

# ── NVIDIA API Setup ─────────────────────────────────────────
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")

client = OpenAI(
    base_url = NVIDIA_BASE_URL,
    api_key  = NVIDIA_API_KEY
)

# ============================================================
# STEP 1 — DATA PROFILER
# ============================================================

def compute_numeric_stats(df: pd.DataFrame):
    """
    Returns a rounded describe() DataFrame for numeric columns,
    or None if the DataFrame has no numeric columns.
    Shared by the text profiler (for the LLM) and the PDF table
    renderer, so both always agree on the same numbers.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return None
    return df[numeric_cols].describe().round(2)


def compute_top_correlations(df: pd.DataFrame, top_n: int = 8):
    """
    Returns the top-N numeric column pairs by absolute Pearson
    correlation, as a list of (col_a, col_b, corr_value) tuples.
    Empty list if fewer than 2 numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].corr()
    pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            val = corr.loc[c1, c2]
            if pd.notna(val):
                pairs.append((c1, c2, round(float(val), 3)))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return pairs[:top_n]


def profile_dataframe(df: pd.DataFrame) -> str:
    """
    Extracts a full statistical profile from any DataFrame.
    Returns a clean text summary to send to the LLM.
    """
    profile = []

    # Basic shape
    profile.append("=== DATASET OVERVIEW ===")
    profile.append(f"Total Rows    : {df.shape[0]}")
    profile.append(f"Total Columns : {df.shape[1]}")
    profile.append(f"Column Names  : {list(df.columns)}")

    # Data types
    profile.append("\n=== COLUMN DATA TYPES ===")
    for col, dtype in df.dtypes.items():
        profile.append(f"  {col:30s} -> {str(dtype)}")

    # Missing values
    profile.append("\n=== MISSING VALUES ===")
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    has_nulls   = False
    for col in df.columns:
        if null_counts[col] > 0:
            profile.append(
                f"  {col:30s} -> {null_counts[col]} missing ({null_pct[col]}%)"
            )
            has_nulls = True
    if not has_nulls:
        profile.append("  No missing values found!")

    # Numeric stats
    profile.append("\n=== NUMERIC COLUMN STATISTICS ===")
    stats = compute_numeric_stats(df)
    if stats is not None:
        profile.append(stats.to_string())
    else:
        profile.append("  No numeric columns found.")

    # Correlations — computed once here and reused by the PDF
    # generator, so the LLM actually sees real correlation values
    # instead of being asked to guess them from raw stats alone.
    profile.append("\n=== TOP CORRELATIONS ===")
    top_corr = compute_top_correlations(df)
    if top_corr:
        for c1, c2, val in top_corr:
            profile.append(f"  {c1} <-> {c2} : {val}")
    else:
        profile.append("  Not enough numeric columns to compute correlations.")

    # Categorical insights
    profile.append("\n=== CATEGORICAL COLUMN INSIGHTS ===")
    cat_cols = df.select_dtypes(
        include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        unique_vals = df[col].nunique()
        top_vals    = df[col].value_counts().head(5).to_dict()
        profile.append(f"\n  Column   : {col}")
        profile.append(f"  Unique   : {unique_vals}")
        profile.append(f"  Top 5    : {top_vals}")

    # Duplicates
    profile.append("\n=== DUPLICATE ROWS ===")
    profile.append(f"  Duplicate Rows: {df.duplicated().sum()}")

    return "\n".join(profile)


# ============================================================
# STEP 2 — LLM INSIGHT GENERATOR
# ============================================================

def generate_llm_insights(data_profile: str,
                          dataset_name: str = "Dataset") -> str:
    """
    Sends data profile to NVIDIA LLM.
    Returns detailed EDA insights in plain English.
    """
    prompt = f"""
You are an expert Data Scientist performing Exploratory Data Analysis.

Below is a complete statistical profile of the '{dataset_name}' dataset.
Analyze it carefully and provide insights in this structure:

{data_profile}

1. DATASET SUMMARY
   - What kind of data is this?
   - What is the likely objective/use case?

2. DATA QUALITY ISSUES
   - Which columns have missing values and how serious?
   - Any columns that should be dropped?
   - Any data type issues?

3. KEY PATTERNS & INSIGHTS
   - Most interesting statistical observations?
   - Which numeric columns show high variance or skewness?
   - What do categorical distributions tell us?

4. POTENTIAL CORRELATIONS TO INVESTIGATE
   - Which column pairs are likely correlated?
   - What relationships would you prioritize?

5. ANOMALIES & OUTLIERS
   - Which columns likely contain outliers?
   - How might these affect analysis?

Be specific. Use actual column names and numbers from the profile.
Write in clear professional English for a business audience.
"""

    response = client.chat.completions.create(
        model    = "meta/llama-3.1-8b-instruct",
        messages = [
            {
                "role": "system",
                "content": "You are an expert data scientist who provides clear, accurate, and actionable EDA insights."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens  = 1500,
        temperature = 0.3
    )
    return response.choices[0].message.content


# ============================================================
# STEP 3 — HYPOTHESIS GENERATOR
# ============================================================

def generate_hypotheses(data_profile: str,
                        llm_insights: str,
                        dataset_name: str = "Dataset") -> str:
    """
    Generates 5 testable business hypotheses with
    causal reasoning, confounders, and statistical tests.
    """
    prompt = f"""
You are a senior Data Scientist analyzing the '{dataset_name}' dataset.

DATASET PROFILE:
{data_profile}

INITIAL INSIGHTS:
{llm_insights}

Generate exactly 5 testable business hypotheses.
For each hypothesis provide:

HYPOTHESIS [N]:
- CLAIM          : Specific, measurable, testable statement
- TYPE           : Direct Effect / Proxy Variable / Confounded Relationship
- REASONING      : Data-driven reasoning using actual stats
- CONFOUNDERS    : Which variables might explain this relationship
- HOW TO TEST    : Exact statistical test + Python library
- EXPECTED RESULT: Quantified expected outcome
- BUSINESS IMPACT: Real-world decision this enables

Focus on CAUSAL reasoning, not just correlation.
Use actual column names and statistics from the profile.
"""

    response = client.chat.completions.create(
        model    = "meta/llama-3.1-8b-instruct",
        messages = [
            {
                "role": "system",
                "content": "You are a senior data scientist who generates precise, testable, and business-relevant hypotheses."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens  = 2000,
        temperature = 0.3
    )
    return response.choices[0].message.content


# ============================================================
# STEP 4 — AUTO VISUALIZATION ENGINE
# ============================================================

def generate_visualizations(df: pd.DataFrame,
                            output_dir: str) -> list:
    """
    Auto-generates charts based on column types.
    Returns list of saved PNG file paths.
    """
    chart_paths  = []
    numeric_cols = df.select_dtypes(
        include=[np.number]).columns.tolist()
    cat_cols     = df.select_dtypes(
        include=["object"]).columns.tolist()

    # Detect target column — matched case-insensitively so a column
    # named "Target", "TARGET", or "SALEPRICE" is still recognized,
    # not just the exact casings that happened to be listed.
    target = None
    _TARGET_NAMES = {"survived", "saleprice", "target"}
    for c in df.columns:
        if c.lower() in _TARGET_NAMES:
            target = c
            break

    # 1. Histograms for numeric columns (max 6)
    for col in numeric_cols[:6]:
        use_color = (
            target and
            col != target and
            df[target].nunique() <= 5
        )
        fig = px.histogram(
            df, x=col,
            color  = target if use_color else None,
            title  = f"Distribution of {col}",
            opacity= 0.75,
            barmode= "overlay"
        )
        fig.update_layout(template="plotly_white",
                          title_font_size=14)
        path = os.path.join(output_dir, f"hist_{col}.png")
        fig.write_image(path)
        chart_paths.append(path)

    # 2. Bar charts for categorical columns (max 3)
    # Only average the target when it's actually numeric — a target
    # column containing strings (e.g. "Yes"/"No") would otherwise
    # blow up .mean() with a TypeError.
    target_is_numeric = (
        target is not None
        and pd.api.types.is_numeric_dtype(df[target])
    )
    for col in cat_cols[:3]:
        if df[col].nunique() <= 15:
            if target_is_numeric:
                data = df.groupby(col)[target]\
                         .mean().reset_index()
                data.columns = [col, f"Avg {target}"]
                fig = px.bar(
                    data, x=col, y=f"Avg {target}",
                    title  = f"Avg {target} by {col}",
                    color  = f"Avg {target}",
                    color_continuous_scale = "RdYlGn"
                )
            else:
                counts = df[col]\
                           .value_counts().reset_index()
                counts.columns = [col, "Count"]
                fig = px.bar(
                    counts, x=col, y="Count",
                    title = f"Distribution of {col}",
                    color = "Count",
                    color_continuous_scale = "Blues"
                )
            fig.update_layout(template="plotly_white",
                              title_font_size=14)
            path = os.path.join(
                output_dir, f"bar_{col}.png")
            fig.write_image(path)
            chart_paths.append(path)

    # 3. Correlation heatmap
    if len(numeric_cols) >= 2:
        top_numeric = numeric_cols[:10]
        corr = df[top_numeric].corr().round(2)
        fig  = px.imshow(
            corr,
            text_auto          = True,
            color_continuous_scale = "RdBu_r",
            title              = "Correlation Heatmap"
        )
        fig.update_layout(template="plotly_white",
                          title_font_size=14)
        path = os.path.join(
            output_dir, "correlation_heatmap.png")
        fig.write_image(path)
        chart_paths.append(path)

    # 4. Scatter plot (top 2 numeric vs target)
    # Requires a numeric target since it's plotted on the y-axis
    # with an OLS trendline.
    non_target_numeric = [c for c in numeric_cols if c != target]
    if target_is_numeric and non_target_numeric:
        x_col = non_target_numeric[0]
        fig = px.scatter(
            df, x=x_col, y=target,
            title  = f"{x_col} vs {target}",
            opacity= 0.6,
            trendline="ols"
        )
        fig.update_layout(template="plotly_white",
                          title_font_size=14)
        path = os.path.join(
            output_dir, f"scatter_{x_col}_{target}.png")
        fig.write_image(path)
        chart_paths.append(path)

    return chart_paths


# ============================================================
# STEP 5 — PDF REPORT GENERATOR
# ============================================================

# fpdf2's core Helvetica font only supports Latin-1. Text coming
# from the profiler or the LLM commonly contains characters
# outside that range (arrows, em/en dashes, smart quotes, bullets,
# emoji-style checkmarks). We proactively map the common ones to
# ASCII equivalents so the PDF stays readable instead of filling
# up with "?" characters; anything left over still falls back to
# a safe latin-1 replace so the PDF never fails to generate.
_PDF_UNICODE_MAP = {
    "\u2192": "->",   # →
    "\u2190": "<-",   # ←
    "\u2013": "-",    # – en dash
    "\u2014": "--",   # — em dash
    "\u2018": "'", "\u2019": "'",   # ' '
    "\u201c": '"', "\u201d": '"',   # " "
    "\u2022": "-",    # • bullet
    "\u2026": "...",  # … ellipsis
    "\u2713": "v",    # ✓
    "\u2705": "[OK]",  # ✅
    "\u274c": "[X]",   # ❌
    "\u26a0": "[!]",   # ⚠
}


def sanitize_for_pdf(text: str) -> str:
    """Makes LLM/profiler text safe to render with a Latin-1 PDF font."""
    for uni_char, ascii_eq in _PDF_UNICODE_MAP.items():
        text = text.replace(uni_char, ascii_eq)
    return text.encode("latin-1", "replace").decode("latin-1")


def render_numeric_stats_table(pdf: "EDAReportPDF", stats_df: pd.DataFrame):
    """
    Renders a describe()-style stats DataFrame as a proper grid
    table (one row per original column) instead of dumping the
    monospace to_string() block into multi_cell, which breaks
    alignment and turns into an unreadable jumble once a dataset
    has more than a handful of numeric columns.
    """
    table_df = stats_df.T.reset_index().rename(columns={"index": "Column"})
    header = [str(c) for c in table_df.columns]
    rows = table_df.astype(str).values.tolist()

    n_stat_cols = len(header) - 1
    stat_col_width = max(14, min(20, 140 // max(n_stat_cols, 1)))
    col_widths = [45] + [stat_col_width] * n_stat_cols

    pdf.set_font("Helvetica", "", 7)
    with pdf.table(col_widths=col_widths, text_align="CENTER",
                   line_height=5) as table:
        header_row = table.row()
        for h in header:
            header_row.cell(sanitize_for_pdf(h))
        for row in rows:
            r = table.row()
            for val in row:
                r.cell(sanitize_for_pdf(val))
    pdf.set_font("Helvetica", "", 9)
    pdf.ln(2)


class EDAReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(20, 20, 20)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10,
                  "  LLM-Powered Automated EDA Report",
                  fill=True,
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(
            0, 10,
            f"Page {self.page_no()} | Generated: "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            align="C"
        )

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(20, 20, 20)
        self.cell(0, 9, f"  {title}",
                  fill=True,
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 5, sanitize_for_pdf(text))
        self.ln(2)


def generate_pdf_report(dataset_name: str,
                        data_profile: str,
                        llm_insights: str,
                        hypotheses: str,
                        chart_paths: list,
                        output_dir: str,
                        numeric_stats: pd.DataFrame = None) -> str:
    """
    Generates complete EDA PDF report.
    Returns path to the saved PDF file.
    """
    pdf = EDAReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Cover
    pdf.set_font("Helvetica", "B", 18)
    pdf.ln(4)
    pdf.cell(0, 12, f"{dataset_name} Dataset",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7,
             "Automated Exploratory Data Analysis Report",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 7,
             f"Generated: {datetime.now().strftime('%B %d, %Y')}",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # Section 1
    pdf.section_title("1. Dataset Profile")

    # The numeric-stats block in `data_profile` is a monospace
    # to_string() table meant for the LLM prompt — rendering it
    # verbatim in the PDF breaks alignment once there are more
    # than a few numeric columns. Split it out and render it as
    # a real grid table instead; everything else stays as text.
    start_marker = "=== NUMERIC COLUMN STATISTICS ==="
    end_marker   = "=== TOP CORRELATIONS ==="
    if start_marker in data_profile and end_marker in data_profile:
        before, rest = data_profile.split(start_marker, 1)
        _, after = rest.split(end_marker, 1)
        pdf.body_text(before)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Numeric Column Statistics",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        if numeric_stats is not None and not numeric_stats.empty:
            render_numeric_stats_table(pdf, numeric_stats)
        else:
            pdf.body_text("  No numeric columns found.")
        pdf.body_text(end_marker + after)
    else:
        pdf.body_text(data_profile)

    # Section 2
    pdf.add_page()
    pdf.section_title("2. AI-Generated EDA Insights")
    pdf.body_text(llm_insights)

    # Section 3
    pdf.add_page()
    pdf.section_title("3. Auto-Generated Visualizations")
    pdf.ln(2)
    for i, path in enumerate(chart_paths):
        if os.path.exists(path):
            name = os.path.basename(path)\
                     .replace(".png", "")\
                     .replace("_", " ").title()
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, f"Chart {i+1}: {name}",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.image(path, x=15, w=175)
            pdf.ln(3)
            if (i + 1) % 2 == 0 and \
               i + 1 < len(chart_paths):
                pdf.add_page()

    # Section 4
    pdf.add_page()
    pdf.section_title("4. AI-Generated Business Hypotheses")
    pdf.body_text(hypotheses)

    out = os.path.join(
        output_dir, f"{dataset_name}_EDA_Report.pdf")
    pdf.output(out)
    return out


# ============================================================
# MASTER PIPELINE FUNCTION
# ============================================================

def run_eda_agent(csv_file, progress=gr.Progress()):
    """
    Master function called by Gradio UI.
    Runs full EDA pipeline on uploaded CSV.
    """
    if csv_file is None:
        return (
            "⚠️ Please upload a CSV file first.",
            None, None
        )

    if not NVIDIA_API_KEY:
        return (
            "⚠️ NVIDIA_API_KEY is not set.\n\n"
            "Set it before running the app:\n"
            "  Linux/macOS : export NVIDIA_API_KEY=\"nvapi-xxxx\"\n"
            "  Windows     : $env:NVIDIA_API_KEY=\"nvapi-xxxx\"\n\n"
            "Get a free key at https://build.nvidia.com",
            None, None
        )

    # Tracks which stage was in progress if something fails, so the
    # error message tells the user where the pipeline broke instead
    # of just showing a bare exception string.
    stage = "loading the dataset"
    try:
        progress(0.1, desc="Loading dataset...")
        df = pd.read_csv(csv_file)
        # splitext only strips the final extension, unlike a blind
        # .replace(".csv", "") which would mangle a name like
        # "2024.csv.backup.csv" by stripping every occurrence.
        dataset_name = os.path.splitext(
            os.path.basename(csv_file))[0]

        # Drop useless ID columns
        drop_cols = [
            c for c in df.columns
            if c.lower() in [
                "id", "passengerid", "name",
                "ticket", "cabin"
            ]
        ]
        df_clean = df.drop(
            columns=drop_cols, errors="ignore")

        stage = "profiling the data"
        progress(0.2, desc="Profiling data...")
        data_profile = profile_dataframe(df_clean)

        stage = "generating AI insights (NVIDIA API call)"
        progress(0.4, desc="Generating AI insights...")
        llm_insights = generate_llm_insights(
            data_profile, dataset_name)

        stage = "generating hypotheses (NVIDIA API call)"
        progress(0.6, desc="Generating hypotheses...")
        hypotheses = generate_hypotheses(
            data_profile, llm_insights, dataset_name)

        stage = "creating visualizations"
        progress(0.75, desc="Creating visualizations...")
        charts_dir = tempfile.mkdtemp()
        chart_paths = generate_visualizations(
            df_clean, charts_dir)

        stage = "generating the PDF report"
        progress(0.9, desc="Generating PDF report...")
        numeric_stats = compute_numeric_stats(df_clean)
        pdf_path = generate_pdf_report(
            dataset_name  = dataset_name,
            data_profile  = data_profile,
            llm_insights  = llm_insights,
            hypotheses    = hypotheses,
            chart_paths   = chart_paths,
            output_dir    = charts_dir,
            numeric_stats = numeric_stats
        )

        progress(1.0, desc="Done!")

        # Build output text
        output_text = f"""✅ Dataset: {dataset_name}
📊 Shape  : {df.shape[0]} rows × {df.shape[1]} columns
⏱️ Status : Analysis Complete!

{'='*60}
📋 DATASET PROFILE
{'='*60}
{data_profile}

{'='*60}
🤖 AI-GENERATED INSIGHTS
{'='*60}
{llm_insights}

{'='*60}
🧠 BUSINESS HYPOTHESES
{'='*60}
{hypotheses}
"""
        return output_text, chart_paths, pdf_path

    except Exception as e:
        return (
            f"❌ Error while {stage}: {str(e)}\n\n"
            "If this was an NVIDIA API call, double-check your "
            "NVIDIA_API_KEY is valid and you haven't hit a rate limit.",
            None, None
        )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title = "LLM-Powered EDA Agent",
    theme = gr.themes.Soft()
) as demo:

    gr.Markdown("""
# 🤖 LLM-Powered Automated EDA Agent
### Upload any CSV → Instant AI insights, charts & PDF report
---
**Powered by:** NVIDIA API (Llama 3.1) + Plotly + Gradio  

    """)

    with gr.Row():
        with gr.Column(scale=1):
            csv_input   = gr.File(
                label      = "📁 Upload Your CSV File",
                file_types = [".csv"]
            )
            analyze_btn = gr.Button(
                "🔍 Analyze My Dataset",
                variant = "primary",
                size    = "lg"
            )
            gr.Markdown("""
**How it works:**
1. 📁 Upload any CSV
2. 🔍 Click Analyze
3. 📊 Get AI insights + charts
4. 📄 Download PDF report

**Works with any domain:**
- 🏥 Healthcare data
- 🏦 Banking & Finance
- 🛒 E-Commerce
- 📱 SaaS & Tech
- 🎓 Education
- 🏭 Manufacturing
            """)

    with gr.Tabs():
        with gr.TabItem("📊 Insights & Profile"):
            insights_out = gr.Textbox(
                label    = "AI Analysis Results",
                lines    = 35,
                max_lines= 60
            )

        with gr.TabItem("📈 Visualizations"):
            gallery_out = gr.Gallery(
                label   = "Auto-Generated Charts",
                columns = 2,
                height  = 700
            )

        with gr.TabItem("📄 Download PDF Report"):
            pdf_out = gr.File(
                label = "⬇️ Download Full EDA Report"
            )
            gr.Markdown("""
### Your PDF Report Includes:
- ✅ Complete Dataset Profile & Statistics
- ✅ AI-Generated EDA Insights (5 sections)
- ✅ All Auto-Generated Visualizations
- ✅ 5 Testable Business Hypotheses
- ✅ Professional formatting with page numbers
            """)

    analyze_btn.click(
        fn      = run_eda_agent,
        inputs  = [csv_input],
        outputs = [insights_out, gallery_out, pdf_out]
    )

    gr.Markdown("""
---
### 💡 Try these datasets:
| Dataset | Link |
|---------|------|
| Titanic | kaggle.com/c/titanic |
| House Prices | kaggle.com/c/house-prices-advanced-regression-techniques |
| Netflix Shows | kaggle.com/datasets/shivamb/netflix-shows |
    """)

if __name__ == "__main__":
    demo.launch()
