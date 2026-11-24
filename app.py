# ============================================================
# LLM-Powered Automated EDA Agent
# Author: Ayush Kumar
# ============================================================

import os
import re
import time
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

# Use fallback placeholder if key is not yet set in environment to allow UI to launch
client = OpenAI(
    base_url = NVIDIA_BASE_URL,
    api_key  = NVIDIA_API_KEY or "not_set"
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


def detect_multicollinearity_risks(df: pd.DataFrame, threshold: float = 0.80) -> list:
    """
    Detects pairs of numeric columns with high correlation (|r| >= threshold).
    High multicollinearity causes instability in linear models and distorts feature importance.
    Returns a list of dicts: [{'col_a': c1, 'col_b': c2, 'r': val, 'level': 'High' | 'Severe'}]
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].corr()
    risks = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            val = corr.loc[c1, c2]
            if pd.notna(val) and abs(val) >= threshold:
                level = "Severe" if abs(val) >= 0.90 else "High"
                risks.append({
                    "col_a": c1,
                    "col_b": c2,
                    "r": round(float(val), 3),
                    "level": level
                })
    risks.sort(key=lambda x: abs(x["r"]), reverse=True)
    return risks


def compute_iqr_outliers(df: pd.DataFrame) -> dict:
    """
    Computes statistical outliers for numeric columns using Tukey's IQR rule:
    values < Q1 - 1.5 * IQR or > Q3 + 1.5 * IQR.
    Returns a dict mapping column names to outlier count, percentage, IQR, and bounds.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_info = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        if iqr <= 0:
            continue
        lower_bound = round(q25 - 1.5 * iqr, 2)
        upper_bound = round(q75 + 1.5 * iqr, 2)
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        count = len(outliers)
        if count > 0:
            pct = round(count / len(series) * 100, 2)
            outlier_info[col] = {
                "count": count,
                "pct": pct,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "iqr": round(iqr, 2)
            }
    return outlier_info


def recommend_imputation_strategies(df: pd.DataFrame, outlier_dict: dict = None) -> dict:
    """
    Analyzes columns with missing values and recommends statistical treatment:
    - If >= 50% missing: Recommend dropping column or adding a missingness indicator.
    - If numeric with outliers: Recommend Median imputation (robust to extreme values).
    - If numeric without outliers: Recommend Mean imputation (symmetric distribution).
    - If categorical/boolean: Recommend Mode imputation or 'Unknown' category.
    """
    if outlier_dict is None:
        outlier_dict = compute_iqr_outliers(df)

    recommendations = {}
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)

    for col in df.columns:
        if null_counts[col] > 0:
            pct = null_pct[col]
            count = int(null_counts[col])
            is_num = pd.api.types.is_numeric_dtype(df[col])

            if pct >= 50.0:
                strategy = "Drop Column (High missingness >= 50%) or flag as Unknown"
            elif is_num:
                if col in outlier_dict and outlier_dict[col]["count"] > 0:
                    strategy = "Impute with Median (Outliers detected; robust to skew)"
                else:
                    strategy = "Impute with Mean (Numeric distribution appears symmetric)"
            else:
                strategy = "Impute with Mode (Categorical; most frequent value or 'Unknown')"

            recommendations[col] = {
                "count": count,
                "pct": pct,
                "strategy": strategy
            }
    return recommendations


def profile_dataframe(df: pd.DataFrame) -> str:
    """
    Extracts a full statistical profile from any DataFrame.
    Returns a clean text summary to send to the LLM.
    """
    profile = []

    # Statistical Outliers (Tukey's IQR Method) computed early for reuse
    outliers = compute_iqr_outliers(df)

    # Basic shape
    profile.append("=== DATASET OVERVIEW ===")
    profile.append(f"Total Rows    : {df.shape[0]}")
    profile.append(f"Total Columns : {df.shape[1]}")
    profile.append(f"Column Names  : {list(df.columns)}")

    # Data types
    profile.append("\n=== COLUMN DATA TYPES ===")
    for col, dtype in df.dtypes.items():
        profile.append(f"  {col:30s} -> {str(dtype)}")

    # Missing values & Treatment Recommendations
    profile.append("\n=== MISSING VALUES & IMPUTATION STRATEGY ===")
    imputation_recs = recommend_imputation_strategies(df, outliers)
    if imputation_recs:
        for col, info in imputation_recs.items():
            profile.append(
                f"  {col:25s} -> {info['count']} missing ({info['pct']}%) | Recommendation: {info['strategy']}"
            )
    else:
        profile.append("  No missing values found! (Dataset is complete)")

    # Numeric stats
    profile.append("\n=== NUMERIC COLUMN STATISTICS ===")
    stats = compute_numeric_stats(df)
    if stats is not None:
        profile.append(stats.to_string())
    else:
        profile.append("  No numeric columns found.")

    # Correlations
    profile.append("\n=== TOP CORRELATIONS ===")
    top_corr = compute_top_correlations(df)
    if top_corr:
        for c1, c2, val in top_corr:
            flag = " ⚠️ [HIGH CORRELATION]" if abs(val) >= 0.80 else ""
            profile.append(f"  {c1} <-> {c2} : {val}{flag}")
    else:
        profile.append("  Not enough numeric columns to compute correlations.")

    # Multicollinearity Risk Analysis
    profile.append("\n=== MULTICOLLINEARITY DETECTION (|r| >= 0.80) ===")
    multicollinear_pairs = detect_multicollinearity_risks(df, threshold=0.80)
    if multicollinear_pairs:
        for item in multicollinear_pairs:
            profile.append(
                f"  ⚠️ {item['level']} Risk: {item['col_a']} <-> {item['col_b']} (r = {item['r']}) "
                f"| Feature redundancy detected; consider dropping one or using PCA."
            )
    else:
        profile.append("  ✅ No severe multicollinearity detected (|r| < 0.80 for all numeric pairs).")

    # Statistical Outliers (Tukey's IQR Method)
    profile.append("\n=== STATISTICAL OUTLIERS (IQR METHOD) ===")
    if outliers:
        for col, info in outliers.items():
            profile.append(
                f"  {col:30s} -> {info['count']} outliers ({info['pct']}%) | "
                f"IQR: {info['iqr']} | Normal Range: [{info['lower_bound']} to {info['upper_bound']}]"
            )
    else:
        profile.append("  No statistical outliers detected (or zero variance).")

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
                          dataset_name: str = "Dataset",
                          analysis_goal: str = "",
                          target_col: str = "") -> str:
    """
    Sends data profile to NVIDIA LLM.
    Returns detailed EDA insights in plain English.
    """
    goal_instruction = ""
    if analysis_goal and analysis_goal.strip():
        goal_instruction = f"""
USER-SPECIFIED ANALYSIS GOAL:
"{analysis_goal.strip()}"
Please prioritize and focus your analysis, patterns, and recommendations specifically around this goal while still covering the required sections below.
"""

    target_instruction = ""
    if target_col and target_col.strip():
        target_instruction = f"""
PRIMARY TARGET VARIABLE:
"{target_col.strip()}"
Pay special attention to relationships, drivers, correlations, and predictive patterns regarding this target variable.
"""

    prompt = f"""
You are an expert Data Scientist performing Exploratory Data Analysis.

Below is a complete statistical profile of the '{dataset_name}' dataset.
{target_instruction}
{goal_instruction}
Analyze it carefully and provide insights in this structure:

{data_profile}

1. DATASET SUMMARY
   - What kind of data is this?
   - What is the likely objective/use case?

2. DATA QUALITY ISSUES
   - Which columns have missing values and how serious?
   - Evaluate the suggested imputation strategies (Mean vs. Median vs. Mode vs. Drop).
   - Any columns that should be dropped?
   - Any data type issues?

3. KEY PATTERNS & INSIGHTS
   - Most interesting statistical observations?
   - Which numeric columns show high variance or skewness?
   - What do categorical distributions tell us?

4. POTENTIAL CORRELATIONS & MULTICOLLINEARITY
   - Which column pairs show the highest correlation?
   - Address any flagged multicollinearity risks (|r| >= 0.80) and suggest remediation (feature pruning, PCA, regularization).
   - What relationships would you prioritize?

5. ANOMALIES & OUTLIERS
   - Refer directly to the computed IQR outlier statistics in the profile. Which columns contain significant outliers?
   - How might these extreme values affect analysis and modeling, and what specific treatment (capping, robust scaling, log-transform) do you recommend?

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
                        dataset_name: str = "Dataset",
                        analysis_goal: str = "",
                        target_col: str = "") -> str:
    """
    Generates 5 testable business hypotheses with
    causal reasoning, confounders, and statistical tests.
    """
    goal_instruction = ""
    if analysis_goal and analysis_goal.strip():
        goal_instruction = f"""
USER-SPECIFIED ANALYSIS GOAL:
"{analysis_goal.strip()}"
Ensure the 5 hypotheses directly investigate, test, and address questions relevant to this goal.
"""

    target_instruction = ""
    if target_col and target_col.strip():
        target_instruction = f"""
PRIMARY TARGET VARIABLE:
"{target_col.strip()}"
Prioritize generating hypotheses that explain or predict variation in this target variable.
"""

    prompt = f"""
You are a senior Data Scientist analyzing the '{dataset_name}' dataset.

DATASET PROFILE:
{data_profile}

INITIAL INSIGHTS:
{llm_insights}
{target_instruction}
{goal_instruction}
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

def sanitize_filename(name: str) -> str:
    """
    Sanitizes column and dataset names for filesystem safety.
    Replaces slashes, colons, spaces, and other invalid path characters with underscores.
    Prevents FileNotFoundError when column names contain characters like '/', '\', or ':'.
    """
    clean_name = re.sub(r'[^\w\-]', '_', str(name).strip())
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    return clean_name or "feature"


def generate_visualizations(df: pd.DataFrame,
                            output_dir: str,
                            target_col: str = None,
                            outlier_info: dict = None) -> list:
    """
    Auto-generates charts based on column types, outliers, and optional target column.
    Includes histograms, outlier boxplots, categorical bar charts, correlation heatmap,
    and target-correlation scatter plot.
    Returns list of saved PNG file paths.
    """
    chart_paths  = []
    numeric_cols = df.select_dtypes(
        include=[np.number]).columns.tolist()
    cat_cols     = df.select_dtypes(
        include=["object", "category"]).columns.tolist()

    # Detect target column — user-selected or auto-detected case-insensitively
    target = None
    if target_col and target_col not in ["None", "None (Auto-detect)", ""] and target_col in df.columns:
        target = target_col
    else:
        _TARGET_NAMES = {"survived", "saleprice", "target"}
        for c in df.columns:
            if c.lower() in _TARGET_NAMES:
                target = c
                break

    # Prioritize target column if numeric so it appears in the distributions
    hist_cols = list(numeric_cols)
    if target and target in hist_cols:
        hist_cols.remove(target)
        hist_cols.insert(0, target)

    # 1. Histograms for numeric columns (max 6)
    for col in hist_cols[:6]:
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
        safe_col = sanitize_filename(col)
        path = os.path.join(output_dir, f"hist_{safe_col}.png")
        fig.write_image(path)
        chart_paths.append(path)

    # 2. Outlier Boxplots (IQR Method)
    # Visualizes empirical Tukey's IQR outliers with quartiles and whiskers
    outliers = outlier_info if outlier_info is not None else compute_iqr_outliers(df)
    sorted_outlier_cols = sorted(
        outliers.keys(),
        key=lambda c: outliers[c]["count"],
        reverse=True
    )
    box_cols = sorted_outlier_cols[:4] if sorted_outlier_cols else numeric_cols[:3]

    for col in box_cols:
        use_color = (
            target and
            col != target and
            df[target].nunique() <= 5
        )
        if col in outliers:
            info = outliers[col]
            box_title = f"Outlier Boxplot: {col} ({info['count']} outliers, {info['pct']}%)"
        else:
            box_title = f"Distribution Boxplot: {col}"

        fig = px.box(
            df,
            y=col,
            x=target if use_color else None,
            color=target if use_color else None,
            points="outliers",
            title=box_title
        )
        fig.update_layout(template="plotly_white",
                          title_font_size=14)
        safe_col = sanitize_filename(col)
        path = os.path.join(output_dir, f"box_{safe_col}.png")
        fig.write_image(path)
        chart_paths.append(path)

    # 3. Bar charts for categorical columns (max 3)
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
            safe_col = sanitize_filename(col)
            path = os.path.join(
                output_dir, f"bar_{safe_col}.png")
            fig.write_image(path)
            chart_paths.append(path)

    # 4. Correlation heatmap
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

    # 5. Scatter plot (most correlated numeric vs target)
    # Requires a numeric target since it's plotted on the y-axis
    # with an OLS trendline.
    non_target_numeric = [c for c in numeric_cols if c != target]
    if target_is_numeric and non_target_numeric:
        corrs = df[non_target_numeric].apply(lambda s: df[target].corr(s)).abs()
        valid_corrs = corrs.dropna()
        x_col = valid_corrs.idxmax() if not valid_corrs.empty else non_target_numeric[0]

        # Safe-guard against NaNs in x_col or target causing OLS fitting failure
        scatter_df = df[[x_col, target]].dropna()
        has_enough_points = (
            len(scatter_df) >= 3 and
            scatter_df[x_col].nunique() > 1 and
            scatter_df[target].nunique() > 1
        )
        use_trendline = "ols" if has_enough_points else None

        fig = px.scatter(
            scatter_df if has_enough_points else df,
            x=x_col, y=target,
            title  = f"{x_col} vs {target}",
            opacity= 0.6,
            trendline=use_trendline
        )
        fig.update_layout(template="plotly_white",
                          title_font_size=14)
        safe_x = sanitize_filename(x_col)
        safe_target = sanitize_filename(target)
        path = os.path.join(
            output_dir, f"scatter_{safe_x}_{safe_target}.png")
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
                        numeric_stats: pd.DataFrame = None,
                        analysis_goal: str = "",
                        target_col: str = "",
                        execution_time: str = "") -> str:
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

    if execution_time and execution_time.strip():
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5,
                 sanitize_for_pdf(f"Analysis Runtime: {execution_time.strip()}"),
                 new_x="LMARGIN", new_y="NEXT", align="C")

    if target_col and target_col.strip():
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 5, sanitize_for_pdf(f"Target Variable: {target_col.strip()}"),
                 new_x="LMARGIN", new_y="NEXT", align="C")

    if analysis_goal and analysis_goal.strip():
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 5, sanitize_for_pdf(f"Analysis Goal: {analysis_goal.strip()}"), align="C")

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
# DATA LOADING & INSTANT PREVIEW HELPERS
# ============================================================

def load_dataset_safely(file_path: str) -> pd.DataFrame:
    """
    Loads dataset from CSV, Excel (.xlsx/.xls), or TSV files.
    Applies fallback encodings to prevent decoding errors.
    """
    ext = os.path.splitext(str(file_path))[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    elif ext in [".tsv", ".tab"]:
        try:
            return pd.read_csv(file_path, sep="\t")
        except UnicodeDecodeError:
            try:
                return pd.read_csv(file_path, sep="\t", encoding="latin1")
            except Exception:
                return pd.read_csv(file_path, sep="\t", encoding="cp1252")
    else:
        try:
            return pd.read_csv(file_path)
        except UnicodeDecodeError:
            try:
                return pd.read_csv(file_path, encoding="latin1")
            except Exception:
                return pd.read_csv(file_path, encoding="cp1252")

# Backward compatibility alias
load_csv_safely = load_dataset_safely


def preview_dataset(dataset_file):
    """
    Instantly extracts shape, missing values, memory usage,
    and head(5) preview as soon as a dataset is uploaded,
    calculates a 0-100% data quality health score,
    generates a column data types breakdown,
    and dynamically populates the target column selector.
    """
    if dataset_file is None:
        return (
            gr.update(visible=False),
            "", "", "", "", "",
            pd.DataFrame(),
            pd.DataFrame(),
            gr.update(choices=["None (Auto-detect)"], value="None (Auto-detect)")
        )
    try:
        df = load_dataset_safely(dataset_file)
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols
        n_nulls = int(df.isnull().sum().sum())
        null_pct = (n_nulls / total_cells * 100) if total_cells > 0 else 0
        n_dups = int(df.duplicated().sum())
        dup_pct = (n_dups / n_rows * 100) if n_rows > 0 else 0

        # Data Quality Score (0-100%)
        quality_score = max(0.0, min(100.0, 100.0 - (null_pct * 1.5) - (dup_pct * 2.0)))
        if quality_score >= 90:
            quality_badge = f"🟢 {quality_score:.1f}% (Excellent)"
        elif quality_score >= 75:
            quality_badge = f"🟡 {quality_score:.1f}% (Good)"
        else:
            quality_badge = f"🔴 {quality_score:.1f}% (Needs Work)"

        mem_bytes = df.memory_usage(deep=True).sum()
        if mem_bytes < 1024 * 1024:
            mem_str = f"{mem_bytes / 1024:.1f} KB"
        else:
            mem_str = f"{mem_bytes / (1024 * 1024):.1f} MB"

        preview_df = df.head(5)

        dtypes_df = pd.DataFrame({
            "Column": list(df.columns),
            "Data Type": [str(t) for t in df.dtypes]
        })

        col_choices = ["None (Auto-detect)"] + list(df.columns)
        default_target = "None (Auto-detect)"
        _TARGET_NAMES = {"survived", "saleprice", "target"}
        for col in df.columns:
            if col.lower() in _TARGET_NAMES:
                default_target = col
                break

        return (
            gr.update(visible=True),
            f"{n_rows:,}",
            f"{n_cols:,}",
            f"{n_nulls:,} ({null_pct:.1f}%)",
            mem_str,
            quality_badge,
            preview_df,
            dtypes_df,
            gr.update(choices=col_choices, value=default_target)
        )
    except Exception as e:
        return (
            gr.update(visible=True),
            "Error", "Error", "Error", str(e), "Error",
            pd.DataFrame(),
            pd.DataFrame(),
            gr.update(choices=["None (Auto-detect)"], value="None (Auto-detect)")
        )


def reset_all_fields():
    """
    Resets all input controls, preview cards, and output panels
    back to their initial state.
    """
    return (
        None,
        "",
        gr.update(choices=["None (Auto-detect)"], value="None (Auto-detect)"),
        gr.update(visible=False),
        "", "", "", "", "",
        pd.DataFrame(),
        pd.DataFrame(),
        "",
        [],
        None,
        None
    )


# ============================================================
# MASTER PIPELINE FUNCTION
# ============================================================

def run_eda_agent(dataset_file,
                  analysis_goal: str = "",
                  target_column: str = "None (Auto-detect)",
                  progress=gr.Progress()):
    """
    Master function called by Gradio UI.
    Runs full EDA pipeline on uploaded dataset (CSV, Excel, TSV), conditioned on an optional user goal
    and an optional target column.
    """
    if dataset_file is None:
        return (
            "⚠️ Please upload a dataset file first (.csv, .xlsx, or .tsv).",
            None, None, None
        )

    current_key = os.environ.get("NVIDIA_API_KEY", "")
    if not current_key or current_key == "not_set":
        return (
            "⚠️ NVIDIA_API_KEY is not set.\n\n"
            "Set it before running the app:\n"
            "  Linux/macOS : export NVIDIA_API_KEY=\"nvapi-xxxx\"\n"
            "  Windows     : $env:NVIDIA_API_KEY=\"nvapi-xxxx\"\n\n"
            "Get a free key at https://build.nvidia.com",
            None, None, None
        )
    client.api_key = current_key

    # Tracks which stage was in progress if something fails, so the
    # error message tells the user where the pipeline broke instead
    # of just showing a bare exception string.
    stage = "loading the dataset"
    pipeline_start = time.perf_counter()
    try:
        progress(0.1, desc="Loading dataset...")
        df = load_dataset_safely(dataset_file)
        # splitext only strips the final extension, unlike a blind
        # .replace(".csv", "") which would mangle a name like
        # "2024.csv.backup.csv" by stripping every occurrence.
        dataset_name = os.path.splitext(
            os.path.basename(dataset_file))[0]

        # Resolve target column
        resolved_target = None
        if target_column and target_column not in ["None", "None (Auto-detect)", ""] and target_column in df.columns:
            resolved_target = target_column
        else:
            _TARGET_NAMES = {"survived", "saleprice", "target"}
            for c in df.columns:
                if c.lower() in _TARGET_NAMES:
                    resolved_target = c
                    break

        # Drop purely uninformative identifier columns (avoid deleting valid features like Name or Cabin)
        drop_cols = [
            c for c in df.columns
            if c.lower() in ["id", "passengerid"] and c != resolved_target
        ]
        df_clean = df.drop(
            columns=drop_cols, errors="ignore")

        stage = "profiling the data"
        progress(0.2, desc="Profiling data...")
        data_profile = profile_dataframe(df_clean)

        stage = "generating AI insights (NVIDIA API call)"
        progress(0.4, desc="Generating AI insights...")
        llm_insights = generate_llm_insights(
            data_profile, dataset_name, analysis_goal, target_col=resolved_target)

        stage = "generating hypotheses (NVIDIA API call)"
        progress(0.6, desc="Generating hypotheses...")
        hypotheses = generate_hypotheses(
            data_profile, llm_insights, dataset_name, analysis_goal, target_col=resolved_target)

        stage = "creating visualizations"
        progress(0.75, desc="Creating visualizations...")
        charts_dir = tempfile.mkdtemp()
        outliers_dict = compute_iqr_outliers(df_clean)
        chart_paths = generate_visualizations(
            df_clean, charts_dir, target_col=resolved_target, outlier_info=outliers_dict)

        stage = "generating the PDF report"
        progress(0.9, desc="Generating PDF report...")
        numeric_stats = compute_numeric_stats(df_clean)
        elapsed = round(time.perf_counter() - pipeline_start, 1)
        execution_time_str = f"{elapsed}s"
        pdf_path = generate_pdf_report(
            dataset_name  = dataset_name,
            data_profile  = data_profile,
            llm_insights  = llm_insights,
            hypotheses    = hypotheses,
            chart_paths   = chart_paths,
            output_dir    = charts_dir,
            numeric_stats = numeric_stats,
            analysis_goal = analysis_goal,
            target_col    = resolved_target or "",
            execution_time = execution_time_str
        )

        # Export preprocessed & cleaned dataset
        cleaned_csv_path = os.path.join(
            charts_dir, f"{dataset_name}_cleaned.csv"
        )
        df_clean.to_csv(cleaned_csv_path, index=False)

        # Compute data quality health rating
        total_cells = df.shape[0] * df.shape[1]
        null_pct = (df.isnull().sum().sum() / total_cells * 100) if total_cells > 0 else 0
        dup_pct = (df.duplicated().sum() / df.shape[0] * 100) if df.shape[0] > 0 else 0
        quality_score = max(0.0, min(100.0, 100.0 - (null_pct * 1.5) - (dup_pct * 2.0)))
        if quality_score >= 90:
            quality_badge = f"🟢 {quality_score:.1f}% (Excellent)"
        elif quality_score >= 75:
            quality_badge = f"🟡 {quality_score:.1f}% (Good)"
        else:
            quality_badge = f"🔴 {quality_score:.1f}% (Needs Work)"

        progress(1.0, desc="Done!")

        # Build output text
        goal_display = f"\n🎯 Goal   : {analysis_goal.strip()}" if analysis_goal and analysis_goal.strip() else ""
        target_display = f"\n🎯 Target : {resolved_target}" if resolved_target else ""
        output_text = f"""✅ Dataset: {dataset_name}
📊 Shape  : {df.shape[0]} rows × {df.shape[1]} columns{target_display}{goal_display}
🛡️ Health : {quality_badge}
⏱️ Runtime: {execution_time_str}
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
        return output_text, chart_paths, pdf_path, cleaned_csv_path

    except Exception as e:
        return (
            f"❌ Error while {stage}: {str(e)}\n\n"
            "If this was an NVIDIA API call, double-check your "
            "NVIDIA_API_KEY is valid and you haven't hit a rate limit.",
            None, None, None
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
### Upload any CSV, Excel (.xlsx/.xls), or TSV → Instant AI insights, charts & PDF report
---
**Powered by:** NVIDIA API (Llama 3.1) + Plotly + Gradio  

    """)

    with gr.Row():
        with gr.Column(scale=1):
            dataset_input = gr.File(
                label      = "📁 Upload Dataset (CSV, Excel, TSV)",
                file_types = [".csv", ".xlsx", ".xls", ".tsv", ".tab"]
            )
            target_dropdown = gr.Dropdown(
                label       = "🎯 Target Column (Optional)",
                choices     = ["None (Auto-detect)"],
                value       = "None (Auto-detect)",
                info        = "Select a column to focus predictions, relationships, and scatter trends.",
                interactive = True
            )
            goal_input  = gr.Textbox(
                label       = "🎯 Analysis Goal (Optional)",
                placeholder = "e.g., Focus on factors driving customer churn, or identify high-risk loan applicants",
                lines       = 2
            )
            with gr.Row():
                analyze_btn = gr.Button(
                    "🔍 Analyze My Dataset",
                    variant = "primary",
                    size    = "lg",
                    scale   = 2
                )
                clear_btn = gr.Button(
                    "🗑️ Reset All",
                    variant = "secondary",
                    size    = "lg",
                    scale   = 1
                )
            gr.Markdown("""
**How it works:**
1. 📁 Upload any CSV, Excel, or TSV file
2. ⚡ Get instant data health & preview
3. 🎯 (Optional) Select a Target Column or add a business goal
4. 🔍 Click Analyze
5. 📊 Get tailored AI insights + charts
6. 📄 Download PDF report & cleaned dataset

**Works with any domain:**
- 🏥 Healthcare data
- 🏦 Banking & Finance
- 🛒 E-Commerce
- 📱 SaaS & Tech
- 🎓 Education
- 🏭 Manufacturing
            """)

        with gr.Column(scale=2, visible=False) as preview_col:
            gr.Markdown("### 📋 Instant Dataset Preview & Health Check")
            with gr.Row():
                stat_rows    = gr.Textbox(label="Total Rows", interactive=False)
                stat_cols    = gr.Textbox(label="Total Columns", interactive=False)
                stat_nulls   = gr.Textbox(label="Missing Cells", interactive=False)
                stat_mem     = gr.Textbox(label="Memory Usage", interactive=False)
                stat_quality = gr.Textbox(label="Data Quality Score", interactive=False)
            preview_df = gr.DataFrame(
                label="Raw Data Preview (First 5 Rows)",
                interactive=False,
                wrap=True
            )
            with gr.Accordion("🔍 Column Data Types", open=True):
                dtypes_df = gr.DataFrame(
                    label="Column Schema & Types",
                    interactive=False,
                    wrap=True,
                    height=200
                )

    dataset_input.change(
        fn      = preview_dataset,
        inputs  = [dataset_input],
        outputs = [preview_col, stat_rows, stat_cols, stat_nulls, stat_mem, stat_quality, preview_df, dtypes_df, target_dropdown]
    )

    dataset_input.clear(
        fn      = preview_dataset,
        inputs  = [dataset_input],
        outputs = [preview_col, stat_rows, stat_cols, stat_nulls, stat_mem, stat_quality, preview_df, dtypes_df, target_dropdown]
    )

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

        with gr.TabItem("📄 Export & Downloads"):
            with gr.Row():
                pdf_out = gr.File(
                    label = "⬇️ Download Full EDA Report (PDF)"
                )
                csv_out = gr.File(
                    label = "⬇️ Download Cleaned Dataset (CSV)"
                )
            gr.Markdown("""
### Available Exports:
- 📄 **PDF Report**: Complete statistical profile, AI-generated insights & hypotheses, and embedded charts.
- 📁 **Cleaned CSV**: Preprocessed dataset with uninformative identifier columns removed, ready for downstream ML modeling.
            """)

    analyze_btn.click(
        fn      = run_eda_agent,
        inputs  = [dataset_input, goal_input, target_dropdown],
        outputs = [insights_out, gallery_out, pdf_out, csv_out]
    )

    clear_btn.click(
        fn      = reset_all_fields,
        inputs  = [],
        outputs = [
            dataset_input,
            goal_input,
            target_dropdown,
            preview_col,
            stat_rows,
            stat_cols,
            stat_nulls,
            stat_mem,
            stat_quality,
            preview_df,
            dtypes_df,
            insights_out,
            gallery_out,
            pdf_out,
            csv_out
        ]
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
