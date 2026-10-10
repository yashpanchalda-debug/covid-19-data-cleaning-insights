"""
covid_pipeline.py
Main data cleaning and analysis pipeline for the COVID-19 dataset.
Steps:
  1. Load raw data
  2. Clean and validate
  3. Compute trend metrics
  4. Visualize
  5. Save outputs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_FILE = "covid_data_raw.csv"
CLEAN_FILE = "covid_data_clean.csv"
CHART_FILE = "output_charts.png"

COUNTRY_MAP = {
    "United States": "USA",
    "usa": "USA",
    "INDIA": "India",
    "brazil": "Brazil",
    "United Kingdom": "UK"
}

COUNTRIES = ["USA", "India", "Brazil", "UK", "Germany"]

# ── Step 1: Load ───────────────────────────────────────────────────────────────

def load_data(path):
    df = pd.read_csv(path)
    print(f"[Load]  Raw shape: {df.shape}")
    return df

# ── Step 2: Clean ──────────────────────────────────────────────────────────────

def clean_data(df):
    original_len = len(df)

    # 2a. Drop junk columns
    junk_cols = [c for c in df.columns if c in ("notes",)]
    df = df.drop(columns=junk_cols)
    print(f"[Clean] Dropped junk columns: {junk_cols}")

    # 2b. Standardize country names
    df["country"] = df["country"].replace(COUNTRY_MAP)
    df = df[df["country"].isin(COUNTRIES)]
    print(f"[Clean] Standardized country names. Kept {df['country'].nunique()} countries.")

    # 2c. Fix date dtype
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        print(f"[Clean] Dropped {bad_dates} rows with unparseable dates.")
    df = df.dropna(subset=["date"])

    # 2d. Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["date", "country"])
    print(f"[Clean] Removed {before - len(df)} duplicate rows.")

    # 2e. Fix numeric types
    for col in ["new_cases", "new_deaths", "total_vaccinations"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2f. Remove outliers using IQR per country per column
    def remove_outliers(group, col):
        Q1 = group[col].quantile(0.25)
        Q3 = group[col].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 4 * IQR  # generous fence to catch only extreme data-entry errors
        return group[group[col].isna() | (group[col] <= upper)]

    before = len(df)
    cleaned_groups = []
    for col in ["new_cases", "new_deaths"]:
        parts = []
        for _, grp in df.groupby("country"):
            parts.append(remove_outliers(grp, col))
        df = pd.concat(parts).drop_duplicates(subset=["date", "country"])
    print(f"[Clean] Removed {before - len(df)} outlier rows.")

    # 2g. Fill remaining missing values with forward-fill then median
    df = df.sort_values(["country", "date"])
    for col in ["new_cases", "new_deaths", "total_vaccinations"]:
        df[col] = df.groupby("country")[col].transform(
            lambda s: s.fillna(method="ffill").fillna(s.median())
        )

    # 2h. Clip negatives to zero
    for col in ["new_cases", "new_deaths"]:
        df[col] = df[col].clip(lower=0)

    # 2i. Cast to int
    df["new_cases"] = df["new_cases"].round().astype(int)
    df["new_deaths"] = df["new_deaths"].round().astype(int)
    df["total_vaccinations"] = df["total_vaccinations"].round().astype(int)

    print(f"[Clean] Final shape: {df.shape}  (started with {original_len} rows)")
    return df.reset_index(drop=True)

# ── Step 3: Feature Engineering ────────────────────────────────────────────────

def compute_metrics(df):
    df = df.sort_values(["country", "date"]).reset_index(drop=True)

    # 7-day rolling average of new cases
    df["cases_7day_avg"] = (
        df.groupby("country")["new_cases"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
        .round(1)
    )

    # 7-day rolling average of new deaths
    df["deaths_7day_avg"] = (
        df.groupby("country")["new_deaths"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
        .round(1)
    )

    # Case Fatality Rate (CFR) = deaths / cases * 100
    df["cfr_pct"] = (
        (df["new_deaths"] / df["new_cases"].replace(0, np.nan)) * 100
    ).round(2)

    print("[Metrics] Computed: 7-day rolling averages, CFR.")
    return df

# ── Step 4: Summarize ──────────────────────────────────────────────────────────

def print_summary(df):
    print("\n" + "=" * 60)
    print("  SUMMARY: Key Findings")
    print("=" * 60)

    summary = (
        df.groupby("country")
        .agg(
            total_cases=("new_cases", "sum"),
            total_deaths=("new_deaths", "sum"),
            peak_daily_cases=("new_cases", "max"),
            avg_cfr=("cfr_pct", "mean"),
            days_of_data=("date", "count")
        )
        .round(2)
        .sort_values("total_cases", ascending=False)
    )
    print(summary.to_string())

    print("\nTop Observations:")
    top_cases = summary["total_cases"].idxmax()
    top_cfr = summary["avg_cfr"].idxmax()
    print(f"  • Highest total reported cases:  {top_cases} ({summary.loc[top_cases, 'total_cases']:,.0f})")
    print(f"  • Highest average CFR:           {top_cfr} ({summary.loc[top_cfr, 'avg_cfr']:.2f}%)")
    print("=" * 60 + "\n")

# ── Step 5: Visualize ──────────────────────────────────────────────────────────

COLORS = {
    "USA": "#1f77b4",
    "India": "#ff7f0e",
    "Brazil": "#d62728",
    "UK": "#2ca02c",
    "Germany": "#9467bd"
}

def plot_results(df):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("COVID-19 Trend Analysis (Jan–Jun 2021)", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, :])   # top row, full width
    ax2 = fig.add_subplot(gs[1, 0])   # bottom left
    ax3 = fig.add_subplot(gs[1, 1])   # bottom right

    # ── Chart 1: 7-day rolling average new cases ───────────────────────────
    for country in COUNTRIES:
        sub = df[df["country"] == country]