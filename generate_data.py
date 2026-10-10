"""
generate_data.py
Generates a realistic but intentionally messy COVID-19 dataset for demonstration purposes.
Introduces: missing values, duplicate rows, inconsistent country names, wrong dtypes, outliers.
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

COUNTRIES = ["USA", "United States", "usa",  # intentional duplicates/variants
             "India", "INDIA",
             "Brazil", "brazil",
             "UK", "United Kingdom",
             "Germany"]

COUNTRY_MAP = {
    "United States": "USA", "usa": "USA",
    "INDIA": "India",
    "brazil": "Brazil",
    "United Kingdom": "UK"
}

CLEAN_COUNTRIES = ["USA", "India", "Brazil", "UK", "Germany"]

# Base daily new case rates per country (rough scale)
BASE_CASES = {
    "USA": 60000,
    "India": 80000,
    "Brazil": 40000,
    "UK": 25000,
    "Germany": 15000
}

BASE_DEATHS = {
    "USA": 900,
    "India": 700,
    "Brazil": 1100,
    "UK": 400,
    "Germany": 200
}

def generate_country_data(country, start="2021-01-01", periods=180):
    dates = pd.date_range(start=start, periods=periods, freq="D")
    base_c = BASE_CASES[country]
    base_d = BASE_DEATHS[country]

    new_cases = (base_c + np.random.normal(0, base_c * 0.15, periods)
                 * np.sin(np.linspace(0, 3 * np.pi, periods)) + base_c * 0.3).clip(0)
    new_deaths = (base_d + np.random.normal(0, base_d * 0.2, periods)).clip(0)

    df = pd.DataFrame({
        "date": dates,
        "country": country,
        "new_cases": new_cases.astype(int),
        "new_deaths": new_deaths.astype(int),
        "total_vaccinations": np.cumsum(np.random.randint(50000, 200000, periods))
    })
    return df

rows = []
for c in CLEAN_COUNTRIES:
    df_c = generate_country_data(c)
    # Introduce messy country name variants for ~30% of rows
    messy_names = {v: k for k, v in COUNTRY_MAP.items()}
    if c in messy_names:
        mask = np.random.rand(len(df_c)) < 0.3
        df_c.loc[mask, "country"] = messy_names[c]
    rows.append(df_c)

df = pd.concat(rows, ignore_index=True)

# 1. Introduce missing values (~5% of new_cases, ~8% of new_deaths)
df.loc[np.random.choice(df.index, size=int(len(df) * 0.05), replace=False), "new_cases"] = np.nan
df.loc[np.random.choice(df.index, size=int(len(df) * 0.08), replace=False), "new_deaths"] = np.nan
df.loc[np.random.choice(df.index, size=int(len(df) * 0.04), replace=False), "total_vaccinations"] = np.nan

# 2. Introduce duplicate rows (~3%)
dup_idx = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# 3. Introduce outliers (data entry errors)
outlier_idx = np.random.choice(df.index, size=10, replace=False)
df.loc[outlier_idx, "new_cases"] = df.loc[outlier_idx, "new_cases"] * np.random.randint(50, 200)

# 4. Convert date to string (wrong dtype) and shuffle
df["date"] = df["date"].astype(str)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Add a junk column
df["notes"] = np.random.choice(["", "preliminary", "revised", np.nan, "N/A"], len(df))

df.to_csv("covid_data_raw.csv", index=False)
print(f"Generated covid_data_raw.csv with {len(df)} rows and {df.shape[1]} columns.")
print("Introduced: missing values, duplicates, inconsistent country names, outliers, wrong dtypes.")