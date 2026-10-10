# COVID-19 Data Cleaning + Insights

A beginner-to-intermediate Python/Pandas project that simulates a real-world data pipeline: starting with a messy dataset, cleaning it, computing trend metrics, and visualizing the results.

## Project Overview

This project demonstrates:
- **Data Cleaning**: Handling missing values, fixing data types, removing duplicates, standardizing formats
- **Feature Engineering**: 7-day rolling averages, case fatality rates, daily new cases
- **Data Visualization**: Line charts, bar charts, and multi-panel plots using Matplotlib
- **Insight Writing**: A plain-language summary of findings

## Project Structure

```
covid-insights/
├── README.md
├── requirements.txt
├── generate_data.py       # Generates a realistic messy dataset
├── covid_pipeline.py      # Main cleaning + analysis pipeline
└── findings.md            # Written summary of insights
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/covid-insights.git
cd covid-insights
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the sample dataset
```bash
python generate_data.py
```
This creates `covid_data_raw.csv` — a realistic but intentionally messy dataset.

### 4. Run the pipeline
```bash
python covid_pipeline.py
```
This will:
- Clean and validate the raw data
- Compute trend metrics
- Save a cleaned CSV (`covid_data_clean.csv`)
- Generate and save charts (`output_charts.png`)
- Print a summary to the console

## Sample Output

After running the pipeline you will find:
- `covid_data_clean.csv` — the cleaned dataset ready for further analysis
- `output_charts.png` — a multi-panel visualization including daily cases, 7-day rolling averages, and case fatality rates by country

## Skills Demonstrated

| Skill | How it's used |
|---|---|
| Python | Core scripting, control flow, functions |
| Pandas | Cleaning, groupby, rolling windows, merging |
| Matplotlib | Multi-panel plots, styling, saving figures |
| Data Wrangling | Nulls, duplicates, type coercion, outliers |

## Requirements

- Python 3.8+
- pandas
- matplotlib
- numpy

## License

MIT