# PySpark NYC Taxi Analytics — Big Data Pipeline and Fare Prediction

An end-to-end big data analytics project that processes over 3 million NYC Yellow Taxi trip
records using PySpark. The project covers distributed data ingestion, exploratory analysis,
Spark SQL, data cleaning, feature engineering, performance tuning, and machine learning —
built entirely on Apache Spark's DataFrame API and MLlib, without Pandas or Scikit-learn in
the core pipeline.

**Tech stack:** PySpark · Spark SQL · Spark MLlib · Google Colab

---

## Overview

Acting as a data analyst for a taxi transportation company, this project analyzes trip-level
data to answer a practical business question: how can travel patterns, revenue concentration,
and operational data be used to improve driver scheduling and fleet positioning decisions?

The dataset is one month of public NYC Taxi and Limousine Commission (TLC) trip records,
containing several million rows, processed in a distributed manner using Apache Spark.

## Key Findings

- Analyzed 2,964,624 taxi trips from January 2024, spanning 19 columns of trip-level data.
- Trip volume peaks at **6 PM (hour 18)**, with 212,788 trips — the clear busiest hour of the
  day. Demand tapers to a low around 6 AM (41,429 trips).
- The highest **average fare** occurs at **5 AM** ($26.62), driven by fewer, longer trips
  (e.g., airport runs) rather than by traffic-driven surcharges.
- Revenue is concentrated far more sharply than trip volume: pickup zone **132** alone
  generated **$8.63M** in fares — nearly 2.5x the next-highest zone — while ranking only 1st
  by trip count as well, making it the single most valuable zone in the dataset by both
  measures.
- **240,971 records (8.13%)** of the raw dataset were identified as invalid (zero/negative
  fares or distances, invalid passenger counts, corrupted timestamps) and removed before
  analysis.
- A **Random Forest regression model** (Spark MLlib) predicting fare amount from trip
  distance, duration, pickup hour, and day of week achieved **RMSE = 6.20, MAE = 1.64, R² =
  0.871** on the held-out test set. Trip distance (65%) and trip duration (34%) accounted for
  nearly all of the model's predictive power.
- A second model (Gradient-Boosted Trees) performed marginally better (RMSE = 6.12, R² =
  0.874), at the cost of longer training time — see the Challenge section of the notebook.
- The performance experiment showed that **caching provided the clearest speedup** (0.91s →
  0.28s on a repeated aggregation), while blind repartitioning on a small 2-core Colab runtime
  actually slowed the same operation down (0.91s → 4.96s) due to shuffle overhead exceeding
  the benefit of added parallelism.

## Repository Contents

| File | Description |
|---|---|
| `PySpark_NYC_Taxi_Lab.ipynb` | Complete analysis notebook, intended to be run end-to-end in Google Colab |
| `PySpark_NYC_Taxi_Lab.py` | The same analysis as a flat Python script, for reference outside a notebook environment |
| `requirements.txt` | Python package dependencies |

## Methodology

1. **Environment setup and data ingestion** — SparkSession configuration, Parquet loading, schema and partition inspection
2. **Exploratory data analysis** — trip volume, fare and distance distributions, passenger counts, peak-demand hours, top pickup locations
3. **Spark SQL analysis** — equivalent queries expressed declaratively via temporary SQL views, compared against the DataFrame API
4. **Data cleaning and feature engineering** — identification and removal of invalid records; derivation of trip duration, pickup hour, day of week, weekend indicator, fare per mile, and average speed
5. **Performance experimentation** — execution-time comparison across default, repartitioned, and cached DataFrame configurations
6. **Machine learning** — a Random Forest regression model built with Spark MLlib to predict fare amount, evaluated using RMSE, MAE, and R²
7. **Business recommendations** — actionable, evidence-based recommendations for driver scheduling and fleet positioning

## Data Source

[NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) —
publicly available Parquet files published by the New York City Taxi and Limousine Commission.

## How to Run

1. Open `PySpark_NYC_Taxi_Lab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Run all cells sequentially (Runtime → Run all). The notebook downloads the required TLC
   Parquet file automatically; no manual data upload is needed.
3. A full run typically takes 10–20 minutes, depending on the allocated Colab runtime.

## Author

[Your Name]
[LinkedIn] · [Email or portfolio link]
