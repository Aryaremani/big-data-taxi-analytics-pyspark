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

- Trip volume shows a clear, repeatable peak-demand hour across the day (see notebook output
  for the exact hour from this run).
- Revenue is concentrated in a small number of pickup zones. The top 10 zones by total
  revenue are not always the same as the top 10 by trip count, which has direct implications
  for where vehicles should be positioned.
- Approximately [X]% of raw records were identified as invalid (zero or negative fares,
  invalid distances or passenger counts, corrupted timestamps) and removed prior to analysis.
- A Random Forest regression model, trained with Spark MLlib, predicts fare amount from trip
  distance, duration, pickup hour, and day of week, achieving an RMSE of [value] and an R² of
  [value] on the held-out test set.

*(Bracketed values reflect placeholders to be filled in with results from an executed run —
see the "How to Run" section below.)*

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
ARYA R 
aryaramani94@gmail.com
