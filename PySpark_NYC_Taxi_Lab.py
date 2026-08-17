try:
    import pyspark
    print("PySpark already available:", pyspark.__version__)
except ImportError:
    !pip install -q pyspark
    import pyspark
    print("Installed PySpark:", pyspark.__version__)

# %% ---------------------------------------------------------

# 2. Create a SparkSession
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
import time

spark = (
    SparkSession.builder
    .appName("NYC_Taxi_BigData_Lab")
    .config("spark.sql.shuffle.partitions", "8")   # small cluster (Colab) -> fewer shuffle partitions
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)

spark

# %% ---------------------------------------------------------

# 3. Verify Spark is running and check version
print("Spark version:", spark.version)
print("Spark master:", spark.sparkContext.master)
print("Default parallelism:", spark.sparkContext.defaultParallelism)

# %% ---------------------------------------------------------

# 4. Download one month of NYC Yellow Taxi Trip Records (Parquet)
# Official TLC trip record data (public, no auth required)
import urllib.request, os

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
LOCAL_PATH = "yellow_tripdata_2024-01.parquet"

if not os.path.exists(LOCAL_PATH):
    urllib.request.urlretrieve(DATA_URL, LOCAL_PATH)

print("File size (MB):", round(os.path.getsize(LOCAL_PATH) / 1e6, 2))

# %% ---------------------------------------------------------

# 5. Load the data into a Spark DataFrame
taxi_df = spark.read.parquet(LOCAL_PATH)
taxi_df.printSchema()

# %% ---------------------------------------------------------

# 6a. Schema (already shown above), 6b. Number of records, 6c. Number of columns
num_records = taxi_df.count()
num_columns = len(taxi_df.columns)
print("Number of records:", num_records)
print("Number of columns:", num_columns)

# %% ---------------------------------------------------------

# 6d. Sample records
taxi_df.show(5, truncate=False)

# %% ---------------------------------------------------------

# 6e. Basic summary statistics
taxi_df.select(
    "trip_distance", "fare_amount", "tip_amount", "total_amount", "passenger_count"
).describe().show()

# %% ---------------------------------------------------------

# 7. Number of partitions of the DataFrame
print("Number of partitions:", taxi_df.rdd.getNumPartitions())

# %% ---------------------------------------------------------

# 1. Total number of taxi trips
total_trips = taxi_df.count()
print("Total taxi trips:", total_trips)

# %% ---------------------------------------------------------

# 2. Min, max, average trip distance
taxi_df.select(
    F.min("trip_distance").alias("min_distance"),
    F.max("trip_distance").alias("max_distance"),
    F.avg("trip_distance").alias("avg_distance"),
).show()

# %% ---------------------------------------------------------

# 3. Min, max, average fare amount
taxi_df.select(
    F.min("fare_amount").alias("min_fare"),
    F.max("fare_amount").alias("max_fare"),
    F.avg("fare_amount").alias("avg_fare"),
).show()

# %% ---------------------------------------------------------

# 4. Distribution of passenger counts
taxi_df.groupBy("passenger_count").count().orderBy("passenger_count").show()

# %% ---------------------------------------------------------

# 5. Which hour of day has the highest number of trips?
trips_by_hour = (
    taxi_df
    .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    .groupBy("pickup_hour")
    .count()
    .orderBy(F.desc("count"))
)
trips_by_hour.show(5)
print("Busiest hour:", trips_by_hour.first()["pickup_hour"])

# %% ---------------------------------------------------------

# 6. Which hour has the highest average fare?
fare_by_hour = (
    taxi_df
    .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    .groupBy("pickup_hour")
    .agg(F.avg("fare_amount").alias("avg_fare"))
    .orderBy(F.desc("avg_fare"))
)
fare_by_hour.show(5)
print("Hour with highest average fare:", fare_by_hour.first()["pickup_hour"])

# %% ---------------------------------------------------------

# 7. Top 10 pickup locations by number of trips
top10_by_trips = (
    taxi_df.groupBy("PULocationID")
    .count()
    .orderBy(F.desc("count"))
    .limit(10)
)
top10_by_trips.show()

# %% ---------------------------------------------------------

# 8. Top 10 pickup locations by total fare revenue
top10_by_revenue = (
    taxi_df.groupBy("PULocationID")
    .agg(F.sum("fare_amount").alias("total_revenue"))
    .orderBy(F.desc("total_revenue"))
    .limit(10)
)
top10_by_revenue.show()

# %% ---------------------------------------------------------

# Register the taxi DataFrame as a temporary SQL view
taxi_df.createOrReplaceTempView("taxi")

# %% ---------------------------------------------------------

# 1. Average fare and average trip distance for each passenger count
spark.sql('''
    SELECT passenger_count,
           AVG(fare_amount)   AS avg_fare,
           AVG(trip_distance) AS avg_distance
    FROM taxi
    GROUP BY passenger_count
    ORDER BY passenger_count
''').show()

# %% ---------------------------------------------------------

# 2. Top 10 pickup locations by number of trips (Spark SQL)
spark.sql('''
    SELECT PULocationID, COUNT(*) AS trip_count
    FROM taxi
    GROUP BY PULocationID
    ORDER BY trip_count DESC
    LIMIT 10
''').show()

# %% ---------------------------------------------------------

# 3. Busiest hours of the day
spark.sql('''
    SELECT HOUR(tpep_pickup_datetime) AS pickup_hour, COUNT(*) AS trip_count
    FROM taxi
    GROUP BY pickup_hour
    ORDER BY trip_count DESC
''').show()

# %% ---------------------------------------------------------

# 4. Average fare for trips grouped by hour
spark.sql('''
    SELECT HOUR(tpep_pickup_datetime) AS pickup_hour, AVG(fare_amount) AS avg_fare
    FROM taxi
    GROUP BY pickup_hour
    ORDER BY pickup_hour
''').show(24)

# %% ---------------------------------------------------------

# 5. Pickup locations with unusually high average fares (> overall mean + 2*stddev)
stats = spark.sql("SELECT AVG(fare_amount) AS m, STDDEV(fare_amount) AS s FROM taxi").first()
threshold = stats["m"] + 2 * stats["s"]
print("Threshold for 'unusually high' avg fare:", round(threshold, 2))

spark.sql(f'''
    SELECT PULocationID, AVG(fare_amount) AS avg_fare, COUNT(*) AS trip_count
    FROM taxi
    GROUP BY PULocationID
    HAVING AVG(fare_amount) > {threshold}
    ORDER BY avg_fare DESC
''').show()

# %% ---------------------------------------------------------

# Compare DataFrame API vs Spark SQL for two problems (top pickup locations, busiest hour)
t0 = time.time()
df_api_result = taxi_df.groupBy("PULocationID").count().orderBy(F.desc("count")).limit(10).collect()
t1 = time.time()
sql_result = spark.sql('''
    SELECT PULocationID, COUNT(*) AS trip_count FROM taxi
    GROUP BY PULocationID ORDER BY trip_count DESC LIMIT 10
''').collect()
t2 = time.time()

print(f"DataFrame API time: {t1 - t0:.3f}s")
print(f"Spark SQL time:     {t2 - t1:.3f}s")
print("Results match:", [r['PULocationID'] for r in df_api_result] == [r['PULocationID'] for r in sql_result])

# %% ---------------------------------------------------------

# 1 & 2. Identify suspicious records and define filtering rules
suspicious_counts = {
    "zero_or_negative_distance": taxi_df.filter(F.col("trip_distance") <= 0).count(),
    "zero_or_negative_fare":     taxi_df.filter(F.col("fare_amount") <= 0).count(),
    "invalid_passenger_count":   taxi_df.filter((F.col("passenger_count") <= 0) | (F.col("passenger_count") > 6)).count(),
    "extremely_long_distance":   taxi_df.filter(F.col("trip_distance") > 100).count(),
    "extremely_large_fare":      taxi_df.filter(F.col("fare_amount") > 500).count(),
    "invalid_timestamps":        taxi_df.filter(F.col("tpep_dropoff_datetime") <= F.col("tpep_pickup_datetime")).count(),
}
for k, v in suspicious_counts.items():
    print(f"{k}: {v}")

# %% ---------------------------------------------------------

# 3. Create a cleaned DataFrame applying reasonable filtering rules
cleaned_df = taxi_df.filter(
    (F.col("trip_distance") > 0) & (F.col("trip_distance") <= 100) &
    (F.col("fare_amount") > 0) & (F.col("fare_amount") <= 500) &
    (F.col("passenger_count") > 0) & (F.col("passenger_count") <= 6) &
    (F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime"))
)

cleaned_count = cleaned_df.count()

# %% ---------------------------------------------------------

# 4. Report how many records were removed
removed = num_records - cleaned_count
print(f"Original records: {num_records}")
print(f"Cleaned records:  {cleaned_count}")
print(f"Removed records:  {removed} ({removed / num_records:.2%})")

# %% ---------------------------------------------------------

# 5. Derived features: trip duration, pickup hour, day of week, weekend indicator,
#    fare per mile, average speed
cleaned_df = (
    cleaned_df
    .withColumn(
        "trip_duration_min",
        (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0
    )
    .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    .withColumn("day_of_week", F.date_format("tpep_pickup_datetime", "E"))  # Mon, Tue, ...
    .withColumn(
        "is_weekend",
        F.when(F.dayofweek("tpep_pickup_datetime").isin([1, 7]), 1).otherwise(0)  # 1=Sun,7=Sat
    )
    .withColumn("fare_per_mile", F.col("fare_amount") / F.col("trip_distance"))
    .withColumn(
        "avg_speed_mph",
        F.when(F.col("trip_duration_min") > 0,
               F.col("trip_distance") / (F.col("trip_duration_min") / 60.0))
    )
)

# Additionally remove rows where duration became invalid after feature engineering
cleaned_df = cleaned_df.filter((F.col("trip_duration_min") > 0) & (F.col("trip_duration_min") < 240))

cleaned_df.select(
    "trip_distance", "fare_amount", "trip_duration_min", "pickup_hour",
    "day_of_week", "is_weekend", "fare_per_mile", "avg_speed_mph"
).show(5)

# %% ---------------------------------------------------------

def run_aggregation(df):
    return df.groupBy("PULocationID").agg(F.avg("fare_amount").alias("avg_fare")).collect()

results = []

# %% ---------------------------------------------------------

# Experiment 1: original DataFrame
print("Original partitions:", cleaned_df.rdd.getNumPartitions())

t0 = time.time()
run_aggregation(cleaned_df)
t1 = time.time()
exp1_time = t1 - t0
results.append(("Original", cleaned_df.rdd.getNumPartitions(), "No", round(exp1_time, 3)))
print(f"Experiment 1 (Original) time: {exp1_time:.3f}s")

# %% ---------------------------------------------------------

# Experiment 2: repartitioned DataFrame
repartitioned_df = cleaned_df.repartition(16)
print("Repartitioned partitions:", repartitioned_df.rdd.getNumPartitions())

t0 = time.time()
run_aggregation(repartitioned_df)
t1 = time.time()
exp2_time = t1 - t0
results.append(("Repartitioned", repartitioned_df.rdd.getNumPartitions(), "No", round(exp2_time, 3)))
print(f"Experiment 2 (Repartitioned) time: {exp2_time:.3f}s")

# %% ---------------------------------------------------------

# Experiment 3: cache/persist and repeat
cached_df = cleaned_df.cache()
cached_df.count()  # materialize the cache

t0 = time.time()
run_aggregation(cached_df)
t1 = time.time()
exp3_time_first = t1 - t0

t0 = time.time()
run_aggregation(cached_df)
t1 = time.time()
exp3_time_second = t1 - t0

results.append(("Cached (1st run)", cached_df.rdd.getNumPartitions(), "Yes", round(exp3_time_first, 3)))
results.append(("Cached (2nd run)", cached_df.rdd.getNumPartitions(), "Yes", round(exp3_time_second, 3)))

print(f"Experiment 3 (Cached, 1st run):  {exp3_time_first:.3f}s")
print(f"Experiment 3 (Cached, 2nd run):  {exp3_time_second:.3f}s")

# %% ---------------------------------------------------------

# Results table
import pandas as pd  # display only — not used for the actual Spark computation
results_df = pd.DataFrame(results, columns=["Experiment", "Number of Partitions", "Cached?", "Execution Time (s)"])
results_df

# %% ---------------------------------------------------------

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# %% ---------------------------------------------------------

# 1 & 2. Select features and prepare the feature vector
ml_df = (
    cleaned_df
    .withColumn("day_of_week_idx_input", F.col("day_of_week"))
    .select(
        "trip_distance", "passenger_count", "trip_duration_min",
        "pickup_hour", "day_of_week_idx_input", "fare_amount"
    )
    .na.drop()
)

day_indexer = StringIndexer(inputCol="day_of_week_idx_input", outputCol="day_of_week_idx")

feature_cols = ["trip_distance", "passenger_count", "trip_duration_min", "pickup_hour", "day_of_week_idx"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# %% ---------------------------------------------------------

# 3. Train/test split
train_df, test_df = ml_df.randomSplit([0.8, 0.2], seed=42)
print("Training rows:", train_df.count())
print("Test rows:", test_df.count())

# %% ---------------------------------------------------------

# 4 & 5. Choose an algorithm and train the model
# RandomForestRegressor is chosen because it: (1) handles non-linear relationships between
# distance/duration/hour and fare, (2) is robust to outliers/skewed features without needing
# feature scaling, and (3) scales well as a distributed Spark ML algorithm.
rf = RandomForestRegressor(featuresCol="features", labelCol="fare_amount", numTrees=50, maxDepth=8, seed=42)

pipeline = Pipeline(stages=[day_indexer, assembler, rf])

model = pipeline.fit(train_df)

# %% ---------------------------------------------------------

# 6. Generate predictions on the test set
predictions = model.transform(test_df)
predictions.select("trip_distance", "trip_duration_min", "pickup_hour", "fare_amount", "prediction").show(10)

# %% ---------------------------------------------------------

# 7. Evaluate the model
evaluator_rmse = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse")
evaluator_mae  = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="mae")
evaluator_r2   = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae  = evaluator_mae.evaluate(predictions)
r2   = evaluator_r2.evaluate(predictions)

print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R2:   {r2:.3f}")

# %% ---------------------------------------------------------

# Feature importances
rf_model = model.stages[-1]
for name, importance in zip(feature_cols, rf_model.featureImportances.toArray()):
    print(f"{name}: {importance:.4f}")

# %% ---------------------------------------------------------

# Example: download a second month and union it to observe how execution time scales
DATA_URL_2 = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet"
LOCAL_PATH_2 = "yellow_tripdata_2024-02.parquet"

if not os.path.exists(LOCAL_PATH_2):
    urllib.request.urlretrieve(DATA_URL_2, LOCAL_PATH_2)

taxi_df_2 = spark.read.parquet(LOCAL_PATH_2)
combined_df = taxi_df.unionByName(taxi_df_2)

t0 = time.time()
combined_count = combined_df.count()
t1 = time.time()

print(f"Combined record count: {combined_count} (vs {num_records} for one month)")
print(f"Count() execution time on combined data: {t1 - t0:.3f}s")

# %% ---------------------------------------------------------

partition_options = [1, 4, 8, 16, 32, 64]
partition_results = []

for n in partition_options:
    test_df_part = cleaned_df.repartition(n)
    t0 = time.time()
    run_aggregation(test_df_part)
    t1 = time.time()
    partition_results.append((n, round(t1 - t0, 3)))
    print(f"Partitions={n}: {t1 - t0:.3f}s")

pd.DataFrame(partition_results, columns=["Partitions", "Execution Time (s)"])

# %% ---------------------------------------------------------

from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(featuresCol="features", labelCol="fare_amount", maxIter=50, maxDepth=6, seed=42)
gbt_pipeline = Pipeline(stages=[day_indexer, assembler, gbt])
gbt_model = gbt_pipeline.fit(train_df)

gbt_predictions = gbt_model.transform(test_df)

gbt_rmse = evaluator_rmse.evaluate(gbt_predictions)
gbt_mae  = evaluator_mae.evaluate(gbt_predictions)
gbt_r2   = evaluator_r2.evaluate(gbt_predictions)

print("Random Forest -> RMSE: {:.3f}, MAE: {:.3f}, R2: {:.3f}".format(rmse, mae, r2))
print("GBT           -> RMSE: {:.3f}, MAE: {:.3f}, R2: {:.3f}".format(gbt_rmse, gbt_mae, gbt_r2))

# %% ---------------------------------------------------------

# Build an hourly-demand-per-location training table
demand_df = (
    cleaned_df
    .groupBy("PULocationID", "pickup_hour", "day_of_week", "is_weekend")
    .agg(F.count("*").alias("trip_count"))
)

demand_indexer = StringIndexer(inputCol="day_of_week", outputCol="day_of_week_idx")
demand_assembler = VectorAssembler(
    inputCols=["PULocationID", "pickup_hour", "day_of_week_idx", "is_weekend"],
    outputCol="features"
)

demand_train, demand_test = demand_df.randomSplit([0.8, 0.2], seed=42)

demand_rf = RandomForestRegressor(featuresCol="features", labelCol="trip_count", numTrees=50, seed=42)
demand_pipeline = Pipeline(stages=[demand_indexer, demand_assembler, demand_rf])
demand_model = demand_pipeline.fit(demand_train)

demand_predictions = demand_model.transform(demand_test)

demand_evaluator = RegressionEvaluator(labelCol="trip_count", predictionCol="prediction", metricName="rmse")
print("Demand model RMSE:", demand_evaluator.evaluate(demand_predictions))