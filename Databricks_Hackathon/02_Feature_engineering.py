# Databricks notebook source
# ==========================================
# REVISED Phase 2: Feature Engineering (60 & 90 Day Windows)
# ==========================================
from pyspark.sql.window import Window
from pyspark.sql.functions import col, avg, sum, count, when, year, coalesce, datediff, to_date

catalog_schema = "workspace.default" 

# 1. Load the Bronze Tables
df_yield = spark.table(f"{catalog_schema}.bronze_yield")
df_weather = spark.table(f"{catalog_schema}.bronze_weather_county")

# ---------------------------------------------------------
# STEP A: Filter for the 3 Target Counties
# ---------------------------------------------------------
df_yield = df_yield.filter(
    ((col("County") == "Franklin") & (col("State") == "Ohio")) |
    ((col("County") == "Hamilton") & (col("State") == "Ohio")) |
    ((col("County") == "Allen") & (col("State") == "Indiana"))
)

df_weather = df_weather.filter(
    ((col("county_name") == "Franklin") & (col("state_abbrev") == "OH")) |
    ((col("county_name") == "Hamilton") & (col("state_abbrev") == "OH")) |
    ((col("county_name") == "Allen") & (col("state_abbrev") == "IN"))
)

# ---------------------------------------------------------
# STEP B: Calculate "Good" vs "Average" Yield (5-Year Rolling)
# ---------------------------------------------------------
windowSpec = (Window.partitionBy("County", "Crop").orderBy("Year").rowsBetween(-5, -1))

df_yield_engineered = df_yield.withColumn("5_Year_Avg_Yield", avg("Yield_Per_Acre").over(windowSpec))

df_yield_engineered = df_yield_engineered.withColumn(
    "Yield_Status",
    when(col("Yield_Per_Acre") > col("5_Year_Avg_Yield") * 1.05, "Good")
    .when(col("Yield_Per_Acre") < col("5_Year_Avg_Yield") * 0.95, "Poor")
    .otherwise("Average")
)

# ---------------------------------------------------------
# STEP C: Prepare Weather & JOIN BEFORE AGGREGATING
# ---------------------------------------------------------
# Clean the temperature column
df_weather = df_weather.withColumn("TAVG_Cleaned", coalesce(col("TAVG"), (col("TMAX") + col("TMIN")) / 2))
df_weather = df_weather.withColumn("Weather_Year", year(col("date")))

# Join the Daily Weather directly to the Yield data so we have Harvest_Date attached to every weather day
df_joined = df_yield_engineered.join(
    df_weather,
    (df_yield_engineered["County"] == df_weather["county_name"]) & 
    (df_yield_engineered["Year"] == df_weather["Weather_Year"]),
    "inner"
)

# Calculate exactly how many days prior to harvest each weather observation was
df_joined = df_joined.withColumn(
    "days_before_harvest", 
    datediff(to_date(col("Harvest_Date")), to_date(col("date")))
)

# ---------------------------------------------------------
# STEP D: Create the 60- and 90-Day Pre-Harvest Aggregations
# ---------------------------------------------------------
# Added a filter here to only include years 2010 through 2013
df_gold_all = df_joined.groupBy(
    "County", "State", "Year", "Crop", "Harvest_Date", "Yield_Per_Acre", "5_Year_Avg_Yield", "Yield_Status"
).agg(
    sum(when((col("days_before_harvest") >= 0) & (col("days_before_harvest") <= 60), col("PRCP")).otherwise(0)).alias("60d_Total_Precip"),
    avg(when((col("days_before_harvest") >= 0) & (col("days_before_harvest") <= 60), col("TAVG_Cleaned"))).alias("60d_Avg_Temp"),
    sum(when((col("days_before_harvest") >= 0) & (col("days_before_harvest") <= 90), col("PRCP")).otherwise(0)).alias("90d_Total_Precip"),
    avg(when((col("days_before_harvest") >= 0) & (col("days_before_harvest") <= 90), col("TAVG_Cleaned"))).alias("90d_Avg_Temp"),
    count(when((col("days_before_harvest") >= 0) & (col("days_before_harvest") <= 90) & (col("TMAX") >= 95), True)).alias("90d_Days_Above_95F")
)

# ---------------------------------------------------------
# STEP E: Split and Save Tables
# ---------------------------------------------------------

# 1. Training Data (2010 - 2013)
df_training = df_gold_all.filter(col("Year") <= 2012).dropna(subset=["5_Year_Avg_Yield"])
df_training.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(f"{catalog_schema}.gold_training_data")

# 2. Production/Test Data (2014) - This is for Phase 4 Scenarios
df_production = df_gold_all.filter(col("Year") == 2013).dropna(subset=["5_Year_Avg_Yield"])
df_production.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(f"{catalog_schema}.gold_production_2014")

print("Phase 2 Complete!")
print(f"Table 1: {catalog_schema}.gold_training_data (2010-2013) -> For Phase 3")
print(f"Table 2: {catalog_schema}.gold_production_2014 (2014) -> For Phase 4 Scenarios")
display(df_training.limit(10))
display(df_production.limit(10))

# Hello World :)