# Databricks notebook source
# ==========================================
# UPDATED Phase 1: Ingest Raw CSVs to Bronze
# ==========================================

catalog_schema = "workspace.default" 

# 1. Load Yield Data 
df_yield = spark.read.csv("/Volumes/workspace/default/dataset/US_crop_data.csv", header=True, inferSchema=True)
df_yield.write.mode("overwrite").format("delta").saveAsTable(f"{catalog_schema}.bronze_yield")
# Hello World :)
# 2. Load the NEW County-Level Daily Weather Data
df_weather_county = spark.read.csv("/Volumes/workspace/default/dataset/county_daily_weather_2010_2024.csv", header=True, inferSchema=True)
df_weather_county.write.mode("overwrite").format("delta").saveAsTable(f"{catalog_schema}.bronze_weather_county")

print("Phase 1 Complete: 2 Bronze tables created successfully! (City mapping no longer needed)")

# Display the loaded data to verify
display(df_yield.limit(5))
display(df_weather_county.limit(5))