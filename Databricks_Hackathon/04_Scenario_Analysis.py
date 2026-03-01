# Databricks notebook source
# DBTITLE 1,Cell 1
# ==========================================
# Phase 4: Scenario Analysis & Prediction
# ==========================================
import pandas as pd
import numpy as np
import mlflow.sklearn

catalog_schema = "workspace.default"

# 1. Load the "Pristine" 2014 Data and the Trained Model
df_2014_spark = spark.table(f"{catalog_schema}.gold_production_2014")
df_2014 = df_2014_spark.toPandas()

# ---------------------------------------------------------
# THE FIX: Keep only one row per County and Crop to clean up the demo UI
# ---------------------------------------------------------
df_2014 = df_2014.drop_duplicates(subset=["County", "Crop"]).reset_index(drop=True)

# Load the model using the Run ID from Phase 3 
run_id = "7d4a52cc20644e9bae21ec8655a02654" 
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# 2. Define the Uncertainty Bounds (Manually enter the numbers from Phase 3)
LOWER_BOUND_ERROR = -7.33 
UPPER_BOUND_ERROR = 18.31 

# ---------------------------------------------------------
# STEP A: Create the Scenarios
# ---------------------------------------------------------

# Scenario 1: Actual 2014 Weather (Baseline)
df_baseline = df_2014.copy()

# Scenario 2: Severe Heatwave Simulation
df_heatwave = df_2014.copy()
df_heatwave["90d_Days_Above_95F"] = df_heatwave["90d_Days_Above_95F"] + 15
df_heatwave["60d_Avg_Temp"] = df_heatwave["60d_Avg_Temp"] + 5
df_heatwave["90d_Avg_Temp"] = df_heatwave["90d_Avg_Temp"] + 5
# Hello World :)
# Scenario 3: Drought Simulation
df_drought = df_2014.copy()
df_drought["60d_Total_Precip"] = df_drought["60d_Total_Precip"] * 0.5
df_drought["90d_Total_Precip"] = df_drought["90d_Total_Precip"] * 0.5

# ---------------------------------------------------------
# STEP B: Generate Predictions & Apply Uncertainty
# ---------------------------------------------------------
def get_predictions(df, scenario_name):
    # Features used for prediction (must match Phase 3 exactly)
    features = df.drop(columns=["Yield_Per_Acre", "Yield_Status", "Harvest_Date"])
    preds = model.predict(features)
    
    df_res = df[["County", "Crop", "Yield_Per_Acre"]].copy()
    df_res["Scenario"] = scenario_name
    df_res["Predicted_Yield"] = preds
    df_res["Lower_Confidence_Bound"] = preds + LOWER_BOUND_ERROR
    df_res["Upper_Confidence_Bound"] = preds + UPPER_BOUND_ERROR
    return df_res

# Run the engine
results_baseline = get_predictions(df_baseline, "Baseline (Actual)")
results_heatwave = get_predictions(df_heatwave, "Simulated Heatwave")
results_drought = get_predictions(df_drought, "Simulated Drought")

# Combine all results
final_comparison = pd.concat([results_baseline, results_heatwave, results_drought])

# 3. Calculate "Risk" (Percentage drop from baseline)
# Using a clean map now that we guarantee unique County/Crop rows
baseline_map = results_baseline.set_index(["County", "Crop"])["Predicted_Yield"]
final_comparison['Yield_Risk_%'] = final_comparison.apply(
    lambda x: ((x['Predicted_Yield'] - baseline_map.loc[(x['County'], x['Crop'])]) / baseline_map.loc[(x['County'], x['Crop'])]) * 100, axis=1
)

# Convert the Pandas DataFrame back to a Spark DataFrame
df_final_spark = spark.createDataFrame(final_comparison)

# Save it as our final presentation table
presentation_table = f"{catalog_schema}.scenario_results_2014"
df_final_spark.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(presentation_table)

print(f"Ready for the Dashboard! Data saved to {presentation_table}")

# Display results clearly formatted
print("Scenario Analysis Complete!")
display(final_comparison.sort_values(by=["County", "Crop", "Scenario"])[
    ["County", "Crop", "Scenario", "Predicted_Yield", "Yield_Risk_%", "Lower_Confidence_Bound", "Upper_Confidence_Bound"]
])

# COMMAND ----------



# COMMAND ----------

