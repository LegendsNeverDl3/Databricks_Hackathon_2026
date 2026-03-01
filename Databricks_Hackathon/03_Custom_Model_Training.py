# Databricks notebook source
# Hello World :)
# ==========================================
# Phase 3: Custom Model Training, Cross Validation & MLflow
# ==========================================
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

catalog_schema = "workspace.default" 

# 1. Load the Gold Table and convert to Pandas for Scikit-Learn
df_gold_spark = spark.table(f"{catalog_schema}.gold_training_data")
df_gold = df_gold_spark.toPandas()

# 2. Separate Features (X) and Target (y)
# VERY IMPORTANT: Drop 'Yield_Status' and 'Harvest_Date' to prevent data leakage!
X = df_gold.drop(columns=["Yield_Per_Acre", "Yield_Status", "Harvest_Date"])
y = df_gold["Yield_Per_Acre"]

numeric_features = ["Year", "5_Year_Avg_Yield", "60d_Total_Precip", "60d_Avg_Temp", 
                    "90d_Total_Precip", "90d_Avg_Temp", "90d_Days_Above_95F"]
categorical_features = ["County", "State", "Crop"]

# 3. Build the Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))
])

# 4. Split the data into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 5. Train the Model and Track with MLflow
# ==========================================
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Crop_Yield_RF_with_CV") as run:
    
    print("🔄 Performing 5-Fold Cross Validation on Training Data...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    cv_mean_rmse = np.sqrt(-cv_scores).mean()
    mlflow.log_metric("cv_mean_rmse", cv_mean_rmse)
    
    print(f"   CV Mean RMSE: {cv_mean_rmse:.2f} bushels/acre")
    print("-" * 50)
    
    print("🚀 Training Final Model and Calculating R-Squared...")
    model_pipeline.fit(X_train, y_train)
    
    # --- NEW: Calculate Train vs Test R-Squared (Accuracy) ---
    train_predictions = model_pipeline.predict(X_train)
    test_predictions = model_pipeline.predict(X_test)
    
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    # Calculate final overall error metrics on test set
    final_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    final_mae = mean_absolute_error(y_test, test_predictions)
    
    # Log our custom R2 metrics to MLflow
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("test_r2", test_r2)
    
    print(f"   Train R² Score: {train_r2:.3f} (How well it learned)")
    print(f"   Test R² Score:  {test_r2:.3f} (How well it generalizes)")
    print(f"   Final Test RMSE: {final_rmse:.2f} bushels/acre")
    print(f"✅ Model saved to MLflow Run ID: {run.info.run_id}")
    print("-" * 50)

# ==========================================
# 6. Calculate Uncertainty Bands
# ==========================================
residuals = y_test - test_predictions
error_10th = np.percentile(residuals, 10)
error_90th = np.percentile(residuals, 90)

print("🎯 UNCERTAINTY BANDS CALCULATED:")
print(f"Worst-Case Drop (10th Percentile Error): {error_10th:.2f} bushels/acre")
print(f"Best-Case Bump (90th Percentile Error):  +{error_90th:.2f} bushels/acre")
print("-" * 50)

# COMMAND ----------

import mlflow.sklearn

loaded_pipeline = mlflow.sklearn.load_model("runs:/7d4a52cc20644e9bae21ec8655a02654/model")

# COMMAND ----------

# Rebuild predictions from the loaded model
test_predictions = loaded_pipeline.predict(X_test)
train_predictions = loaded_pipeline.predict(X_train)

residuals = y_test - test_predictions
error_10th = np.percentile(residuals, 10)
error_90th = np.percentile(residuals, 90)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
# ---------------------------------------------------------
# 1. Prediction Intervals & Uncertainty Bands
# ---------------------------------------------------------
sort_idx = np.argsort(test_predictions)
sorted_preds = test_predictions[sort_idx]
sorted_actuals = y_test.values[sort_idx]

lower_band = sorted_preds + error_10th
upper_band = sorted_preds + error_90th

fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(range(len(sorted_preds)), lower_band, upper_band,
                alpha=0.3, color="skyblue", label="80% Prediction Interval")
ax.plot(range(len(sorted_preds)), sorted_preds, color="blue", linewidth=2, label="Predicted Yield")
ax.scatter(range(len(sorted_actuals)), sorted_actuals, color="red", s=30, zorder=5, label="Actual Yield")
ax.set_xlabel("Test Samples (sorted by prediction)")
ax.set_ylabel("Yield (bushels/acre)")
ax.set_title("Crop Yield Predictions with Uncertainty Bands")
ax.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2. Feature Importance
# ---------------------------------------------------------
regressor = model_pipeline.named_steps["regressor"]
importances = regressor.feature_importances_

num_names = numeric_features
cat_names = list(model_pipeline.named_steps["preprocessor"]
                 .named_transformers_["cat"]
                 .get_feature_names_out(categorical_features))
all_feature_names = num_names + cat_names

feat_importance = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
top_10 = feat_importance.head(10)
ax.barh(top_10["Feature"], top_10["Importance"], color="steelblue")
ax.set_xlabel("Feature Importance")
ax.set_title("Top 10 Factors Affecting Crop Yield")
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 3. Farmer Summary
# ---------------------------------------------------------
print("=" * 60)
print("🌾 FARMER-FRIENDLY YIELD SUMMARY")
print("=" * 60)

df_analysis = X.copy()
df_analysis["Yield_Per_Acre"] = y.values

high_heat = df_analysis[df_analysis["90d_Days_Above_95F"] > 4]["Yield_Per_Acre"].mean()
low_heat = df_analysis[df_analysis["90d_Days_Above_95F"] <= 4]["Yield_Per_Acre"].mean()
heat_impact = ((high_heat - low_heat) / low_heat) * 100
print(f"\n🌡️  HEAT STRESS:")
print(f"   If temperatures exceed 95°F for more than 4 days in the")
print(f"   90 days before harvest, expected yield changes by {heat_impact:.1f}%.")

median_precip = df_analysis["90d_Total_Precip"].median()
high_rain = df_analysis[df_analysis["90d_Total_Precip"] > median_precip]["Yield_Per_Acre"].mean()
low_rain = df_analysis[df_analysis["90d_Total_Precip"] <= median_precip]["Yield_Per_Acre"].mean()
rain_impact = ((high_rain - low_rain) / low_rain) * 100
print(f"\n🌧️  PRECIPITATION:")
print(f"   When 90-day rainfall is above {median_precip:.1f} units,")
print(f"   yield changes by {rain_impact:.1f}% compared to drier conditions.")

median_temp = df_analysis["90d_Avg_Temp"].median()
warm = df_analysis[df_analysis["90d_Avg_Temp"] > median_temp]["Yield_Per_Acre"].mean()
cool = df_analysis[df_analysis["90d_Avg_Temp"] <= median_temp]["Yield_Per_Acre"].mean()
temp_impact = ((warm - cool) / cool) * 100
print(f"\n🌡️  AVERAGE TEMPERATURE:")
print(f"   When the 90-day average temperature exceeds {median_temp:.1f}°F,")
print(f"   yield changes by {temp_impact:.1f}%.")

print(f"\n📊 PREDICTION CONFIDENCE:")
print(f"   Our model predictions are accurate within a range of")
print(f"   {error_10th:.1f} to +{error_90th:.1f} bushels/acre (80% of the time).")
print("=" * 60)

# COMMAND ----------

# Create a results dataframe with predictions + uncertainty bands
import pyspark.sql.functions as F

df_results = X_test.copy()
df_results["Yield_Per_Acre"] = y_test.values
df_results["Predicted_Yield"] = test_predictions
df_results["Lower_Bound"] = test_predictions + error_10th
df_results["Upper_Bound"] = test_predictions + error_90th
df_results["Residual"] = residuals.values

# Convert to Spark and save as Delta table
spark_results = spark.createDataFrame(df_results)
spark_results.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(f"{catalog_schema}.gold_predictions")

print("✅ Predictions table saved to workspace.default.gold_predictions")
display(spark_results.limit(10))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   County, Crop, Year,
# MAGIC   Predicted_Yield,
# MAGIC   Yield_Per_Acre AS Actual_Yield,
# MAGIC   Lower_Bound,
# MAGIC   Upper_Bound
# MAGIC FROM workspace.default.gold_predictions
# MAGIC ORDER BY Predicted_Yield

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT County, Crop, Year, 
# MAGIC   AVG(Yield_Per_Acre) as Avg_Yield,
# MAGIC   AVG(Predicted_Yield) as Avg_Predicted
# MAGIC FROM workspace.default.gold_predictions
# MAGIC GROUP BY County, Crop, Year
# MAGIC ORDER BY Year

# COMMAND ----------

# Add this after the feature importance calculation in Phase 3
spark_importance = spark.createDataFrame(feat_importance)
spark_importance.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(f"{catalog_schema}.gold_feature_importance")

# COMMAND ----------

