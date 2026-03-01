# Quick Information



### `data` folder

This holds all the csv files for the data that we used for our python code



### `databricks_data_code` folder

The python scripts in here use an api key from NOAA (NCDC web service key token request).

### `Databricks_Hackathon_2026` folder
This folder holds all the python notebooks (applicable ones are number 01 through 04 in the names) that handle things such as loading the csv files, using databricks library things, cleaning up the data into forms that we want, creating a model (random forests + scikit learn).

# About the project
See our devpost link for a video: [devpost](https://devpost.com/software/databricks-temp-1)

### Tech stack
The goal was to use as much Databricks internal tools as possible. We used Databrick's *MLflow* when creating our model and loaded it with Databricks *Genie* so that users can communicate with Genie about the actual statistics. We also created a pipeline/job in *Jobs & Pipelines* that automatically updates on a trigger, which is when the dataset folder experiences any changes. Additionally, we created visualizations in Databrick's *Dashboards* area.

### App
We made an external app that is not actually connected with *Genie* in the backend (due to lack of time), but that helps display what it might look like: [github](https://github.com/SamyagJ/databricksCropForecastingHackathon) and [website](https://databricks-crop-forecasting-hackath-omega.vercel.app/)







