import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import time

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
TOKEN = "yOVBbvDtWGPTHLppTsfQxvlMCRJBDxhG"
START_YEAR = 2010
END_YEAR = 2024

headers = {"token": TOKEN}

# Target counties (STATEFP + COUNTYFP)
TARGET_GEOIDS = ["39049", "39061", "18003"]

# Weather variables (Top 5 impacting crop growth available via GHCND)
DATATYPES = [
    # 1. Temperature (Heat/Freeze stress & Crop growth)
    "TMAX", "TMIN", "TAVG",
    # 2. Precipitation (Water availability)
    "PRCP",
    # 3. Evaporation (Water loss / Transpiration)
    "EVAP",
    # 4. Sunlight (Solar radiation for photosynthesis)
    "TSUN",
    # 5. Wind (Affects evapotranspiration, and physical stress)
    "AWND"
]

# --------------------------------------------------
# STEP 1: Download Station Metadata
# --------------------------------------------------
print("Downloading station metadata...")

stations_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/stations"
stations = []
for geoid in TARGET_GEOIDS:
    offset = 1
    while True:
        params = {
            "datasetid": "GHCND",
            "locationid": f"FIPS:{geoid}",
            "limit": 1000,
            "offset": offset
        }
        r = requests.get(stations_url, headers=headers, params=params)
        
        try:
            data = r.json()
        except Exception as e:
            print(f"Error fetching stations for {geoid} (offset {offset}): {r.status_code} {r.text[:100]}")
            time.sleep(5)
            continue

        if "results" not in data:
            break

        stations.extend(data["results"])

        if len(data["results"]) < 1000:
            break

        offset += 1000
        time.sleep(0.3)

stations_df = pd.DataFrame(stations)
print("Total stations downloaded:", len(stations_df))

# --------------------------------------------------
# STEP 2: Convert Stations to GeoDataFrame
# --------------------------------------------------
stations_df = stations_df.dropna(subset=["latitude", "longitude"])

geometry = [
    Point(xy) for xy in zip(stations_df["longitude"], stations_df["latitude"])
]

stations_gdf = gpd.GeoDataFrame(stations_df, geometry=geometry, crs="EPSG:4326")

# --------------------------------------------------
# STEP 3: Load & Filter Counties (Direct from Census)
# --------------------------------------------------
print("Loading counties from Census TIGER...")

county_url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_500k.zip"

counties = gpd.read_file(county_url)
counties = counties.to_crs("EPSG:4326")

counties["GEOID"] = counties["STATEFP"] + counties["COUNTYFP"]
counties = counties[counties["GEOID"].isin(TARGET_GEOIDS)]

counties = counties.rename(columns={
    "NAME": "county_name",
    "STATEFP": "state_fips",
    "COUNTYFP": "county_fips",
    "STUSPS": "state_abbrev"
})

# --------------------------------------------------
# STEP 4: Spatial Join (Stations in Target Counties)
# --------------------------------------------------
print("Finding stations inside selected counties...")

stations_with_county = gpd.sjoin(
    stations_gdf,
    counties,
    how="inner",
    predicate="within"
)

# Prioritize major airport/climate stations (USW) because they have all 5 crop metrics, 
# unlike volunteer COOP stations which only measure rain. Sort by coverage as well.
stations_with_county["is_usw"] = stations_with_county["id"].str.startswith("GHCND:USW")
stations_with_county = stations_with_county.sort_values(
    by=["is_usw", "datacoverage"], 
    ascending=[False, False]
)

# Keep only 1 station per county to reduce downloading time
stations_with_county = stations_with_county.drop_duplicates(subset=["GEOID"])

print("Stations in selected counties:", len(stations_with_county))

# --------------------------------------------------
# STEP 5: Download Daily Weather Data (2010–2024)
# --------------------------------------------------
print("Downloading daily weather data...")

data_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
weather_records = []

for station_id in tqdm(stations_with_county["id"].unique()):

    for year in range(START_YEAR, END_YEAR + 1):

        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": f"{year}-01-01",
            "enddate": f"{year}-12-31",
            "units": "standard",
            "datatypeid": DATATYPES,
            "limit": 1000
        }

        offset = 1
        retries = 0

        while True:
            params["offset"] = offset
            r = requests.get(data_url, headers=headers, params=params)
            
            try:
                data = r.json()
                retries = 0
            except Exception as e:
                print(f"Error fetching data for {station_id} {year} (offset {offset}): {r.status_code} {r.text[:200]}")
                retries += 1
                if retries > 3:
                    print(f"Skipping {station_id} {year} offset {offset} after 3 retries.")
                    break
                time.sleep(5)
                continue

            if "results" not in data:
                break

            for record in data["results"]:
                record["station"] = station_id
                weather_records.append(record)

            if len(data["results"]) < 1000:
                break

            offset += 1000
            time.sleep(0.3)

        time.sleep(0.3)

weather_df = pd.DataFrame(weather_records)
print("Total weather records downloaded:", len(weather_df))

# --------------------------------------------------
# STEP 6: Merge County Info into Weather Data
# --------------------------------------------------
weather_df = weather_df.merge(
    stations_with_county[[
        "id",
        "county_name",
        "state_abbrev",
        "state_fips",
        "county_fips"
    ]],
    left_on="station",
    right_on="id",
    how="left"
)

# --------------------------------------------------
# STEP 7: Aggregate to County-Level Daily Averages
# --------------------------------------------------
county_daily = (
    weather_df
    .groupby([
        "state_abbrev",
        "state_fips",
        "county_name",
        "county_fips",
        "date",
        "datatype"
    ])
    .agg({"value": "mean"})
    .reset_index()
    .pivot(
        index=["state_abbrev", "state_fips", "county_name", "county_fips", "date"],
        columns="datatype",
        values="value"
    )
    .reset_index()
)

# Rename columns to remove the datatype axis name from pivot
county_daily.columns.name = None

county_daily.to_csv("county_daily_weather_2010_2024.csv", index=False)

print("Done! Saved as county_daily_weather_2010_2024.csv")