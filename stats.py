import rasterio
import requests
import io
import os
import geopandas as gpd
import numpy as np

from rasterstats import zonal_stats

URL = 'https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection_10/lulc/coverage/brazil_coverage_{0}.tif'
GPKG_PATH = r'c:\Users\adminitsd\Documents\ufrgs\mestrado\qgz\final_master.gpkg'


years = [2000, 2022]
# Define paths to your raster and zone shapefile

def get_raster(url: str):
    r = requests.get(url)
    content = io.BytesIO(r.content)
    return content

raster_path = "path/to/your/raster.tif"

# Load zone features
zones = gpd.read_file(GPKG_PATH, layer='se_rs_clip')

# Calculate zonal statistics
# You can specify a list of desired statistics
stats = zonal_stats(zones, raster_path, stats=['mean', 'sum', 'min', 'max'])

# The 'stats' variable will be a list of dictionaries, 
# where each dictionary contains the calculated statistics for a zone.
# You can then add these statistics back to your GeoDataFrame if desired.
zones['mean_value'] = [s['mean'] for s in stats]
zones['sum_value'] = [s['sum'] for s in stats]

# Print or further process the results
print(zones.head())