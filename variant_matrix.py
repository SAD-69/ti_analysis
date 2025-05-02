import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import geopandas as gpd

GPKG_PATH = r'c:\Users\adminitsd\Documents\ufrgs\mestrado\qgz\final_master.gpkg'

# --- 1. Define Paths and Parameters ---
mapbiomas_2000_path = r'c:\Users\adminitsd\Downloads\brazil_coverage_2000.tif'
mapbiomas_2022_path = r'c:\Users\adminitsd\Downloads\brazil_coverage_2024.tif'
aoi_geometry = gpd.read_file(GPKG_PATH, layer='pampa')  # Your GeoJSON-like geometry object for clipping
# aoi_geometry = aoi_geometry.loc[aoi_geometry['TI'] == 'Takua Ovy']

aoi_geometry.to_crs(epsg=4326, inplace=True)
# Get the MapBiomas classification dictionary (CRUCIAL!)
# This is a simplified example. You MUST get the exact class IDs from MapBiomas.
# Example: {1: 'Forest Formation', 3: 'Pasture', 4: 'Agriculture', ...}
class_dict = {
    1: "Forest",
    3: "Forest Formation",
    4: "Savana",
    5: "Mangrove",
    9: "Forest Plantation",
    11: "Wetland",
    12: "Grassland",
    21: "Mosaic of uses",
    25: "Other non vegetaded areas",
    33: "River, lake, ocean",
    41: "Other temporary Crops"
    # ... add all other relevant classes from MapBiomas
}

# --- 2. Clip Rasters to your Area of Interest ---
def clip_raster(raster_path, aoi_geom: gpd.GeoDataFrame):
    """Clips a raster to the provided geometry."""
    with rasterio.open(raster_path) as src:
        clipped_image, clipped_transform = mask(src, [aoi_geom.geometry.union_all()], crop=True, all_touched=True)
        # Read the first band, squeeze to 2D, and set no-data to NaN
        clipped_array = clipped_image[0].astype('float32')
        clipped_array[clipped_array == src.nodata] = np.nan
    return clipped_array

# Clip both years
lulc_2000 = clip_raster(mapbiomas_2000_path, aoi_geometry)
lulc_2022 = clip_raster(mapbiomas_2022_path, aoi_geometry)

# --- 3. Create the Transition Matrix ---
# Flatten the arrays from 2D images to 1D arrays for crosstabulation
flat_2000 = lulc_2000.flatten()
flat_2022 = lulc_2022.flatten()

# Create a DataFrame for easy cross-tabulation
df = pd.DataFrame({'from': flat_2000, 'to': flat_2022})

# Remove pixels with NoData (NaN) in either year
df = df.dropna()

# Convert float class values to integers for clean grouping
df['from'] = df['from'].astype(int)
df['to'] = df['to'].astype(int)

# Create the cross-tabulation (Transition Matrix)
transition_matrix = pd.crosstab(df['from'], df['to'], margins=False)

# Optional: Filter the matrix to only include classes of interest
# This helps simplify the Sankey diagram. You can comment this out to see everything.
classes_of_interest = [1, 3, 4, 5, 12, 15, 25]  # IDs from your class_dict
classes_of_interest = [0, 3, 9, 11, 12, 21, 25, 33, 41]
transition_matrix = transition_matrix.loc[classes_of_interest, classes_of_interest]

# Convert the matrix from pixel count to area in km² (assuming 30m x 30m resolution)
pixel_area_km2 = 0.03 * 0.03 # Area of one 30m pixel in km²
transition_matrix_km2 = transition_matrix * pixel_area_km2

print("Transition Matrix (Area in km²):")
print(transition_matrix_km2.round(2))

# --- 4. Build the Sankey Diagram ---
# Prepare the data for Plotly Sankey
# The Sankey diagram needs three lists:
# 1. source: indices of the starting nodes for each flow
# 2. target: indices of the ending nodes for each flow
# 3. value: values (areas) for each flow

# Get the unique class labels for 2000 and 2022
all_classes = sorted(set(transition_matrix.index) | set(transition_matrix.columns))
class_labels = [class_dict.get(cls_id, f'Class {cls_id}') for cls_id in all_classes]

# Map the class IDs to their index position in the `all_classes` list
class_to_index = {cls_id: idx for idx, cls_id in enumerate(all_classes)}

# Initialize lists for Sankey
source = []
target = []
value = []
node_customdata = [] # Will store the class ID for more interactive hover

# Iterate through the transition matrix to populate source, target, and value
for from_class in transition_matrix.index:
    for to_class in transition_matrix.columns:
        area_value = transition_matrix_km2.at[from_class, to_class]
        # Only include flows above a certain threshold to avoid clutter
        if area_value > 1.0: # e.g., only show flows larger than 1 km²
            source.append(class_to_index[from_class])
            target.append(class_to_index[to_class])
            value.append(area_value)

# Create the Sankey Diagram Figure
fig = go.Figure(data=[go.Sankey(
    node = dict(
        pad = 20,
        thickness = 20,
        line = dict(color = "black", width = 1.0),
        label = class_labels,
        # Optional: Color the nodes by type. You can define your own color map.
        # color = ["green", "yellow", "brown", ...]
    ),
    link = dict(
        source = source,
        target = target,
        value = value,
        # Color the links based on the source node
        color = 'rgba(150, 150, 150, 0.4)',
    ))]
)

# Update layout and show
fig.update_layout(
    title_text=f"Land Use Change ({2000} - {2022})<br>AOI: Your AOI Name",
    font_size=12,
    height=800, # Adjust height as needed
    width=1000  # Adjust width as needed
)

fig.show()

# Optional: Save the figure as an HTML file for sharing
# fig.write_html("lulc_sankey_diagram.html")