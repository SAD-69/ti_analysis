import numpy as np
import pandas as pd
from models.gpkg import GeoPackage
from scipy.spatial import KDTree

gpkg = GeoPackage()
# 1. Load the Data
# Assuming you have a GeoPackage class instance `gpkg`
tis = gpkg.read_layer('ti_ac_preliminar_tx_alfabetizacao')  # Your TI layer
defense_institutions = gpkg.read_layer('instituicoes_indigena')  # Your polygon layer for institutions

# 2. Preprocess the Data
# Create centroids for both layers to calculate point-to-point distances
tis_centroids = tis.copy()
tis_centroids['geometry'] = tis_centroids.centroid

defense_centroids = defense_institutions.copy()
defense_centroids['geometry'] = defense_centroids.centroid

# 3. Calculate Distance to the Nearest Defense Institution
# Extract coordinates for the KDTree
defense_coords = np.array(list(defense_centroids.geometry.apply(lambda geom: [geom.x, geom.y])))
ti_coords = np.array(list(tis_centroids.geometry.apply(lambda geom: [geom.x, geom.y])))

# Build a spatial index for the defense institutions for fast nearest-neighbor search
tree = KDTree(defense_coords)

# Query the tree for the single nearest defense institution to each TI
distances, indices = tree.query(ti_coords, k=1)
tis['dist_nearest_institution'] = distances

# 4. Calculate the Proximity Index
# Find the minimum and maximum distance in the dataset
min_distance = tis['dist_nearest_institution'].min()
max_distance = tis['dist_nearest_institution'].max()

print(f"Min distance: {min_distance:.2f} meters")
print(f"Max distance: {max_distance:.2f} meters")

# Calculate the normalized proximity index (closer = higher score)
# Formula: 1 - ((distance - min_distance) / (max_distance - min_distance))
tis['proximity_index'] = 1 - (
    (tis['dist_nearest_institution'] - min_distance) /
    (max_distance - min_distance)
)

# 5. Classify the Index for Interpretation (Optional)
# You can bin the continuous index into categories like "High", "Medium", "Low"
tis['proximity_category'] = pd.cut(tis['proximity_index'],
                                   bins=[-0.01, 0.33, 0.66, 1.01],
                                   labels=['Low', 'Medium', 'High'])

# 6. Inspect the Results
print(tis[['dist_nearest_institution', 'proximity_index', 'proximity_category']].describe())
print("\nFirst few rows:")
print(tis[['dist_nearest_institution', 'proximity_index', 'proximity_category']].head())

tis.to_file("data/tis_proximity_institutions.geojson")
# 7. Save the Results (Optional)
# tis.to_file("output_tis_with_proximity_index.gpkg", layer='tis_proximity', driver="GPKG")