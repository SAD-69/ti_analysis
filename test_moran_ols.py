import numpy as np
import geopandas as gpd
from libpysal.weights import DistanceBand, KNN
from esda.moran import Moran
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from models.gpkg import GeoPackage

gpkg = GeoPackage()
gdf = gpkg.read_layer('vindex_1985_2023_good')
gdf.geometry = gdf.geometry.centroid

coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T
y = gdf['vulnerability_index'].values

# 2) k-NN baseline (k=4)
k = 4
nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
distances, indices = nbrs.kneighbors(coords)
# distances[:, -1] = distance to k-th neighbor for each observation
kth_distances = distances[:, -1]

# candidate thresholds (percentiles of kth-distance)
cand = np.percentile(kth_distances, np.linspace(25, 100, 20))

moran_I = []
moran_p = []
pct_connected = []

for d in cand:
    w = DistanceBand.from_array(coords, threshold=d, binary=True, silence_warnings=True)
    n_with_neighbors = sum(len(nb) > 0 for nb in w.neighbors.values())
    pct_connected.append(n_with_neighbors / len(coords))
    try:
        m = Moran(y, w)
        moran_I.append(m.I)
        moran_p.append(m.p_norm)
    except Exception:
        moran_I.append(np.nan)
        moran_p.append(np.nan)

# 3) plot results
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(cand, moran_I, '-o')
plt.xlabel('Distance threshold')
plt.ylabel("Global Moran's I")
plt.title("Moran's I vs distance")

plt.subplot(1,2,2)
plt.plot(cand, pct_connected, '-o')
plt.xlabel('Distance threshold')
plt.ylabel('% connected')
plt.axhline(1.0, color='grey', linestyle='--')
plt.title("Percent connected vs distance")
plt.tight_layout()
plt.show()