import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

import libpysal
from esda.moran import Moran
from models.gpkg import GeoPackage
# =============================
# 1️⃣ Prepare your GeoDataFrame
# =============================
# Assuming you have 'gdf' with geometry, y column, and X columns

gpkg = GeoPackage()
gdf = gpkg.read_layer('vindex_1985_2023_entrega')
# gdf.fillna(0, inplace=True)
coords = np.array(list(zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y)))
gdf.geometry = gdf.geometry.buffer(gdf.dist_buf)
y = gdf['vulnerability_index'].values.reshape((-1, 1))
X_cols = ['mining_threat_area_ratio_ex']  # you can add more
X = gdf[X_cols].values

# Optional: standardize independent variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =============================
# 2️⃣ Fit GWR
# =============================
bw = Sel_BW(coords, y, X).search()
print("Optimal bandwidth:", bw)

gwr_model = GWR(coords, y, X, bw).fit()

# Add coefficients to GeoDataFrame
gdf['gwr_intercept'] = gwr_model.params[:, 0]
for i, col in enumerate(X_cols):
    gdf[f'gwr_coef_{col}'] = gwr_model.params[:, i+1]

# Residuals
gdf['gwr_residuals'] = gwr_model.resid_response.flatten()

# =============================
# 3️⃣ Reproject for contextily (Web Mercator)
# =============================
gdf = gdf.to_crs(epsg=3857)

# =============================
# 4️⃣ Plot GWR coefficients & residuals
# =============================
def plot_gwr_map(gdf, column, title, cmap='coolwarm'):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    gdf.plot(column=column, cmap=cmap, legend=True, ax=ax)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16)
    plt.show()

# Plot intercept
plot_gwr_map(gdf, 'gwr_intercept', 'GWR Local Intercept')

# Plot coefficients
for col in X_cols:
    plot_gwr_map(gdf, f'gwr_coef_{col}', f'GWR Coefficient: {col}')

# Plot residuals
plot_gwr_map(gdf, 'gwr_residuals', 'GWR Residuals', cmap='bwr')

# =============================
# 5️⃣ Moran's I for residuals
# =============================
# Use Queen contiguity or KNN
w = libpysal.weights.KNN.from_dataframe(gdf, k=5)
w.transform = 'R'  # row-standardized

moran_res = Moran(gdf['gwr_residuals'], w)
print("Moran's I:", moran_res.I)
print("p-value:", moran_res.p_sim)

# Optional: plot Moran scatter
import seaborn as sns
from splot.esda import moran_scatterplot

fig, ax = moran_scatterplot(moran_res)
plt.show()
