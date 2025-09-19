import geopandas as gpd
import libpysal
from spreg import OLS, ML_Lag, ML_Error, diagnostics
# from spreg.diagnostics_sp import LM_Lag, LM_Error, LM_robust
from spreg.diagnostics_sp import LMtests
import numpy as np
import matplotlib.pyplot as plt
from esda.moran import Moran
from models.gpkg import GeoPackage
from libpysal.weights import KNN

gpkg = GeoPackage()
gdf = gpkg.read_layer('vindex_1985_2023_entrega')
gdf.geometry = gdf.geometry.buffer(gdf.dist_buf)

gdf['soybean'] = gdf['soybean_ex'] + gdf['soybean_se']

w = KNN.from_dataframe(gdf, k=6)
w.transform = 'r'

y = gdf['vulnerability_index'].values.reshape((-1,1))
X = gdf['soybean'].values

# OLS
ols = OLS(y, X, name_y="Vulnerabilidade", name_x=["Soja"])
print(ols.summary)

lm_test = LMtests(ols, w)

lm_test.rlme

print(f"LM Error: {lm_test.lme[0]:.3f}  p_value={lm_test.lme[1]:.3f}")
print(f"LM Robust Error: {lm_test.rlme[0]:.3f}  p_value={lm_test.rlme[1]:.3f}\n")
print(f"LM Lag: {lm_test.lml[0]:.3f}  p_value={lm_test.lml[1]:.3f}")
print(f"LM Robust Lag: {lm_test.rlml[0]:.3f}  p_value={lm_test.rlml[1]:.3f}\n")
print(f"LM sp durbin: {lm_test.lmspdurbin[0]:.3f}  p_value={lm_test.lmspdurbin[1]:.3f}")