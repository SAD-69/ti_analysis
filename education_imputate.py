import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree

from models.gpkg import GeoPackage

gpkg = GeoPackage()
# --- 1. Carregar dados ---
tis = gpkg.read_layer('ti_ac_preliminar_tx_alfabetizacao')   # deve ter coluna 'educ_pct'
escolas = gpkg.read_layer('inep_dados').dropna() # deve ter coluna 'IDEB'

# Converte dados negativos para nulo
tis['TxAlfabetI'] = tis['TxAlfabetI'].mask(tis['TxAlfabetI'] < 0)

# Criar centroides das TIs
tis["geometry_centroid"] = tis.centroid
tis_centroids = tis.set_geometry("geometry_centroid")

# --- 2. Calcular distâncias TI–Escolas ---
# Extrair coordenadas
ti_coords = np.array([[geom.x, geom.y] for geom in tis_centroids.geometry])
escola_coords = np.array([[geom.x, geom.y] for geom in escolas.geometry])

tree = KDTree(escola_coords)

# Escola mais próxima (distância + IDEB)
dist, idx = tree.query(ti_coords, k=1)
tis["dist_nearest_school"] = dist
tis["ideb_nearest"] = escolas.iloc[idx]["IDEB_2023"].values

# IDW (ex.: com 5 vizinhos mais próximos)
distances, indices = tree.query(ti_coords, k=5)
# indices = [i for sublist in indices for i in sublist]


vals = escolas["IDEB_2023"].to_numpy()        # (n_escolas,)
ideb_vals = np.take(vals, indices)

weights = 1 / (distances + 1e-6)
weights /= weights.sum(axis=1, keepdims=True)
idw_vals = np.sum(ideb_vals * weights, axis=1)
tis["ideb_idw"] = idw_vals

# --- 3. Verificar correlação nos dados completos ---
print(tis[["TxAlfabetI", "ideb_nearest", "ideb_idw"]].corr())

# --- 4. Imputar faltantes ---
features = ["ideb_nearest", "ideb_idw", "dist_nearest_school"]  # pode adicionar pop, renda, etc.
scaler = StandardScaler()
df = tis[["TxAlfabetI"] + features]
# df["TxAlfabetI"].fillna(df["TxAlfabetI"].mean(), inplace=True)

imp = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=100, random_state=0),
    max_iter=10, random_state=42, sample_posterior=False
)
df_imputed = imp.fit_transform(df)

# Substituir na tabela
tis_imputed = tis.copy()
tis_imputed["educ_pct_imputed"] = df_imputed[:, 0]
print(tis_imputed.head())
print(tis_imputed.isna().sum())
print(tis_imputed.describe())

# --- 5. Validar (opcional: leave-one-out) ---
# Retirar algumas TIs conhecidas, imputar, calcular erro...
