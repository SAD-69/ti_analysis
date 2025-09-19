import geopandas as gpd
import numpy as np
from libpysal.weights import DistanceBand
from esda.moran import Moran
from spreg import OLS, ML_Lag, ML_Error

from models.gpkg import GeoPackage

gpkg = GeoPackage()
gdf = gpkg.read_layer('vindex_1985_2023_entrega')

# gdf.dropna(subset=['TxAlfabetI'], inplace=True)

# Variáveis dependente (y) e independentes (X)
cols = ['ca_score', 'se_score', 'ex_score']
y = gdf["vulnerability_index"].values.reshape(-1,1)
X = gdf[cols].values

# 2. Criar matriz de pesos espaciais baseada em distância (ex.: 150 km)
w = DistanceBand.from_dataframe(gdf, threshold=40000, silence_warnings=True)
w.transform = "r"   # normalizar pesos

# 3. Regressão OLS (baseline)
ols = OLS(y, X, name_y="Taxa Alfabetizacao",
          name_x=["ideb ponderado", "ideb mais proximo", "distancia do mais proximo"])
print("\n--- OLS ---")
print(ols.summary)

# 4. Testar autocorrelação espacial nos resíduos da OLS
moran_res = Moran(ols.u, w)
print("\nMoran's I dos resíduos OLS:")
print(f"I = {moran_res.I:.4f}, p-value = {moran_res.p_sim:.4f}")

# 5. Se houver autocorrelação -> rodar modelos espaciais
if moran_res.p_sim < 0.05:
    print("\n--- Spatial Lag Model (SAR) ---")
    sar = ML_Lag(y, X, w=w, name_y="Vulnerabilidade",
                 name_x=cols)
    print(sar.summary)

    print("\n--- Spatial Error Model (SEM) ---")
    sem = ML_Error(y, X, w=w, name_y="Vulnerabilidade",
                   name_x=cols)
    print(sem.summary)

    # Comparar modelos pelo AIC
    print("\nComparação de modelos:")
    print(f"AIC OLS: {ols.aic:.2f}")
    print(f"AIC SAR: {sar.aic:.2f}")
    print(f"AIC SEM: {sem.aic:.2f}")

    if sar.aic < sem.aic and sar.aic < ols.aic:
        print("👉 Melhor modelo: Spatial Lag (SAR)")
    elif sem.aic < sar.aic and sem.aic < ols.aic:
        print("👉 Melhor modelo: Spatial Error (SEM)")
    else:
        print("👉 Melhor modelo: OLS (sem autocorrelação significativa nos resíduos)")
else:
    print("👉 Sem autocorrelação espacial nos resíduos. OLS é suficiente.")
