import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from libpysal.weights import DistanceBand, KNN
from esda.moran import Moran
from spreg import OLS, ML_Lag, ML_Error

from models.gpkg import GeoPackage

# Configuração dos plots
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

gpkg = GeoPackage()
gdf = gpkg.read_layer('vindex_1985_2023_godmode')
gdf.geometry = gdf.geometry.buffer(gdf.dist_buf)
gdf['soybean'] = gdf['soybean_ex'] + gdf['soybean_se']
gdf['agropec'] = gdf['agropec_ex'] + gdf['agropec_se']
gdf['degeneration_%'] = gdf['degeneration_%_ex'] + gdf['degeneration_%_se']
gdf['mining'] = gdf['mining_threat_area_ratio_se'] + gdf['mining_threat_area_ratio_ex']
# gdf.dropna(subset=['TxAlfabetI'], inplace=True)
y_col = 'vulnerability_index'
cols = ['regeneration_%_total']
# Variáveis dependente (y) e independentes (X)
y = gdf[y_col].values.reshape(-1,1)
X = gdf[cols].values


# 2. Criar matriz de pesos espaciais baseada em distância (ex.: 150 km)
# w = DistanceBand.from_dataframe(gdf, threshold=40000, silence_warnings=True)
w = KNN.from_dataframe(gdf, k=6)
w.transform = "r"   # normalizar pesos

# 3. Regressão OLS (baseline)
ols = OLS(y, X, name_y=y_col,
          name_x=cols)
print("\n--- OLS ---")
print(ols.summary)

# Plot OLS - Valores Observados vs Previstos
axes[0].scatter(y, ols.predy, alpha=0.7, color='blue', edgecolor='black')
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Linha de perfeita previsão
axes[0].set_xlabel('Valores Observados')
axes[0].set_ylabel('Valores Previstos')
axes[0].set_title(f'OLS - R² = {ols.r2:.3f}')
axes[0].grid(True, alpha=0.3)

# 4. Testar autocorrelação espacial nos resíduos da OLS
moran_res = Moran(ols.u, w)
print("\nMoran's I dos resíduos OLS:")
print(f"I = {moran_res.I:.4f}, p-value = {moran_res.p_sim:.4f}")

# 5. Se houver autocorrelação -> rodar modelos espaciais
if moran_res.p_sim < 0.05:
    print("\n--- Spatial Lag Model (SAR) ---")
    sar = ML_Lag(y, X, w=w, name_y=y_col,
                 name_x=cols)
    print(sar.summary)
    
    # Plot SAR - Valores Observados vs Previstos
    axes[1].scatter(y, sar.predy, alpha=0.7, color='green', edgecolor='black')
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[1].set_xlabel('Valores Observados')
    axes[1].set_ylabel('Valores Previstos')
    axes[1].set_title(f'Spatial Lag (SAR) - Pseudo R² = {sar.pr2:.3f}')
    axes[1].grid(True, alpha=0.3)

    print("\n--- Spatial Error Model (SEM) ---")
    sem = ML_Error(y, X, w=w, name_y=y_col,
                   name_x=cols)
    print(sem.summary)
    
    # Plot SEM - Valores Observados vs Previstos
    axes[2].scatter(y, sem.predy, alpha=0.7, color='orange', edgecolor='black')
    axes[2].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[2].set_xlabel('Valores Observados')
    axes[2].set_ylabel('Valores Previstos')
    axes[2].set_title(f'Spatial Error (SEM) - Pseudo R² = {sem.pr2:.3f}')
    axes[2].grid(True, alpha=0.3)

    # Comparar modelos pelo AIC
    print("\nComparação de modelos:")
    print(f"AIC OLS: {ols.aic:.2f}")
    print(f"AIC SAR: {sar.aic:.2f}")
    print(f"AIC SEM: {sem.aic:.2f}")

    # Plot comparativo de AIC
    models = ['OLS', 'SAR', 'SEM']
    aic_values = [ols.aic, sar.aic, sem.aic]
    
    bars = axes[3].bar(models, aic_values, color=['blue', 'green', 'orange'])
    axes[3].set_ylabel('Valor AIC')
    axes[3].set_title('Comparação de AIC (menor é melhor)')
    axes[3].grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, value in zip(bars, aic_values):
        axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.2f}', ha='center', va='bottom')

    if sem.aic < sar.aic and sem.aic < ols.aic:
        melhor_modelo = "Spatial Error (SEM)"
        print("👉 Melhor modelo: Spatial Error (SEM)")
    elif sar.aic < sem.aic and sar.aic < ols.aic:
        melhor_modelo = "Spatial Lag (SAR)"
        print("👉 Melhor modelo: Spatial Lag (SAR)")
    else:
        melhor_modelo = "OLS"
        print("👉 Melhor modelo: OLS (sem autocorrelação significativa nos resíduos)")
        
    # Adicionar título geral
    fig.suptitle(f'Comparação de Modelos de Regressão - Melhor: {melhor_modelo}', 
                fontsize=16, fontweight='bold')
    
else:
    print("👉 Sem autocorrelação espacial nos resíduos. OLS é suficiente.")
    # Esconder os eixos não utilizados
    for ax in axes[1:]:
        ax.set_visible(False)

plt.tight_layout()
plt.show()