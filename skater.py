import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from libpysal.weights import KNN, DistanceBand
from esda.moran import Moran_Local, Moran
from spopt.region import Skater
from models.gpkg import GeoPackage
import contextily as ctx
from matplotlib.patches import Patch
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from splot.esda import moran_scatterplot



gpkg = GeoPackage()

gdf = gpkg.read_layer('vindex_1985_2023_entrega')
biomas = gpkg.read_layer('biomas_rs_faixa_trans_final')

gdf.geometry = gdf.geometry.buffer(gdf.dist_buf)


# -------------------------------
# 1. Carregar dados
# -------------------------------
# Sua camada de TIs, já com uma coluna "vulnerabilidade"
# gdf = gpd.read_file("terras_indigenas.shp")

# Se você já tem a coluna do bioma (Pampa, Transição, Mata Atlântica)
# Se não, pode cruzar com shapefile de biomas (IBGE)
# print(gdf.columns)
if gdf.crs != "EPSG:3857":
    gdf = gdf.to_crs(epsg=3857)
    biomas.to_crs(3857, inplace=True)


# -------------------------------
# 2. Rodar LISA (Moran Local)
# -------------------------------
# Cria matriz de vizinhança por k-vizinhos
w = KNN.from_dataframe(gdf, k=6)
# w = DistanceBand.from_dataframe(gdf, 40000)
w.transform = "r"

# Roda Moran Local no índice de vulnerabilidade
gdf['soybean'] = gdf['soybean_ex'] + gdf['soybean_se']
y = gdf["soybean"].values
moran_loc = Moran_Local(y, w)
# moran_I = Moran(y, w, transformation='r')
moran_loc.plot_combination(gdf, "soybean")


# moran_I.plot_scatter(p=moran_I.p_sim)
# moran_scatterplot(moran_loc, p=moran_loc.p_sim)
# moran_I.p()

# Adiciona resultados no GeoDataFrame
gdf["lisa_cluster"] = moran_loc.q   # Quadrante (HH, LL, HL, LH)
gdf["lisa_sig"] = moran_loc.p_sim   # p-valor

# -------------------------------
# 3. Criar sub-regiões com SKATER
# -------------------------------
# Variáveis usadas para agrupar (pode incluir mais além de vulnerabilidade)
# attrs = gdf[["vulnerability_index"]].values.tolist()

# SKATER precisa de grafo de vizinhança
graph = w

# Define número de clusters que você quer gerar por bioma
n_clusters = 4

model = Skater(gdf, graph, ['soybean'], n_clusters)
model.solve()

# Resultados dos clusters
gdf["subregiao"] = model.labels_

# -------------------------------
# 4. Plotar mapas
# -------------------------------

lisa_colors = {
    1: "red",      # High-High
    2: "lightblue",# Low-High
    3: "blue",     # Low-Low
    4: "pink"      # High-Low
}
lisa_labels = {
    1: "HH", 
    2: "LH", 
    3: "LL", 
    4: "HL"
}

gdf['lisa_class'] = gdf['lisa_cluster'].map(lisa_labels)

lisa_colors_final = {0: "lightgrey", **lisa_colors}
gdf["lisa_plot"] = np.where(gdf["lisa_sig"], gdf["lisa_cluster"], 0)


fig, axes = plt.subplots(1, 2, figsize=(12, 6))

biomas.plot(ax=axes[0], color='green', edgecolor='white', linewidth=0.5)

gdf.plot(column="lisa_plot",
         ax=axes[0],
         color=gdf["lisa_plot"].map(lisa_colors_final),
         edgecolor="black", linewidth=0.3)



ctx.add_basemap(axes[0], source=ctx.providers.OpenStreetMap.Mapnik)
# LISA
# gdf.plot(column="lisa_cluster", categorical=True, legend=True, ax=axes[0])
legend_elements = [
    Patch(facecolor="red", label="HH (Alta-Alta)"),
    Patch(facecolor="blue", label="LL (Baixa-Baixa)"),
    Patch(facecolor="pink", label="HL (Alta-Baixa)"),
    Patch(facecolor="lightblue", label="LH (Baixa-Alta)"),
    Patch(facecolor="lightgrey", label="Não significativo")
]
axes[0].set_title("Clusters LISA (Moran Local, p<0.05)", fontsize=14)
# SKATER
# Plot SKATER com paleta discreta
biomas.plot(ax=axes[1], color='#0c2903', edgecolor='white', linewidth=0.5)


gdf.plot(column="subregiao",
         cmap="Set2",   # pode testar: Set1, tab10, Paired
         legend=True,
         ax=axes[1], edgecolor="black", linewidth=0.3)



axes[1].set_title("Sub-regiões (SKATER)")
ctx.add_basemap(axes[1], source=ctx.providers.OpenStreetMap.Mapnik)
plt.tight_layout()
plt.show()

y = gdf[['vulnerability_index']].values.reshape((-1,1))
X = gdf[['soybean']].values  # pode incluir mais colunas
coords = np.array([(geom.x, geom.y) for geom in gdf.geometry.centroid])

# Seleção automática de bandwidth
bw = Sel_BW(coords, y, X).search()

# Ajusta modelo GWR
gwr_model = GWR(coords, y, X, bw)
gwr_results = gwr_model.fit()

# -------------------------------
# 6. Guardar coeficientes no GeoDataFrame
# -------------------------------
gdf['gwr_intercept'] = gwr_results.params[:,0]
gdf['gwr_beta'] = gwr_results.params[:,1]   # se tiver 1 variável explicativa
gdf['gwr_R2'] = gwr_results.localR2

# -------------------------------
# 7. Plots do GWR
# -------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

biomas.plot(ax=axes[0], color='0c2903', edgecolor='white', linewidth=0.5)

# Intercepto
gdf.plot(column='gwr_intercept', cmap='coolwarm', legend=True,
         ax=axes[0], edgecolor='black', linewidth=0.3)
axes[0].set_title("GWR Intercepto")
ctx.add_basemap(axes[0], source=ctx.providers.OpenStreetMap.Mapnik)

biomas.plot(ax=axes[1], color='#0c2903', edgecolor='white', linewidth=0.5)
# Coeficiente da variável explicativa
gdf.plot(column='gwr_beta', cmap='coolwarm', legend=True,
         ax=axes[1], edgecolor='black', linewidth=0.3)
axes[1].set_title("Coeficiente GWR (var_explicativa)")
ctx.add_basemap(axes[1], source=ctx.providers.OpenStreetMap.Mapnik)

biomas.plot(ax=axes[2], color='#0c2903', edgecolor='white', linewidth=0.5)

# Local R²
gdf.plot(column='gwr_R2', cmap='coolwarm', legend=True,
         ax=axes[2], edgecolor='black', linewidth=0.3)
axes[2].set_title("Local R² (qualidade do ajuste)")
ctx.add_basemap(axes[2], source=ctx.providers.OpenStreetMap.Mapnik)

plt.tight_layout()
plt.show()

# -------------------------------
# 8. Salvar camada com resultados GWR
# -------------------------------
gpkg.save_layer(gdf, 'lisa_skater_ex_gwr')

# gpkg.save_layer(gdf, 'lisa_skater_ex')

# print(gdf[['vulnerability_index', 'lisa_cluster', 'lisa_sig', 'subregiao']])