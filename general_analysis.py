import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import ols
from models.gpkg import GeoPackage

gpkg = GeoPackage()
gdf = gpkg.read_layer('vindex_1985_2023_godmode')
# 1. Carregar os dados
# Suponha que seu arquivo seja um Shapefile ou GeoJSON com as TIs e uma coluna vindex_col
# Substitua 'seu_arquivo.shp' pelo caminho do seu arquivo
# gdf = gpd.read_file('seu_arquivo.shp')

# gdf = gdf[gdf['bioma'] != 'Pampa']
# Criar uma coluna com as coordenadas do centróide de cada TI
gdf['longitude'] = gdf.geometry.centroid.x
gdf['latitude'] = gdf.geometry.centroid.y

vindex_col = 'vulnerability_index'
# 2. Análise Visual
# Mapa de calor: plotar as TIs coloridas pelo índice de vulnerabilidade
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
gdf.plot(column=vindex_col, ax=ax, legend=True,
         cmap='Reds', markersize=50)
# Adicionar rótulos para cada TI (opcional)
for idx, row in gdf.iterrows():
    ax.annotate(text=row['nome_ti'], xy=(row['longitude'], row['latitude']),
                xytext=(3, 3), textcoords="offset points", fontsize=8)
plt.title("Mapa de Vulnerabilidade das Terras Indígenas")
plt.tight_layout()
plt.show()

# Boxplot: comparar Oeste vs Leste
# Primeiro, precisamos definir o que é Oeste e Leste. Vamos usar a mediana da longitude como corte.
longitude_median = gdf['longitude'].median()
gdf['regiao'] = gdf['longitude'].apply(lambda x: 'Oeste' if x < longitude_median else 'Leste')

plt.figure(figsize=(8, 6))
sns.boxplot(x='regiao', y=vindex_col, data=gdf)
plt.title('Distribuição do Índice de Vulnerabilidade por Região')
plt.xlabel('Região')
plt.ylabel('Índice de Vulnerabilidade')
plt.show()

# 3. Estatísticas Descritivas e Teste de Hipótese
# Estatísticas descritivas por grupo
print(gdf.groupby('regiao')[vindex_col].describe())

# Teste de Mann-Whitney U (não paramétrico) para comparar os dois grupos
oeste = gdf[gdf['regiao'] == 'Oeste'][vindex_col]
leste = gdf[gdf['regiao'] == 'Leste'][vindex_col]
stat, p_value = mannwhitneyu(oeste, leste, alternative='two-sided')
print(f"\nTeste de Mann-Whitney U:\nEstatística= {stat:.3f}, p-value= {p_value:.3f}")

# Interpretação do p-value
alpha = 0.05
if p_value < alpha:
    print("Há uma diferença estatisticamente significativa entre as regiões.")
else:
    print("Não há diferença estatisticamente significativa entre as regiões.")

# 4. Análise de Correlação
# Correlação entre longitude e índice de vulnerabilidade
corr_pearson, p_pearson = pearsonr(gdf['longitude'], gdf[vindex_col])
corr_spearman, p_spearman = spearmanr(gdf['longitude'], gdf[vindex_col])

print(f"\nCorrelação de Pearson: r= {corr_pearson:.3f}, p-value= {p_pearson:.3f}")
print(f"Correlação de Spearman: r= {corr_spearman:.3f}, p-value= {p_spearman:.3f}")

# 5. Modelagem de Regressão
# a) Regressão Linear Simples: Índice ~ Longitude
X = sm.add_constant(gdf['longitude'])  # adiciona intercepto
modelo_simples = sm.OLS(gdf[vindex_col], X).fit()
print("\n--- Regressão Linear Simples ---")
print(modelo_simples.summary())

# b) Regressão Múltipla: Índice ~ Longitude + Área + População
# Verifique se suas colunas de área e população existem no dataset
# Supondo que se chamam 'area' e 'populacao'
# Se não existirem, pule esta parte ou use outras variáveis

try:
    # Preparar matriz de variáveis independentes
    X_multiplo = gdf[['longitude']]  # substitua pelos nomes corretos
    X_multiplo = sm.add_constant(X_multiplo)
    
    modelo_multiplo = sm.OLS(gdf[vindex_col], X_multiplo).fit()
    print("\n--- Regressão Múltipla ---")
    print(modelo_multiplo.summary())
    
except KeyError as e:
    print(f"\nVariável não encontrada: {e}. Verifique os nomes das colunas para regressão múltipla.")

# 6. Visualização da Regressão Simples
# Gráfico de dispersão com linha de regressão
plt.figure(figsize=(10, 6))
sns.regplot(x='longitude', y=vindex_col, data=gdf, 
            line_kws={"color": "red"})
plt.title('Relação entre Longitude e Índice de Vulnerabilidade')
plt.xlabel('Longitude')
plt.ylabel('Índice de Vulnerabilidade')
plt.show()

# 7. Análise de Resíduos (para verificar suposições do modelo)
# Apenas para o modelo simples
residuos = modelo_simples.resid
valores_previstos = modelo_simples.predict(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Resíduos vs Valores Preditos
axes[0].scatter(valores_previstos, residuos, alpha=0.7)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Valores Preditos')
axes[0].set_ylabel('Resíduos')
axes[0].set_title('Resíduos vs Preditos')

# QQ-Plot para normalidade dos resíduos
sm.qqplot(residuos, line='s', ax=axes[1])
axes[1].set_title('QQ-Plot dos Resíduos')

plt.tight_layout()
plt.show()