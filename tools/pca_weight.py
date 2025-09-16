import matplotlib.pyplot as plt
import seaborn as sns

from geopandas import GeoDataFrame, read_file
from pandas import DataFrame, Series
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def pca_measure_weight(
        main_gdf: GeoDataFrame, 
        weight_dict: dict[str, dict], 
        component: str,
        first_comp: bool = False,
        n_components: int = None):
    gdf = main_gdf.copy()
    indicators = list(weight_dict[component].keys())
    X = gdf[indicators].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_components:
        pca = PCA(n_components=min(n_components, len(indicators)))
    pca = PCA()
    pca.fit(X_scaled)

    loadings = DataFrame(
        pca.components_.T,
        index=indicators,
        columns=[f"PC{i+1}" for i in range(len(indicators))]
    )
    
    #PC1
    weights_pc1: Series = abs(loadings["PC1"])
    
    # Variance contribution method
    explained_var = pca.explained_variance_ratio_
    weights_all = (abs(loadings.values) * explained_var).sum(axis=1)
    weights_all = weights_all / weights_all.sum()
    weights_all = Series(weights_all, index=indicators)
    if first_comp:
        for key, value in weights_pc1.items():
            weight_dict[component][key] = value
        return weight_dict
    else:
        for key, value in weights_all.items():
            weight_dict[component][key] = value
        return weight_dict

def plot_pca_weights(weights: dict, component: str):
    """Plot PCA-based weights for a given component"""
    data = weights[component]
    indicators = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(8, 5))
    plt.bar(indicators, values, color="steelblue", edgecolor="black")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Weight")
    plt.title(f"PCA Weights for {component}")
    plt.tight_layout()
    plt.show()

def plot_all_pca_weights(weights: dict, bioma: str = 'RS'):
    """
    Plot PCA-based weights for all components in one grouped bar chart.
    """
    # Convert dict -> DataFrame
    df = DataFrame(weights).fillna(0)

    # Plot grouped bars
    ax = df.plot(
        kind="bar",
        figsize=(10, 6),
        width=0.75,
        edgecolor="black"
    )

    plt.title(f"PCA Weights for All Components: {bioma}")
    plt.ylabel("Weight")
    plt.xlabel("Indicators")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Component")
    plt.tight_layout()
    plt.show()



def plot_pca_heatmap(weights: dict, bioma: str = 'RS'):
    """
    Plot PCA-based weights as a heatmap (indicators × components).
    """
    # Convert dict -> DataFrame (indicators as rows, components as columns)
    df = DataFrame(weights).fillna(0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df,
        annot=True, fmt=".3f", cmap="Oranges", 
        cbar_kws={'label': 'Weight'}
    )
    plt.title(f"Pesos PCA: {bioma}")
    plt.ylabel("Indicadores")
    plt.xlabel("Componentes")
    plt.tight_layout()
    plt.show()

def plot_explained_variance(gdf: GeoDataFrame, indicators: list, component: str):
    X = gdf[indicators].copy()
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(indicators)+1), pca.explained_variance_ratio_)
    plt.plot(range(1, len(indicators)+1), 
             pca.explained_variance_ratio_.cumsum(), 
             'ro-')
    plt.xlabel('Componentes Principais')
    plt.ylabel('Variância Explicada')
    plt.title(f'Variância Explicada - {component}')
    plt.grid(True)
    plt.show()

def plot_scree(gdf: GeoDataFrame, indicators: list, component: str):
    X = gdf[indicators].copy()
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(indicators)+1), pca.explained_variance_, 'bo-')
    plt.xlabel('Componentes Principais')
    plt.ylabel('Autovalores')
    plt.title(f'Scree Plot - {component}')
    plt.grid(True)
    plt.axhline(y=1, color='r', linestyle='--')  # Critério de Kaiser
    plt.show()

def pca_measure_weight_flexible(
        main_gdf: GeoDataFrame, 
        weight_dict: dict[str, dict], 
        component: str,
        method: str = "variance",  # "variance", "fixed", "first"
        n_components: int = None,
        variance_threshold: float = 0.8):
    """
    method: "variance" (threshold), "fixed" (n componentes), "first" (apenas PC1)
    """
    gdf = main_gdf.copy()
    indicators = list(weight_dict[component].keys())
    X = gdf[indicators].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Configurar PCA baseado no método
    if method == "variance":
        pca = PCA(n_components=variance_threshold)
    elif method == "fixed" and n_components is not None:
        pca = PCA(n_components=min(n_components, len(indicators)))
    else:
        pca = PCA()
    
    pca.fit(X_scaled)

    loadings = DataFrame(
        pca.components_.T,
        index=indicators,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    
    explained_var = pca.explained_variance_ratio_
    
    if method == "first":
        weights = abs(loadings["PC1"])
    else:
        weights = (abs(loadings.values) * explained_var).sum(axis=1)
        weights = weights / weights.sum()
    
    weights = Series(weights, index=indicators)
    
    for key, value in weights.items():
        weight_dict[component][key] = value
    
    print(f"{component}: {pca.n_components_} componentes, {sum(explained_var):.2%} variância")
    
    return weight_dict


def analyze_bioma_weights(main_gdf: GeoDataFrame, weight_dict: dict, bioma_name: str):
    """
    Analisa pesos PCA para um bioma específico
    """
    # Filtrar por bioma
    gdf_bioma = main_gdf[main_gdf['bioma'] == bioma_name].copy()
    
    results = {}
    
    for component in weight_dict.keys():
        # Calcular pesos com variância acumulada (80%)
        weights_bioma = pca_measure_weight_flexible(
            gdf_bioma, 
            weight_dict.copy(), 
            component,
            method="variance", 
            variance_threshold=0.8
        )
        
        results[component] = weights_bioma[component]
        
        # Gerar visualizações
        plot_pca_weights(weights_bioma, component)
        plot_pca_heatmap(weights_bioma, bioma_name)
    
    return results

def plot_bioma_comparison(weights_pampa: dict, weights_mata: dict):
    """
    Heatmap comparativo entre os dois biomas
    """
    # Criar DataFrame comparativo
    compare_data = {}
    
    for component in weights_pampa.keys():
        pampa_vals: dict = weights_pampa[component]
        mata_vals: dict = weights_mata[component]
        
        # Calcular diferenças
        differences = {}
        for indicator in pampa_vals.keys():
            diff = pampa_vals[indicator] - mata_vals.get(indicator, 0)
            differences[indicator] = diff
        
        compare_data[component] = differences
    
    # Plot heatmap de diferenças
    df_compare = DataFrame(compare_data).fillna(0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_compare, annot=True, fmt=".3f", cmap="Oranges", 
                center=0, cbar_kws={'label': 'Diferença de Peso (Pampa - Mata)'})
    plt.title("Diferenças de Pesos entre Biomas")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    from pprint import pprint
    weights = {
        'ex_score': {
            'forest_plantation_ex': 1/7,
            'soybean_ex': 1/7,
            'urban_area_ex': 1/7,
            'degeneration_%_ex': 1/7,
            'agropec_ex': 1/7,
            'mining_threat_density_ex': 1/7,
        },
        'se_score': {
            'forest_plantation_se': 1/7,
            'soybean_se': 1/7,
            'urban_area_se': 1/7,
            'degeneration_%_se': 1/7,
            'agropec_se': 1/7,
            'mining_threat_density_se': 1/7,
        },
        'ca_score': {
            'forest_cover_2023_%_ex': 1/5,
            'regeneration_%_ex': 1/5,
            'txalfabeti_imputed': 1/5,
            'proximity_index': 1/5,
            'status_fundiario': 1/5
        }
    }
    
    gdf = read_file(r'data\vindex_2010_2023_v8.geojson')
    pca_measure_weight(gdf, weights, 'ex_score')
    # pprint(weights['ex_score'])
    weights = pca_measure_weight(gdf, weights, 'se_score')
    # pprint(weights['se_score'])
    weights = pca_measure_weight(gdf, weights, 'ca_score')
    pprint(weights)