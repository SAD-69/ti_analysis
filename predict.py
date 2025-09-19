import pandas as pd
import geopandas as gpd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================
# 1. Cenários simulados
# ======================================
def simulate_scenario(gdf, weight_dict, factor="mining_threat_area_ratio", multiplier=2):
    """
    Cria cenário simulado multiplicando um fator e recalculando índice de vulnerabilidade.
    gdf: GeoDataFrame com variáveis
    weight_dict: dicionário de pesos {coluna: peso}
    factor: variável a simular
    multiplier: fator de multiplicação (ex.: 2 = dobrar)
    """
    df = gdf.copy()
    
    # índice base
    df["vuln_base"] = sum(df[var] * w for var, w in weight_dict.items())
    
    # índice cenário
    df[f"{factor}_sim"] = df[factor] * multiplier
    weight_dict_sim = weight_dict.copy()
    df["vuln_scenario"] = sum(
        df[var if var != factor else f"{factor}_sim"] * w
        for var, w in weight_dict_sim.items()
    )
    
    df["delta"] = df["vuln_scenario"] - df["vuln_base"]
    return df


# ======================================
# 2. Análise de interações
# ======================================
def interaction_analysis(df, formula="vuln_base ~ mining_threat_area_ratio * silvicultura + soja"):
    """
    Testa interações entre fatores com OLS (pode trocar por spatial models depois).
    """
    model = smf.ols(formula=formula, data=df).fit()
    return model


# ======================================
# 3. Correlação social x ambiental
# ======================================
def correlation_analysis(df, social_cols, env_cols):
    """
    Plota correlação entre indicadores sociais e ambientais.
    """
    cols = social_cols + env_cols
    corr = df[cols].corr()
    
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlação entre fatores sociais e ambientais")
    plt.show()
    
    return corr


# ======================================
# 4. Análise temporal
# ======================================
def temporal_analysis(df_temporal, ti_col="TI", var_col="area_silvicultura", ano_col="ano", ti_escolhida=None):
    """
    Plota evolução temporal de um indicador de uso do solo.
    df_temporal: DataFrame longo com colunas [TI, ano, variável]
    ti_escolhida: se None, plota todas as TIs agregadas
    """
    if ti_escolhida:
        subset = df_temporal[df_temporal[ti_col] == ti_escolhida]
        plt.plot(subset[ano_col], subset[var_col], marker="o")
        plt.title(f"Evolução de {var_col} - {ti_escolhida}")
    else:
        grouped = df_temporal.groupby(ano_col)[var_col].mean().reset_index()
        plt.plot(grouped[ano_col], grouped[var_col], marker="o")
        plt.title(f"Evolução média de {var_col} em todas as TIs")
    
    plt.xlabel("Ano")
    plt.ylabel(var_col)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    from models.gpkg import GeoPackage
    gpkg = GeoPackage()
    gdf = gpkg.read_layer('vindex_1985_2023_entrega')
    # biomas = gpkg.read_layer('biomas_rs_faixa_trans_final')

    gdf.geometry = gdf.geometry.buffer(gdf.dist_buf)
    weights = {
        "mining_threat_area_ratio_ex": 0.25,
        "forest_formation_ex": 0.25,
        "soybean_ex": 0.25,
        "agropec_ex": 0.25
    }

    df_scen = simulate_scenario(gdf, weights, factor="mining_threat_area_ratio_ex", multiplier=2)
    df_scen[["pol_id", "vuln_base", "vuln_scenario", "delta"]].head()
    model = interaction_analysis(df_scen, "vuln_base ~ mining_threat_area_ratio_ex * forest_formation_ex + soybean_ex + agropec_ex")
    print(model.summary())
    corr = correlation_analysis(
    df_scen,
    social_cols=["txalfabeti_imputed", "rep_index"],
    env_cols=["mining_threat_area_ratio_ex", "forest_formation_ex", "soybean_ex", "agropec_ex"]
)   
    # temporal_analysis(df_temporal, ti_escolhida="TI_X")  # Para uma TI específica
    # temporal_analysis(df_temporal)  # Média geral


