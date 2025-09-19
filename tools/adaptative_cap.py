import numpy as np

from scipy.spatial import KDTree
from geopandas import GeoDataFrame

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from geopandas import overlay


def clean_string(string: str) -> str:
    return string.replace(' ', '_').lower()

def normalized_proximity_index(aoi_gdf: GeoDataFrame, dist_gdf: GeoDataFrame) -> GeoDataFrame:
    aoi_pt = aoi_gdf.copy()
    dist_pt = dist_gdf.copy()

    aoi_pt.geometry = aoi_pt.centroid
    dist_pt.geometry = dist_pt.centroid

    dist_coords = np.array(list(dist_pt.geometry.apply(lambda geom: [geom.x, geom.y])))
    aoi_coords = np.array(list(aoi_pt.geometry.apply(lambda geom: [geom.x, geom.y])))

    tree = KDTree(dist_coords)

    distances, _ = tree.query(aoi_coords, k=1)
    aoi_pt['dist_nearest_point'] = distances

    min_dist = aoi_pt['dist_nearest_point'].min()
    max_dist = aoi_pt['dist_nearest_point'].max()

    aoi_pt['proximity_index'] = 1 - (
        (aoi_pt['dist_nearest_point'] - min_dist) / (max_dist - min_dist)
    )

    return aoi_pt

def imput_missing_data_distance_based(missing_data_gdf: GeoDataFrame, auxiliary_gdf: GeoDataFrame,
                                    missing_data_col: str, aux_data_col: str):
    gdf = missing_data_gdf.copy()
    aux_gdf = auxiliary_gdf.copy().dropna()

    # Dicionario nome do atributo -> nome das colunas
    features = {
        "nearest_data": f"{clean_string(aux_data_col)}_nearest",
        "idw": f"{clean_string(aux_data_col)}_idw",
        "distance": "dist_nearest_point",
        "imputed": f"{clean_string(missing_data_col)}_imputed"
    }

    # Convert negative data to null
    gdf[missing_data_col] = gdf[missing_data_col].mask(gdf[missing_data_col] < 0)

    # Calcular distâncias entre gdf e aux_gdf
    gdf_coords = np.array([[geom.x, geom.y] for geom in gdf.geometry.centroid.geometry])
    aux_coords = np.array([[geom.x, geom.y] for geom in aux_gdf.geometry.centroid.geometry])

    tree = KDTree(aux_coords)

    # Nearest point
    dist, idx = tree.query(gdf_coords, k=1)
    gdf[features["distance"]] = dist
    gdf[features["nearest_data"]] = aux_gdf.iloc[idx][aux_data_col].values

    # IDW (5 nearest neighbours ponderation)
    distances, indices = tree.query(gdf_coords, k=5)

    vals = aux_gdf[aux_data_col].to_numpy()
    aux_vals = np.take(vals, indices)

    weights: np.ndarray = 1 / (distances + 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    idw_vals = np.sum(aux_vals * weights, axis=1)
    gdf[features["idw"]] = idw_vals

    # list_features = [v for k,v in features.items() if k != "imputed"]
    list_features = [v for k,v in features.items() if k not in ("imputed", "nearest_data")]
    # scale = StandardScaler()
    df = gdf[[missing_data_col] + list_features]

    imp = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=100, random_state=0
        ),
        max_iter=10,
        random_state=42,
        sample_posterior=False
    )
    df_imputed = imp.fit_transform(df)

    # adicionar na tabela final
    gdf[features["imputed"]] = df_imputed[:, 0]

    return gdf

def gerar_representividade(
        ti_gdf: GeoDataFrame, 
        mun_gdf: GeoDataFrame, 
        primary_key: str = 'pol_id', 
        mun_id: str = 'mun_id',
        pop_total: str = 'pop_total',
        pop_indig_total: str = 'pop_indigena',
        pop_indig_ti: str = 'pop_indigena_ti'):
    parts = overlay(ti_gdf[[primary_key, 'geometry']], mun_gdf[[mun_id, 'geometry']], how='intersection')
    parts['area_km2'] = parts.geometry.area / 1e6

    areasum = parts.groupby(mun_id, as_index=False).area_km2.sum().rename(columns={'area_km2': 'areasum_mun'})
    parts = parts.merge(areasum, on=mun_id, how='left')
    parts = parts.merge(mun_gdf[[mun_id, pop_total, pop_indig_total, pop_indig_ti]], on=mun_id, how='left')
    
    # Pop indigena total por mun
    pop_total_municipal = parts[[mun_id, pop_indig_total]].drop_duplicates()[pop_indig_total].sum()
    # Pop indigena parcial (por TI) para cada mun
    pop_total_ti = parts[[mun_id, pop_indig_ti]].drop_duplicates()[pop_indig_ti].sum()
    
    # 3️⃣ CORREÇÃO: Calcular fator de correção para população MUNICIPAL
    # (se necessário, dependendo da qualidade dos dados)
    fator_correcao_mun = pop_total_municipal / parts[[mun_id, pop_indig_total]].drop_duplicates()[pop_indig_total].sum()
    parts['pop_indigena_corrigida'] = parts[pop_indig_total] * fator_correcao_mun

    # 4️⃣ Calcular a área total de cada município que intersecta TIs
    mun_area = parts.groupby(mun_id)['area_km2'].sum().reset_index()
    mun_area.rename(columns={'area_km2': 'area_total_mun_tis'}, inplace=True)
    parts = parts.merge(mun_area, on=mun_id)

    # 5️⃣ Alocação principal: usar pop_indigena_ti quando disponível (dados DIRETOS da TI)
    parts['pop_alloc'] = np.where(
        parts[pop_indig_ti].notna() & (parts['areasum_mun'] > 0),
        parts[pop_indig_ti] * (parts['area_km2'] / parts['areasum_mun']),
        np.nan
    )

    # 6️⃣ Fallback: se pop_indigena_ti é NaN, usar população MUNICIPAL proporcional à área
    parts['pop_alloc_fallback'] = np.where(
        parts[pop_indig_ti].isna() & (parts['area_total_mun_tis'] > 0),
        parts['pop_indigena_corrigida'] * (parts['area_km2'] / parts['area_total_mun_tis']),
        np.nan
    )



    # 7️⃣ Combinar alocação principal + fallback
    parts['pop_part_final'] = parts['pop_alloc'].combine_first(parts['pop_alloc_fallback'])
    # 8️⃣ VERIFICAÇÃO: Comparar com dado real quando disponível
    total_estimado_ti = parts['pop_part_final'].sum()
    total_real_ti = pop_total_ti

    # 9️⃣ Calcular fator de correção FINAL baseado na superestimativa
    if total_real_ti > 0:
        fator_correcao_final = total_real_ti / total_estimado_ti
        print(f"Fator de correção final: {fator_correcao_final:.4f}")
        parts['pop_part_final_corrigido'] = parts['pop_part_final'] * fator_correcao_final
    else:
        parts['pop_part_final_corrigido'] = parts['pop_part_final']

    # 🔟 Calcular IR_municipal com fallback robusto
    parts['IR_municipal_corrigido'] = np.where(
        parts[pop_indig_ti].notna() & (parts[pop_indig_ti] > 0),
        parts['pop_part_final_corrigido'] / parts[pop_indig_ti],
        np.where(
            parts['pop_indigena_corrigida'].notna() & (parts['pop_indigena_corrigida'] > 0),
            parts['pop_part_final_corrigido'] / parts['pop_indigena_corrigida'],
            0
        )
    )

    # 1️⃣1️⃣ Agrupar por TI para resultados finais
    ti_grouped = parts.groupby(primary_key, as_index=False).agg({
        'pop_part_final_corrigido': 'sum',
        'area_km2': 'sum'
    })
    IR_mun_corrigido_agg = parts.groupby(primary_key).apply(
        lambda x: np.average(x['IR_municipal_corrigido'], weights=x['area_km2'])
    ).reset_index(name='IR_municipal_corrigido_avg')

    # Calcular IR_global
    ti_grouped['pop_part_final_corrigido'] = ti_grouped['pop_part_final_corrigido'].astype(int)
    ti_max_pop = ti_grouped['pop_part_final_corrigido'].max()
    ti_grouped['IR_global'] = ti_grouped['pop_part_final_corrigido'] / ti_max_pop
    ti_grouped = ti_grouped.merge(IR_mun_corrigido_agg, on=primary_key, how='left')


    return ti_grouped