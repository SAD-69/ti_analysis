import numpy as np
from geopandas import GeoDataFrame, sjoin

SIGMINE_CONDITION = [
    'REQUERIMENTO DE PESQUISA',
    'REQUERIMENTO DE LAVRA GARIMPEIRA',
    'REQUERIMENTO DE REGISTRO DE EXTRAÇÃO',
    'REQUERIMENTO DE LICENCIAMENTO',
    'DIREITO DE REQUERER A LAVRA',
    'DADO NÃO CADASTRADO',
    'AUTORIZAÇÃO DE PESQUISA',
    'DISPONIBILIDADE',
    'APTO PARA DISPONIBILIDADE'
]

def calc_mining_threat(main_gdf: GeoDataFrame, count_gdf: GeoDataFrame, fase_col: str = 'fase', primary_key: str = 'pol_id') -> GeoDataFrame:
    gdf = main_gdf.copy()
    sigmine_gdf = count_gdf.copy()
    condition = sigmine_gdf[fase_col].isin(SIGMINE_CONDITION)
    sigmine_gdf['status_proj'] = np.where(condition, 'futuro', 'corrente')
    sigmine_gdf['status_weight'] = np.where(sigmine_gdf['status_proj'].isin(['futuro']), 0.5, 1)
    intersect_gdf = sjoin(gdf, sigmine_gdf, how='inner', predicate='intersects')

    qt_gdf = (
        intersect_gdf
        .groupby([primary_key, 'status_proj'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    w_gdf = (
        intersect_gdf
        .groupby(primary_key)['status_weight']
        .sum()
        .reset_index()
    )
    gdf = gdf.merge(qt_gdf, on=primary_key, how='left')
    gdf = gdf.merge(w_gdf, on=primary_key, how='left')
    gdf[['corrente', 'futuro', 'status_weight']] = gdf[['corrente', 'futuro', 'status_weight']].fillna(0)
    
    # Qt de processos/km² (1e-6)
    gdf['mining_threat_density'] = gdf['status_weight'] / (gdf.geometry.area * 1e-6)
    return gdf