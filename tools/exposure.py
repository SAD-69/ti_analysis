import numpy as np
from geopandas import GeoDataFrame, sjoin, overlay

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


def count_fire_incidents(main_gdf: GeoDataFrame, pt_heat_gdf: GeoDataFrame, fire: bool = True):
    if fire:
        qt_col = 'qt_foco_calor'
        dens_col = 'heat_density'
    else:
        qt_col = 'qt_uhe'
        dens_col = 'uhe_density'
    gdf = main_gdf.copy()
    joined = sjoin(pt_heat_gdf, gdf, how='inner', predicate='within')
    counts = joined.groupby('index_right').size()
    gdf[qt_col] = gdf.index.map(counts).fillna(0).astype(int)
    gdf[dens_col] = gdf[qt_col] / (gdf.geometry.area * 1e-6)
    return gdf


def spatial_join_pampa(main_gdf: GeoDataFrame, biome_gdf: GeoDataFrame, primary_key: str = 'pol_id', bioma_col: str = 'bioma') -> GeoDataFrame:
    gdf = main_gdf.copy()
    join_gdf = sjoin(gdf, biome_gdf[[bioma_col, 'geometry']], how='left', predicate='intersects')
    join_gdf.drop(columns=['index_right'], inplace=True)
    join_gdf.drop_duplicates(subset=primary_key, inplace=True)
    return join_gdf

def buffer_cut(main_gdf: GeoDataFrame, buffer_col: str = 'dist_buf') -> GeoDataFrame:
    gdf = main_gdf.copy()
    gdf.geometry = gdf.geometry.buffer(gdf[buffer_col])
    gdf.geometry = gdf.geometry.difference(main_gdf.geometry)
    return gdf

def road_density(main_gdf: GeoDataFrame, road_gdf: GeoDataFrame, primary_key: str = 'pol_id') -> GeoDataFrame:
    gdf = main_gdf.copy()
    roads_inside = overlay(road_gdf, gdf, how='intersection')
    roads_inside['road_length'] = roads_inside.geometry.length
    road_length_per_ti = roads_inside.groupby(primary_key)['road_length'].sum()
    gdf['road_length'] = gdf.index.map(road_length_per_ti).fillna(0)
    # km/km²
    gdf['road_density'] = (gdf['road_length'] / 1000) / (gdf.geometry.area / 1e6)
    return gdf