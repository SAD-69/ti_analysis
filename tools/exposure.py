import numpy as np
from geopandas import GeoDataFrame, sjoin, overlay
import pandas as pd

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

    mining = sigmine_gdf.copy()

    # Define status e peso
    condition = mining[fase_col].isin(SIGMINE_CONDITION)
    mining['status_proj'] = np.where(condition, 'futuro', 'corrente')
    mining['status_weight'] = np.where(mining['status_proj'] == 'futuro', 0.5, 1.0)

    # Interseção TI x processos
    intersect = overlay(gdf, mining, how='intersection')

    # Calcula área ponderada
    intersect['intersect_area'] = intersect.geometry.area
    intersect['weighted_area'] = intersect['intersect_area'] * intersect['status_weight']

    # Soma por TI
    mining_area = (
        intersect.groupby(primary_key)['weighted_area']
        .sum()
        .reset_index()
    )

    # Junta com as TIs
    gdf = gdf.merge(mining_area, on=primary_key, how='left')
    gdf['weighted_area'] = gdf['weighted_area'].fillna(0)

    # Normaliza pela área da TI (proporção da área afetada)
    gdf['mining_threat_area_ratio'] = gdf['weighted_area'] / gdf.geometry.area
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


def estrutura_fundiaria_sum(main_gdf: GeoDataFrame, car_gdf: GeoDataFrame, primary_key: str = 'pol_id'):
    gdf = main_gdf.copy()
    gdf = gdf[[primary_key, 'geometry']]
    car = car_gdf.copy()
    car["tipo_propriedade"] = pd.cut(
        car["mod_fiscal"],
        bins=[-float("inf"), 0.99, 4, 15, float("inf")],
        labels=["Minifúndio", "Pequena propriedade", "Média propriedade", "Grande propriedade"]
    )

    pesos = {
        "Minifúndio": 0.125,
        "Pequena propriedade": 0.25,
        "Média propriedade": 0.5,
        "Grande propriedade": 1
    }

    car['est_fundiaria'] = car['tipo_propriedade'].map(pesos).astype(float)
    joined_gdf = sjoin(gdf, car[['geometry', 'tipo_propriedade', 'est_fundiaria']])
    joined_gdf = joined_gdf.groupby(primary_key)['est_fundiaria'].sum()
    gdf = pd.merge(gdf, joined_gdf, on=primary_key, how='outer')
    return gdf