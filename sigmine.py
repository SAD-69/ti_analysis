from models.gpkg import GeoPackage
import numpy as np
from geopandas import sjoin

gpkg = GeoPackage()

# print(gpkg.layer_list)

sigmine_gdf = gpkg.read_layer('sigmine_rs')

# Clean data

print(sigmine_gdf.fase.unique())

future_condition = [
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

condition = sigmine_gdf['fase'].isin(future_condition)

sigmine_gdf['status_proj'] = np.where(condition, 'futuro', 'corrente')
sigmine_gdf['status_weight'] = np.where(sigmine_gdf['status_proj'].isin(['futuro']), 0.5, 1)


print(sigmine_gdf.head())

ti_aoi_gdf = gpkg.read_layer('ti_all_revisada_v1')
ti_aoi_gdf['pol_id'] = ti_aoi_gdf.index

intersect_gdf = sjoin(ti_aoi_gdf, sigmine_gdf, how='inner', predicate='intersects')

intersect_gdf['count'] = 1

qt_projetos_aoi_ti = intersect_gdf.groupby('status_proj')['count'].sum().reset_index()

qt_proj_por_aoi_ti = (
    intersect_gdf
    .groupby(['pol_id', 'status_proj'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Merge back into ti_aoi_gdf
ti_aoi_gdf = ti_aoi_gdf.merge(qt_proj_por_aoi_ti, on='pol_id', how='left')

# Fill NaNs with 0
ti_aoi_gdf[['corrente', 'futuro']] = ti_aoi_gdf[['corrente', 'futuro']].fillna(0).astype(int)

print(ti_aoi_gdf[['pol_id', 'corrente', 'futuro']].describe())