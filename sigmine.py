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

ti_aoi_gdf = gpkg.read_layer('ti_inside_pampa_aoi')

intersect_gdf = sjoin(ti_aoi_gdf, sigmine_gdf, how='inner', predicate='intersects')

intersect_gdf['count'] = 1

qt_projetos_aoi_ti = intersect_gdf.groupby('status_proj')['count'].sum().reset_index()

qt_proj_por_aoi_ti = intersect_gdf.groupby('geometry')['count'].sum().reset_index()

