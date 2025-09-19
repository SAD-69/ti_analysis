from models.gpkg import GeoPackage
from tools.graphmaker import GraphMaker
from tools.exposure import spatial_join_pampa, calc_mining_threat

gpkg = GeoPackage()
min_gdf = gpkg.read_layer('sigmine_rs')
ti = gpkg.read_layer('ti_all_revisada_v1')
ti['pol_id'] = ti.index

a = calc_mining_threat(ti, min_gdf)
# gdf = gpkg.read_layer('lisa_skater')
# se = gpkg.read_layer('sistemas_ecologicos_RS_hasenack')
# graph = GraphMaker(
#     gdf,
#     region='SISTEMA')
# gdf = spatial_join_pampa(gdf, se, bioma_col='SISTEMA')

# graph.plot_etnia()

