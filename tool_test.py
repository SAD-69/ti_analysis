
import pandas as pd

from tools.raster import calc_forest_changes, lulc_percentage
from tools.adaptative_cap import normalized_proximity_index, imput_missing_data_distance_based
from models.gpkg import GeoPackage
from models.mapbiomas import MapBiomas
from models.ipcc import VulnerabilityIndex
from geopandas import sjoin


if __name__ == '__main__':
    raster_1985 = r"data\ti_lulc_1985_reproject.tif"
    raster_2023 = r"data\ti_lulc_2023_reproject.tif"
    mp = MapBiomas()
    gpkg = GeoPackage()
    
    status_map = {
        "Regularizada": 1,
        "Declarada": 0.75,
        "Delimitada": 0.5,
        "Em Estudo": 0.25,
        None: 0
    }

    gdf = gpkg.read_layer('ti_ac_preliminar')
    gdf['status_fundiario'] = gdf['fase_ti'].map(status_map)
    inst_gdf = gpkg.read_layer('instituicoes_indigena')
    ed_gdf = gpkg.read_layer('censo_tx_alfabet')
    inep_gdf = gpkg.read_layer('inep_dados')
    gdf['pol_id'] = gdf.index

    ti_ed_gdf = sjoin(gdf, ed_gdf[['geometry', 'TxAlfabetI']], how='left')
    buffer_gdf = gdf.copy()
    buffer_gdf.geometry = buffer_gdf.geometry.buffer(gdf.dist_buf)
    forest_list = mp.natural_classes
    non_forest_list = mp.human_classes
    
    degen_gdf = calc_forest_changes(raster_1985, raster_2023, gdf, forest_list, non_forest_list)
    degen_gdf = degen_gdf[['pol_id', 'geometry', 'regeneration_%', 'degeneration_%', 'forest_cover_2023_%']]

    lulc_perc_gdf = lulc_percentage(gdf, raster_2023, mp.class_names)
    cols = [i for i in mp.class_names.values() if i in lulc_perc_gdf.columns]
    lulc_perc_gdf = lulc_perc_gdf[['pol_id', 'forest_formation', 'soybean', 'forest_plantation', 'urban_area']]

    b_degen_gdf = calc_forest_changes(raster_1985, raster_2023, buffer_gdf, forest_list, non_forest_list)
    b_degen_gdf = b_degen_gdf[['pol_id', 'geometry', 'regeneration_%', 'degeneration_%', 'forest_cover_2023_%']]

    b_lulc_perc_gdf = lulc_percentage(buffer_gdf, raster_2023, mp.class_names)
    b_lulc_perc_gdf = b_lulc_perc_gdf[['pol_id', 'forest_formation', 'soybean', 'forest_plantation', 'urban_area']]

    ca_prox_gdf = normalized_proximity_index(gdf, inst_gdf)
    ca_ed_gdf = imput_missing_data_distance_based(ti_ed_gdf, inep_gdf, 'TxAlfabetI', 'IDEB_2023')
    ca_prox_gdf = ca_prox_gdf[['pol_id', 'geometry', 'proximity_index', 'status_fundiario']]
    ca_ed_gdf = ca_ed_gdf[['pol_id', 'txalfabeti_imputed']]

    ca_gdf = pd.merge(ca_prox_gdf, ca_ed_gdf, on='pol_id', how='left')
    ex_gdf = pd.merge(b_degen_gdf, b_lulc_perc_gdf, on='pol_id', how='left')
    se_gdf = pd.merge(degen_gdf, lulc_perc_gdf, on='pol_id', how='left')


    se_ex_gdf = pd.merge(se_gdf, ex_gdf, on='pol_id', how='left', suffixes=['_se', '_ex'])
    final_gdf = pd.merge(se_ex_gdf, ca_gdf, on='pol_id', how='left')
    
    indicators = [
        'forest_plantation',
        'soybean',
        'urban_area',
        'degeneration_%'
    ]
    weights = {
        'ex_score': {
            'forest_plantation_ex': 0.10,
            'soybean_ex': 0.5,
            'urban_area_ex': 0.267,
            'degeneration_%_ex': 0.1288
        },
        'se_score': {
            'forest_plantation_se': 0.10,
            'soybean_se': 0.5,
            'urban_area_se': 0.267,
            'degeneration_%_se': 0.1288
        },
        'ca_score': {
            'forest_cover_2023_%_se': 0.483285,
            'regeneration_%_se': 0.357158,
            'txalfabeti_imputed': 0.051692,
            'proximity_index': 0.046991,
            'status_fundiario': 0.060874

        }
    }
    ex_ind = [f'{i}_ex' for i in indicators]
    indicators = [f'{i}_se' for i in indicators]
    
    indicators = indicators + ex_ind + list(weights['ca_score'].keys())
    vindex = VulnerabilityIndex(final_gdf, weights)
    df = vindex.calc_ipcc_vulnerability()
    df.drop(columns=['geometry', 'geometry_ex']).to_file("data/vulnerability_index_revised_v1.geojson")
    print("NORMALIZED DATA")
    print("="*80)
    print(df[indicators].head())
    print(df[indicators].describe())
    print("="*80)
    print("VULNERABILITY INDEX")
    print(df['vulnerability_index'].describe())
    
