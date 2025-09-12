
import pandas as pd

from tools.raster import calc_forest_changes, lulc_percentage, lulc_percentage_enhanced
from tools.adaptative_cap import normalized_proximity_index, imput_missing_data_distance_based
from tools.exposure import calc_mining_threat
from models.gpkg import GeoPackage
from models.mapbiomas import MapBiomas
from models.ipcc import VulnerabilityIndex
from geopandas import sjoin


if __name__ == '__main__':
    YEAR_1 = 2010
    YEAR_2 = 2023
    raster_1985 = f"data/ti_lulc_{YEAR_1}_reproject.tif"
    raster_2023 = f"data/ti_lulc_{YEAR_2}_reproject.tif"
    mp = MapBiomas()
    gpkg = GeoPackage()
    
    status_map = {
        "Regularizada": 1,
        "Declarada": 0.75,
        "Delimitada": 0.5,
        "Em Estudo": 0.25,
        None: 0
    }

    gdf = gpkg.read_layer('ti_all_revisada_v1')
    gdf['status_fundiario'] = gdf['fase_ti'].map(status_map)
    gdf['pol_id'] = gdf.index
    gdf = gdf.rename(columns={'TI': 'nome_ti'})

    inst_gdf = gpkg.read_layer('instituicoes_indigena')
    ed_gdf = gpkg.read_layer('censo_tx_alfabet')
    inep_gdf = gpkg.read_layer('inep_dados')
    mine_gdf = gpkg.read_layer('sigmine_rs')
    

    ti_ed_gdf = sjoin(gdf, ed_gdf[['geometry', 'TxAlfabetI']], how='left')
    buffer_gdf = gdf.copy()
    buffer_gdf.geometry = buffer_gdf.geometry.buffer(gdf.dist_buf)
    forest_list = mp.natural_classes
    non_forest_list = mp.human_classes
    
    degen_gdf = calc_forest_changes(raster_1985, raster_2023, gdf, forest_list, non_forest_list)
    degen_gdf = degen_gdf[['pol_id',  'nome_ti', 'etnia_nome', 'geometry', 'regeneration_%', 'degeneration_%', 'forest_cover_2023_%']]
    agg_dict = {
        "agropec": [14, 15, 18, 19, 21]
    }
    # lulc_perc_gdf = lulc_percentage(gdf, raster_2023, mp.class_names)
    lulc_perc_gdf = lulc_percentage_enhanced(gdf, raster_2023, mp.class_names, agg_dict)
    cols = [i for i in mp.class_names.values() if i in lulc_perc_gdf.columns]
    lulc_perc_gdf = lulc_perc_gdf[['pol_id', 'forest_formation', 'soybean', 'forest_plantation', 'urban_area', 'agropec']]

    sigmine_gdf = calc_mining_threat(gdf, mine_gdf)
    sigmine_gdf = sigmine_gdf[['pol_id', 'mining_threat_density']]

    b_degen_gdf = calc_forest_changes(raster_1985, raster_2023, buffer_gdf, forest_list, non_forest_list)
    b_degen_gdf = b_degen_gdf[['pol_id', 'geometry', 'regeneration_%', 'degeneration_%', 'forest_cover_2023_%']]

    b_lulc_perc_gdf = lulc_percentage_enhanced(buffer_gdf, raster_2023, mp.class_names, agg_dict)
    # b_lulc_perc_gdf = lulc_percentage(buffer_gdf, raster_2023, mp.class_names)
    b_lulc_perc_gdf = b_lulc_perc_gdf[['pol_id', 'forest_formation', 'soybean', 'forest_plantation', 'urban_area', 'agropec']]

    b_sigmine_gdf = calc_mining_threat(buffer_gdf, mine_gdf)
    b_sigmine_gdf = sigmine_gdf[['pol_id', 'mining_threat_density']]

    ca_prox_gdf = normalized_proximity_index(gdf, inst_gdf)
    ca_ed_gdf = imput_missing_data_distance_based(ti_ed_gdf, inep_gdf, 'TxAlfabetI', 'IDEB_2023')
    ca_prox_gdf = ca_prox_gdf[['pol_id', 'geometry', 'proximity_index', 'status_fundiario']]
    ca_ed_gdf = ca_ed_gdf[['pol_id', 'txalfabeti_imputed']]

    ca_gdf = pd.merge(ca_prox_gdf, ca_ed_gdf, on='pol_id', how='left')
    ex_gdf = pd.merge(b_degen_gdf, b_lulc_perc_gdf, on='pol_id', how='left')
    se_gdf = pd.merge(degen_gdf, lulc_perc_gdf, on='pol_id', how='left')

    both_mining_gdf = pd.merge(sigmine_gdf, b_sigmine_gdf, on='pol_id', how='left', suffixes=['_se', '_ex'])

    se_ex_gdf = pd.merge(se_gdf, ex_gdf, on='pol_id', how='left', suffixes=['_se', '_ex'])
    ipcc_mining_gdf = pd.merge(se_ex_gdf, both_mining_gdf, on='pol_id', how='left')
    final_gdf = pd.merge(ipcc_mining_gdf, ca_gdf, on='pol_id', how='left')
    # final_gdf = pd.merge(semifinal_gdf, ipcc_mining_gdf, on='pol_id', how='left')
    
    indicators = [
        'forest_plantation',
        'soybean',
        'urban_area',
        'agropec',
        'degeneration_%',
        'mining_threat_density'
    ]

    weights = {
        'ex_score': {
            'forest_plantation_ex': 1/6,
            'soybean_ex': 1/6,
            'urban_area_ex': 1/6,
            'degeneration_%_ex': 1/6,
            'agropec_ex': 1/6,
            'mining_threat_density_ex': 1/6
        },
        'se_score': {
            'forest_plantation_se': 1/6,
            'soybean_se': 1/6,
            'urban_area_se': 1/6,
            'degeneration_%_se': 1/6,
            'agropec_se': 1/6,
            'mining_threat_density_se': 1/6
        },
        'ca_score': {
            'forest_cover_2023_%_ex': 1/5,
            'regeneration_%_ex': 1/5,
            'txalfabeti_imputed': 1/5,
            'proximity_index': 1/5,
            'status_fundiario': 1/5
        }
    }
    ex_ind = [f'{i}_ex' for i in indicators]
    indicators = [f'{i}_se' for i in indicators]
    
    indicators = indicators + ex_ind + list(weights['ca_score'].keys())
    vindex = VulnerabilityIndex(final_gdf, weights)
    df = vindex.calc_ipcc_vulnerability()
    df.drop(columns=['geometry', 'geometry_ex']).to_file(f"data/vindex_{YEAR_1}_{YEAR_2}_v8.geojson")
    print("NORMALIZED DATA")
    print("="*80)
    print(df[indicators].head())
    print(df[indicators].describe())
    print("="*80)
    print("VULNERABILITY INDEX")
    print(df['vulnerability_index'].describe())
    
