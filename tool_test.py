
import pandas as pd
import copy

from tools.raster import calc_forest_changes, lulc_percentage, lulc_percentage_enhanced
from tools.adaptative_cap import (
    normalized_proximity_index, 
    imput_missing_data_distance_based,
    gerar_representividade)
from tools.exposure import (
    calc_mining_threat, 
    count_fire_incidents, 
    spatial_join_pampa, 
    buffer_cut,
    road_density,
    estrutura_fundiaria_sum)
# from tools.pca_weight import (
#     pca_measure_weight, 
#     plot_pca_heatmap, 
#     plot_explained_variance, 
#     plot_scree,
#     analyze_bioma_weights,
#     plot_bioma_comparison,
#     pca_measure_weight_flexible)
from models.gpkg import GeoPackage
from models.mapbiomas import MapBiomas
from models.ipcc import VulnerabilityIndex
from geopandas import sjoin

def run(year_0: int, year_f: int):
    raster_1985 = f"data/ti_lulc_{year_0}_reproject.tif"
    raster_2023 = f"data/ti_lulc_{year_f}_reproject.tif"
    mp = MapBiomas()
    gpkg = GeoPackage()
    name_ti = gpkg.read_layer('ind_represent')
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
    gdf = pd.merge(gdf, name_ti[['pol_id', 'nome_ti']], on='pol_id', how='left')
    # gdf = gdf.rename(columns={'TI': 'nome_ti'})

    # BIOMAS
    biomas = gpkg.read_layer('biomas_rs_faixa_trans')
    gdf = spatial_join_pampa(gdf, biomas)

    inst_gdf = gpkg.read_layer('instituicoes_indigena')
    ed_gdf = gpkg.read_layer('censo_tx_alfabet')
    inep_gdf = gpkg.read_layer('inep_dados')
    mine_gdf = gpkg.read_layer('sigmine_rs')
    mine_gdf.geometry = mine_gdf.geometry.force_2d()
    foco_calor = gpkg.read_layer('focos_calor_all_year')
    uhe_gdf = gpkg.read_layer('uhe')
    road_gdf = gpkg.read_layer('roads')
    pop_ti = gpkg.read_layer('pop_indigena_mun')
    pop_ti['mun_id'] = pop_ti.index
    pop_ti['PessInd'] = pop_ti['PessInd'].fillna(0).astype(int)
    pop_ti['pop_indigena'] = pop_ti['PessInd'].mask(pop_ti['PessInd'] < 0)
    pop_ti['PopIndEmT']= pop_ti['PessIndEmT'].fillna(0).astype(int)
    pop_ti['pop_indigena_ti'] = pop_ti['PessIndEmT'].mask(pop_ti['PessIndEmT'] < 0)
    pop_ti['PopResid'] = pop_ti['PopResid'].fillna(0).astype(int)
    pop_ti['pop_total'] = pop_ti['PopResid'].mask(pop_ti['PopResid'] < 0)
    pop_ti['perc_indigena'] = pop_ti['pop_indigena'] / pop_ti['pop_total']
    pop_ti['perc_em_ti'] = pop_ti['pop_indigena_ti'] / pop_ti['pop_indigena'] 


    car = gpkg.read_layer('sicar_rs')
    ti_ed_gdf = sjoin(gdf, ed_gdf[['geometry', 'TxAlfabetI']], how='left')
    buffer_gdf = buffer_cut(gdf)
    buffer_gdf = estrutura_fundiaria_sum(buffer_gdf, car)
    forest_list = mp.natural_classes
    non_forest_list = mp.human_classes

    # Indice representatividade
    ir_gdf = gerar_representividade(gdf, pop_ti)
    
    # gpkg.save_layer(ir_gdf[['pol_id', 'nome_ti', 'etnia_nome', 'bioma', 'fonte', 'source','geometry', 'pop_part_final_corrigido', 'IR_municipal_corrigido', 'IR_global']])
    # print(ir_gdf)
    ir_gdf = ir_gdf[['pol_id', 'IR_municipal_corrigido_avg']].rename(columns={"IR_municipal_corrigido_avg": "rep_index"})

    # All data from 1998 to 2024 (risco_fogo > 0.5)
    foco_gdf = count_fire_incidents(gdf, foco_calor)
    foco_gdf = foco_gdf[['pol_id', 'heat_density', 'dist_buf']]
    b_foco_gdf = count_fire_incidents(buffer_gdf, foco_calor)
    b_foco_gdf = b_foco_gdf[['pol_id', 'heat_density', 'est_fundiaria']]

    # All UHE (UHE, CGH, PCH)
    uhe_se = count_fire_incidents(gdf, uhe_gdf, fire=False)
    uhe_se = uhe_se[['pol_id', 'uhe_density']]
    b_uhe_ex = count_fire_incidents(buffer_gdf, uhe_gdf, fire=False)
    b_uhe_ex = b_uhe_ex[['pol_id', 'uhe_density']]

    uhe_gdf = pd.merge(uhe_se, b_uhe_ex, on='pol_id', how='left', suffixes=['_se', '_ex'])
    foco_gdf = pd.merge(foco_gdf, b_foco_gdf, on='pol_id', how='left', suffixes=['_se', '_ex'])

    degen_gdf = calc_forest_changes(raster_1985, raster_2023, gdf, forest_list, non_forest_list)
    degen_gdf = degen_gdf[['pol_id',  'nome_ti', 'etnia_nome', 'geometry', 'regeneration_%', 'degeneration_%', 'forest_cover_2023_%', 'bioma']]
    agg_dict = {
        "agropec": [14, 15, 18, 19, 21]
    }
    # lulc_perc_gdf = lulc_percentage(gdf, raster_2023, mp.class_names)
    lulc_perc_gdf = lulc_percentage_enhanced(gdf, raster_2023, mp.class_names, agg_dict)
    # cols = [i for i in mp.class_names.values() if i in lulc_perc_gdf.columns]
    lulc_perc_gdf = lulc_perc_gdf[['pol_id', 'forest_formation', 'grassland', 'soybean', 'forest_plantation', 'urban_area', 'agropec']]

    sigmine_gdf = calc_mining_threat(gdf, mine_gdf)
    sigmine_gdf = sigmine_gdf[['pol_id', 'mining_threat_area_ratio']]

    b_degen_gdf = calc_forest_changes(raster_1985, raster_2023, buffer_gdf, forest_list, non_forest_list)
    b_degen_gdf = b_degen_gdf[['pol_id', 'geometry', 'regeneration_%', 'degeneration_%', 'forest_cover_2023_%']]

    b_lulc_perc_gdf = lulc_percentage_enhanced(buffer_gdf, raster_2023, mp.class_names, agg_dict)
    # b_lulc_perc_gdf = lulc_percentage(buffer_gdf, raster_2023, mp.class_names)
    b_lulc_perc_gdf = b_lulc_perc_gdf[['pol_id', 'forest_formation', 'grassland', 'soybean', 'forest_plantation', 'urban_area', 'agropec']]

    b_sigmine_gdf = calc_mining_threat(buffer_gdf, mine_gdf)
    b_sigmine_gdf = b_sigmine_gdf[['pol_id', 'mining_threat_area_ratio']]

    ca_prox_gdf = normalized_proximity_index(gdf, inst_gdf)
    ca_ed_gdf = imput_missing_data_distance_based(ti_ed_gdf, inep_gdf, 'TxAlfabetI', 'IDEB_2023')
    ca_prox_gdf = ca_prox_gdf[['pol_id', 'geometry', 'proximity_index', 'status_fundiario']]
    ca_ed_gdf = ca_ed_gdf[['pol_id', 'txalfabeti_imputed']]

    ca_gdf = pd.merge(ca_prox_gdf, ca_ed_gdf, on='pol_id', how='left')
    ex_gdf = pd.merge(b_degen_gdf, b_lulc_perc_gdf, on='pol_id', how='left')
    se_gdf = pd.merge(degen_gdf, lulc_perc_gdf, on='pol_id', how='left')

    both_mining_gdf = pd.merge(sigmine_gdf, b_sigmine_gdf, on='pol_id', how='left', suffixes=['_se', '_ex'])

    # road density
    road_dens = road_density(gdf, road_gdf)
    road_dens = road_dens[['pol_id', 'road_density']]
    b_road_dens = road_density(buffer_gdf, road_gdf)
    b_road_dens = b_road_dens[['pol_id', 'road_density']]
    both_road = pd.merge(road_dens, b_road_dens, on='pol_id', how='left', suffixes=['_se', '_ex'])

    se_ex_gdf = pd.merge(se_gdf, ex_gdf, on='pol_id', how='left', suffixes=['_se', '_ex'])
    ipcc_mining_gdf = pd.merge(se_ex_gdf, both_mining_gdf, on='pol_id', how='left')
    ipcc_heat_gdf = pd.merge(ipcc_mining_gdf, ca_gdf, on='pol_id', how='left')
    road_ipcc_gdf = pd.merge(ipcc_heat_gdf, both_road, on='pol_id', how='left')
    road_rep_gdf = pd.merge(road_ipcc_gdf, ir_gdf, on='pol_id', how='left')
    # car_road_gdf =  pd.merge(road_ipcc_gdf, ir_gdf, on='pol_id', how='left')
    semifinal_gdf = pd.merge(road_rep_gdf, uhe_gdf, on='pol_id', how='left')
    
    final_gdf = pd.merge(semifinal_gdf, foco_gdf, on='pol_id', how='left')
    final_gdf['regeneration_%_total'] = final_gdf['regeneration_%_ex'] + final_gdf['regeneration_%_se']
    final_gdf['remanescentes_veg'] = final_gdf['forest_formation_ex'] + final_gdf['forest_formation_se'] + final_gdf['grassland_ex'] + final_gdf['grassland_se']
    # gpkg.save_layer(final_gdf, 'pre_ipcc')
    indicators = [
        'forest_plantation',
        'soybean',
        'urban_area',
        'agropec',
        'degeneration_%',
        'mining_threat_area_ratio',
        'heat_density',
        'road_density',
        'uhe_density'
    ]

    weights = {
        'ex_score': {
            'forest_plantation_ex': 0.031,
            'soybean_ex': 0.164,
            'urban_area_ex': 0.021,
            'degeneration_%_ex': 0.201,
            'agropec_ex': 0.169,
            'mining_threat_area_ratio_ex': 0.137,
            'heat_density_ex': 0.16,
            'road_density_ex': 0.012,
            'uhe_density_ex': 0.046,
            'est_fundiaria': 0.06
        },
        'se_score': {
            'forest_plantation_se': 0.031,
            'soybean_se': 0.238,
            'urban_area_se': 0.045,
            'degeneration_%_se': 0.102,
            'agropec_se': 0.246,
            'mining_threat_area_ratio_se': 0.159,
            'heat_density_se': 0.09,
            'road_density_se': 0.013,
            'uhe_density_se': 0.075            
        },
        'ca_score': {
            'regeneration_%_total': 0.466,
            'remanescentes_veg': 0.315,
            'txalfabeti_imputed': 0.051,
            'proximity_index': 0.029,
            'status_fundiario': 0.098,
            'rep_index': 0.04
        }
    }
    weight_mata = copy.deepcopy(weights)

    pampa = final_gdf[final_gdf['bioma'] != 'Mata atlântica']
    mata = final_gdf[final_gdf['bioma'] == 'Mata atlântica']
    comps = list(weights.keys())
    comps.pop()

    ex_ind = [f'{i}_ex' for i in indicators]
    indicators = [f'{i}_se' for i in indicators]
    
    indicators = indicators + ex_ind + list(weights['ca_score'].keys())
    vindex_pampa = VulnerabilityIndex(pampa, weights)
    vindex_mata = VulnerabilityIndex(mata, weight_mata)
    df_pampa = vindex_pampa.calc_ipcc_vulnerability()
    df_mata = vindex_mata.calc_ipcc_vulnerability()
    df = pd.concat([df_pampa, df_mata])

    df = df.drop(columns=['geometry', 'geometry_ex'])
    bins = 6
    labels = ['Muito baixo', 'Baixo', 'Médio', 'Alto', 'Muito Alto', 'Crítico']
    ca_labels = ['Muito baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta', 'Excepcional']

    df['vulnerability_category'] = pd.qcut(df['vulnerability_index'], q=bins, labels=labels)
    df['ca_category_norm'] = pd.qcut(df['ca_score'], q=bins, labels=ca_labels)
    df['se_category_norm'] = pd.qcut(df['se_score'], q=bins, labels=labels)
    df['ex_category_norm'] = pd.qcut(df['ex_score'], q=bins, labels=labels)
    gpkg.save_layer(df, f'vindex_{year_0}_{year_f}_godmode')

    print("NORMALIZED DATA")
    print("="*80)
    print(df[indicators].head())
    print(df[indicators].describe())
    print("="*80)
    print("VULNERABILITY INDEX")
    print(df['vulnerability_index'].describe())

if __name__ == '__main__':
    run(1985, 2023)
    run(1985, 2000)
    run(2000, 2010)
    run(2010, 2023)
    
    
