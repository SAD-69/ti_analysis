from models.gpkg import GeoPackage
from models.mapbiomas import MapBiomas
from models.ipcc import VulnerabilityIndex
from geopandas import sjoin
from copy import deepcopy
from functools import cached_property
from tools.exposure import spatial_join_pampa, buffer_cut, estrutura_fundiaria_sum
from tools.adaptative_cap import (
    normalized_proximity_index, 
    imput_missing_data_distance_based,
    gerar_representividade)

mp = MapBiomas()


class Analysis(GeoPackage):
    def __init__(self, 
    layer: str = 'ti_all_revisada_v1', 
    year_i: int = 1985, year_f: int = 2023,
    column_list: list[str] = [
        'pol_id', 
        'geometry', 
        'status_fundiario', 
        'bioma', 
        'etnia_nome', 
        'nome_ti', 
        'source', 
        'fonte',
        'dist_buf'
        ],
    forest_list: list[int] = mp.natural_classes,
    non_forest_list: list[int] = mp.human_classes
    ):
        super().__init__()
        self.layer = layer
        self.year_i = year_i
        self.year_f = year_f
        self.status_map = {
            "Regularizada": 1,
            "Declarada": 0.75,
            "Delimitada": 0.5,
            "Em Estudo": 0.25,
            None: 0
        }
        self.biomas = self.read_layer('biomas_rs_faixa_trans_final')
        self.funai_gdf = self.read_layer('instituicoes_indigena')
        self.ed_gdf = self.read_layer('censo_tx_alfabet')
        self.inep = self.read_layer('inep_dados')
        self.sigmine = self.read_layer('sigmine_rs')
        self.foco_calor = self.read_layer('focos_calor_all_year')
        self.uhe = self.read_layer('uhe')
        self.roads = self.read_layer('roads')
        self._cols_to_keep = column_list
        self.forest_list = forest_list
        self.non_forest_lit = non_forest_list

    def count_etnia_bioma(self, save: bool = True):
        data = self.gdf.groupby(['bioma', 'etnia_nome']).size()
        if save:
            data.to_csv("data/qt_etnia_bioma.csv", encoding='utf-8')
        print(data)

    @cached_property
    def gdf(self):
        gdf = self.read_layer(self.layer)
        gdf['pol_id'] = gdf.index
        gdf['status_fundiario'] = gdf['fase_ti'].map(self.status_map)
        gdf = gdf.rename(columns={'TI': 'nome_ti'})
        gdf = spatial_join_pampa(gdf, self.biomas)
        gdf = gdf[self._cols_to_keep]
        gdf['etnia_nome'] = gdf['etnia_nome'].replace(
            {
                "Guaraní": "Guarani",
                "Guaraní,Kaingang": "Guarani - Kaingang",
                "Guarani Mbya": "Guarani",
                None: "Sem dados"
            }
        )
        return gdf
    
    @cached_property
    def buffer_gdf(self):
        return buffer_cut(self.gdf)
    
    @cached_property
    def ti_ed_gdf(self):
        gdf = sjoin(self.gdf, self.ed_gdf[['geometry', 'TxAlfabetI']], how='left')
        gdf = imput_missing_data_distance_based(gdf, self.inep, 'TxAlfabetI', 'IDEB_2023')
        gdf = gdf[['pol_id', 'txalfabeti_imputed']]
        return gdf
    
    @cached_property
    def pop_ti(self):
        gdf = self.read_layer('pop_indigena_mun')
        gdf['mun_id'] = gdf.index
        gdf['PessInd'] = gdf['PessInd'].fillna(0).astype(int)
        gdf['pop_indigena'] = gdf['PessInd'].mask(gdf['PessInd'] < 0)
        gdf['PopIndEmT']= gdf['PessIndEmT'].fillna(0).astype(int)
        gdf['pop_indigena_ti'] = gdf['PessIndEmT'].mask(gdf['PessIndEmT'] < 0)
        gdf['PopResid'] = gdf['PopResid'].fillna(0).astype(int)
        gdf['pop_total'] = gdf['PopResid'].mask(gdf['PopResid'] < 0)
        gdf['perc_indigena'] = gdf['pop_indigena'] / gdf['pop_total']
        gdf['perc_em_ti'] = gdf['pop_indigena_ti'] / gdf['pop_indigena']
        gdf = gerar_representividade(self.gdf, gdf)
        gdf = gdf[['pol_id', 'IR_municipal_corrigido_avg']].rename(columns={"IR_municipal_corrigido_avg": "rep_index"})
        return gdf
    
    @cached_property
    def car(self):
        gdf = self.read_layer('sicar_rs')
        gdf = estrutura_fundiaria_sum(self.buffer_gdf, gdf)
        return gdf