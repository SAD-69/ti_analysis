import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pandas import DataFrame, cut
from geopandas import GeoDataFrame

class VulnerabilityIndex:
    def __init__(self, df: DataFrame | GeoDataFrame, weight: dict):
        self.df = df
        if weight:
            self.weight = weight
        else:
            self.weight = {}

    def _calc_indicator(self, indicator_col: str):
        return sum(self.df[indicator] * weight for indicator, weight in self.weight[indicator_col].items())
    
    def calc_ipcc_vulnerability(self) -> DataFrame | GeoDataFrame:
        df = self.df.copy()
        components = ('ex_score', 'se_score', 'ac_score')
        for comp in components:
            df[comp] = self._calc_indicator(comp)
        df['vulnerability_raw'] = (df[components[0]] + df[components[1]]) - df[components[2]]
        df['vulnerability index'] = MinMaxScaler().fit_transform(df['vulnerability_raw'])
        return df
