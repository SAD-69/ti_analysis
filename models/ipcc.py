from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pandas import DataFrame
from geopandas import GeoDataFrame

class VulnerabilityIndex:
    """Classe para geração de Indicador de Vulnerabilidade
    """
    def __init__(self, df: DataFrame | GeoDataFrame, weight: dict[str, dict]):
        self.df = df
        if weight:
            self.weight = weight
        else:
            self.weight = {}

    @property
    def components(self):
        return self.weight.keys()
    
    @property
    def indicators(self):
        indicators = []
        for _, v in self.weight.items():
            indicators.extend(list(v.keys()))
        return indicators

    def normalize_indicator(self, df: GeoDataFrame | DataFrame, indicators: list[str],  method='minmax', direction='positive') -> DataFrame | GeoDataFrame:
        """
        Normalize indicators to 0-1 scale
        direction: 'positive' if higher values are better, 'negative' if higher values are worse
        """
        df.fillna(0, inplace=True)
        scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
        for indicator in indicators:
        #     if direction == 'negative':
        #         df[indicator] = 1 - scaler.fit_transform(df[[indicator]].values.reshape(-1, 1))
        #     else:
            df[indicator] = scaler.fit_transform(df[[indicator]].values.reshape(-1, 1))
        return df
    
    def _calc_indicator(self, df: DataFrame | GeoDataFrame,  indicator_col: str):
        """Calcula os indicadores baseados em seus pesos (AHP)

        Args:
            indicator_col (str): Nome da coluna do indicador

        Returns:
            int: valor do indicador * peso definido. Ex.: indicaodr(50) * peso (0.5) = 25
        """
        return sum(df[indicator] * weight for indicator, weight in self.weight[indicator_col].items())
    
    def calc_ipcc_vulnerability(self, ex_key: str = 'ex_score', se_key: str = 'se_score', ca_key: str = 'ca_score') -> DataFrame | GeoDataFrame:
        """Cálculo de Vulnerabilidade baseado na metodologia do IPCC

        Returns:
            DataFrame | GeoDataFrame: Tabela de dados
        """
        df = self.df.copy()
        ex_indicators = list(self.weight[ex_key].keys())
        se_indicators = list(self.weight[se_key].keys())
        ca_indicators = list(self.weight[ca_key].keys())
        df = self.normalize_indicator(df, ex_indicators)
        df = self.normalize_indicator(df, se_indicators)
        df = self.normalize_indicator(df, ca_indicators)
        # components = ('ex_score', 'se_score', 'ac_score')
        for comp in self.components:
            df[comp] = self._calc_indicator(df, comp)
        df['vulnerability_raw'] = (df[ex_key] + df[se_key]) + (1 - df[ca_key]) / 3
        df['vulnerability_index'] = MinMaxScaler().fit_transform(df[['vulnerability_raw']])
        return df

