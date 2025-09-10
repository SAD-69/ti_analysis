
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import geopandas as gpd

from models.gpkg import GeoPackage

def normalize_indicators(df: pd.DataFrame | gpd.GeoDataFrame, indicators, method='minmax', direction='positive'):
    """
    Normalize indicators to 0-1 scale
    direction: 'positive' if higher values are better, 'negative' if higher values are worse
    """
    normalized_df = df.copy()
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    
    for indicator in indicators:
        if direction == 'negative':
            # Invert so higher values always mean worse outcome for vulnerability
            normalized_df[indicator] = 1 - scaler.fit_transform(df[[indicator]].values.reshape(-1, 1))
        else:
            normalized_df[indicator] = scaler.fit_transform(df[[indicator]].values.reshape(-1, 1))
    
    return normalized_df


def calculate_ipcc_vulnerability(df, weights=None):
    """
    Calculate IPCC Vulnerability Index
    V = (Exposure + Sensitivity) - Adaptive Capacity
    """
    if weights is None:
        # Default equal weights - YOU SHOULD ADJUST THESE BASED ON EXPERT KNOWLEDGE
        weights = {
            'adaptive_capacity': {'educ_pct_imputed': 0.25, 'proximity_index': 0.25,
                                 'regeneration_%': 0.25, 'forest_cover_2023_%': 0.25}
        }
    
    df['adaptive_capacity_score'] = sum(df[indicator] * weight for indicator, weight 
                                      in weights['adaptive_capacity'].items())
    
    # # Calculate vulnerability (normalized to 0-1)
    # df['vulnerability_raw'] = (df['exposure_score'] + df['sensitivity_score']) - df['adaptive_capacity_score']
    # df['vulnerability_index'] = MinMaxScaler().fit_transform(df[['vulnerability_raw']])
    
    return df


if __name__ == '__main__':
    gpkg = GeoPackage()
    gdf = gpkg.read_layer('CA_TESTE')
    indicators = ['educ_pct_imputed', 'proximity_index', 'regeneration_%', 'forest_cover_2023_%']
    df = normalize_indicators(gdf, indicators)
    df = calculate_ipcc_vulnerability(df)
    print(df.head())
    df.to_file("data/CA_teste.geojson", driver='GeoJSON')