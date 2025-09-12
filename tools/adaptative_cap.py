import numpy as np

from scipy.spatial import KDTree
from geopandas import GeoDataFrame

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def clean_string(string: str) -> str:
    return string.replace(' ', '_').lower()

def normalized_proximity_index(aoi_gdf: GeoDataFrame, dist_gdf: GeoDataFrame) -> GeoDataFrame:
    aoi_pt = aoi_gdf.copy()
    dist_pt = dist_gdf.copy()

    aoi_pt.geometry = aoi_pt.centroid
    dist_pt.geometry = dist_pt.centroid

    dist_coords = np.array(list(dist_pt.geometry.apply(lambda geom: [geom.x, geom.y])))
    aoi_coords = np.array(list(aoi_pt.geometry.apply(lambda geom: [geom.x, geom.y])))

    tree = KDTree(dist_coords)

    distances, _ = tree.query(aoi_coords, k=1)
    aoi_pt['dist_nearest_point'] = distances

    min_dist = aoi_pt['dist_nearest_point'].min()
    max_dist = aoi_pt['dist_nearest_point'].max()

    aoi_pt['proximity_index'] = 1 - (
        (aoi_pt['dist_nearest_point'] - min_dist) / (max_dist - min_dist)
    )

    return aoi_pt

def imput_missing_data_distance_based(missing_data_gdf: GeoDataFrame, auxiliary_gdf: GeoDataFrame,
                                    missing_data_col: str, aux_data_col: str):
    gdf = missing_data_gdf.copy()
    aux_gdf = auxiliary_gdf.copy().dropna()

    # Dicionario nome do atributo -> nome das colunas
    features = {
        "nearest_data": f"{clean_string(aux_data_col)}_nearest",
        "idw": f"{clean_string(aux_data_col)}_idw",
        "distance": "dist_nearest_point",
        "imputed": f"{clean_string(missing_data_col)}_imputed"
    }

    # Convert negative data to null
    gdf[missing_data_col] = gdf[missing_data_col].mask(gdf[missing_data_col] < 0)

    # Calcular distâncias entre gdf e aux_gdf
    gdf_coords = np.array([[geom.x, geom.y] for geom in gdf.geometry.centroid.geometry])
    aux_coords = np.array([[geom.x, geom.y] for geom in aux_gdf.geometry.centroid.geometry])

    tree = KDTree(aux_coords)

    # Nearest point
    dist, idx = tree.query(gdf_coords, k=1)
    gdf[features["distance"]] = dist
    gdf[features["nearest_data"]] = aux_gdf.iloc[idx][aux_data_col].values

    # IDW (5 nearest neighbours ponderation)
    distances, indices = tree.query(gdf_coords, k=5)

    vals = aux_gdf[aux_data_col].to_numpy()
    aux_vals = np.take(vals, indices)

    weights: np.ndarray = 1 / (distances + 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    idw_vals = np.sum(aux_vals * weights, axis=1)
    gdf[features["idw"]] = idw_vals

    # list_features = [v for k,v in features.items() if k != "imputed"]
    list_features = [v for k,v in features.items() if k not in ("imputed", "nearest_data")]
    # scale = StandardScaler()
    df = gdf[[missing_data_col] + list_features]

    imp = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=100, random_state=0
        ),
        max_iter=10,
        random_state=42,
        sample_posterior=False
    )
    df_imputed = imp.fit_transform(df)

    # adicionar na tabela final
    gdf[features["imputed"]] = df_imputed[:, 0]

    return gdf