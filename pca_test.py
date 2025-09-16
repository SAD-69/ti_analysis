import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from models.gpkg import GeoPackage

gpkg = GeoPackage()
df = gpkg.read_layer('vindex_2000_2023_v8')
# Example: indicators for ex_score
ex_indicators = [
    'forest_plantation_ex',
    'soybean_ex',
    'urban_area_ex',
    'degeneration_%_ex',
    'agropec_ex',
    'mining_threat_density_ex',
]

weights = {
        'ex_score': {
            'forest_plantation_ex': 1/7,
            'soybean_ex': 1/7,
            'urban_area_ex': 1/7,
            'degeneration_%_ex': 1/7,
            'agropec_ex': 1/7,
            'mining_threat_density_ex': 1/7,
            'heat_density_ex': 1/7
        },
        'se_score': {
            'forest_plantation_se': 1/7,
            'soybean_se': 1/7,
            'urban_area_se': 1/7,
            'degeneration_%_se': 1/7,
            'agropec_se': 1/7,
            'mining_threat_density_se': 1/7,
            'heat_density_se': 1/7
        },
        'ca_score': {
            'forest_cover_2023_%_ex': 1/5,
            'regeneration_%_ex': 1/5,
            'txalfabeti_imputed': 1/5,
            'proximity_index': 1/5,
            'status_fundiario': 1/5
        }
    }
# Subset dataframe
X = df[ex_indicators].copy()

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
pca.fit(X_scaled)

# Get loadings (importance of each variable in each component)
loadings = pd.DataFrame(
    pca.components_.T,
    index=ex_indicators,
    columns=[f"PC{i+1}" for i in range(len(ex_indicators))]
)

# Option 1: weights from absolute loadings of PC1
weights_pc1 = abs(loadings["PC1"]) / abs(loadings["PC1"]).sum()

print("Weights (from PC1 loadings):")
print(weights_pc1)

# Option 2: variance contribution method
explained_var = pca.explained_variance_ratio_
weights_all = (abs(loadings.values) * explained_var).sum(axis=1)
weights_all = weights_all / weights_all.sum()
weights_all = pd.Series(weights_all, index=ex_indicators)

print("\nWeights (variance contribution):")
print(weights_all)
