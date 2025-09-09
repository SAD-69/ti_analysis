from models.mapbiomas import MapBiomas
from models.gpkg import GeoPackage

import os
import rasterio

from rasterio.mask import mask
from rasterio.warp import reproject

import geopandas as gpd
import numpy as np

from tqdm import tqdm


YEARS = (1985, 2023)
if __name__ == '__main__':
    gpkg = GeoPackage()
    map_biomas = MapBiomas()
    
    # RS
    rs_gdf = gpkg.read_layer('rs_buffer_poly')
    # TI's
    gdf = gpkg.read_layer('se_rs_clip')
    # gdf.to_crs(4326, inplace=True)

    # LULC Mapbiomas (1985-2023)
    for year in YEARS:
        lu_lc_path = f'data/ti_lulc_{year}.tif'
        if not os.path.exists(lu_lc_path):
            map_biomas.clip_by_year(year, rs_gdf, lu_lc_path)
    
    mpbiomas_classes = map_biomas.classes

    CLASS_NAMES = dict(zip(mpbiomas_classes["Class_ID"], mpbiomas_classes["Description"]))
    NATURAL_CLASSES = mpbiomas_classes['Class_ID'][mpbiomas_classes['cover_landuse'] == 'cobertura_natural'].values
    HUMAN_CLASSES = mpbiomas_classes['Class_ID'][mpbiomas_classes['cover_landuse'] == 'uso_solo'].values
    
    results = []
    with rasterio.open('data/ti_lulc_1985_reproject.tif') as src_1985:
        with rasterio.open('data/ti_lulc_2023_reproject.tif') as src_2023:
            for idx, row in tqdm(gdf.iterrows()):
                geom = [row.geometry]
                try:
                    lulc_1985_masked, transform_1985 = mask(src_1985, geom, crop=True, all_touched=True)
                    lulc_2023_masked, transform_2023 = mask(src_2023, geom, crop=True, all_touched=True)
                    # Get the affine transform for the masked area
                    out_meta = src_1985.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": lulc_1985_masked.shape[1],
                        "width": lulc_1985_masked.shape[2],
                        "transform": transform_1985
                    })
                    
                    # Flatten arrays for analysis
                    lulc_1985_flat = lulc_1985_masked.flatten()
                    lulc_2023_flat = lulc_2023_masked.flatten()

                    transition_counts = {f"{CLASS_NAMES[from_cls]}_to_{CLASS_NAMES[to_cls]}": 0 
                                        for from_cls in NATURAL_CLASSES for to_cls in HUMAN_CLASSES}
                    
                    total_transitions = 0

                    # Analyze each pixel
                    for i in range(len(lulc_1985_flat)):
                        class_1985 = lulc_1985_flat[i]
                        class_2023 = lulc_2023_flat[i]
                        
                        # Check if transition from natural to human use
                        if class_1985 in NATURAL_CLASSES and class_2023 in HUMAN_CLASSES:
                            transition_key = f"{CLASS_NAMES[class_1985]}_to_{CLASS_NAMES[class_2023]}"
                            transition_counts[transition_key] += 1
                            total_transitions += 1
                    
                    # Calculate area of transitions (assuming pixel size in square meters)
                    pixel_size = src_1985.res[0] * src_1985.res[1]  # in square units of the CRS
                    transition_area = {k: v * pixel_size for k, v in transition_counts.items()}
                    
                    # Add results for this polygon
                    result_row = {
                        'polygon_id': idx,
                        'geometry': row.geometry,
                        'total_transitions': total_transitions,
                        'total_transition_area': total_transitions * pixel_size
                    }
                    
                    # Add individual transition counts and areas
                    result_row.update(transition_counts)
                    for key, value in transition_area.items():
                        result_row[f"{key}_area"] = value
                    
                    results.append(result_row)
                    
                except Exception as e:
                    print(f"Error processing polygon {idx}: {e}")
                    # Add a row with zero values for this polygon
                    result_row = {
                        'polygon_id': idx,
                        'geometry': row.geometry,
                        'total_transitions': 0,
                        'total_transition_area': 0
                    }
                    # Initialize zero values for all transition types
                    for from_cls in NATURAL_CLASSES:
                        for to_cls in HUMAN_CLASSES:
                            key = f"{CLASS_NAMES[from_cls]}_to_{CLASS_NAMES[to_cls]}"
                            result_row[key] = 0
                            result_row[f"{key}_area"] = 0
                    
                    results.append(result_row)
    
    # Convert results to GeoDataFrame
    results_gdf = gpd.GeoDataFrame(results, geometry='geometry')
    results_gdf.crs = gdf.crs  # Set the CRS from the original territory data
    
    # print(results_gdf.head())
    results_gdf.to_file("data/se_rs_clip.geojson")