from models.mapbiomas import MapBiomas
from models.gpkg import GeoPackage

import os
import rasterio

from rasterio.mask import mask

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

    results = []
    with rasterio.open('data/ti_lulc_1985_reproject.tif') as src_1985:
        with rasterio.open('data/ti_lulc_2023_reproject.tif') as src_2023:
            for idx, row in tqdm(gdf.iterrows()):
                geom = [row.geometry]
                try:
                    lulc_1985_masked, transform_1985 = mask(src_1985, geom, crop=True, all_touched=True)
                    lulc_2023_masked, transform_2023 = mask(src_2023, geom, crop=True, all_touched=True)
                    
                    # Flatten arrays for analysis
                    lulc_1985_flat = lulc_1985_masked.flatten()
                    lulc_2023_flat = lulc_2023_masked.flatten()
                    
                    # Remove nodata values if any
                    valid_mask = (lulc_1985_flat != src_1985.nodata) & (lulc_2023_flat != src_2023.nodata)
                    lulc_1985_valid = lulc_1985_flat[valid_mask]
                    lulc_2023_valid = lulc_2023_flat[valid_mask]
                    
                    # VECTORIZED APPROACH - Much faster
                    # Create masks for category transitions
                    natural_mask = np.isin(lulc_1985_valid, map_biomas.natural_classes)
                    human_mask = np.isin(lulc_2023_valid, map_biomas.human_classes)
                    transition_mask = natural_mask & human_mask
                    
                    # Count total transitions
                    total_transitions = np.sum(transition_mask)
                    
                    # Get the specific class transitions for those pixels that changed categories
                    from_classes = lulc_1985_valid[transition_mask]
                    to_classes = lulc_2023_valid[transition_mask]
                    
                    # Count transitions by specific class pairs (if needed for detailed analysis)
                    class_transitions = {}
                    for from_cls in map_biomas.natural_classes:
                        for to_cls in map_biomas.human_classes:
                            count = np.sum((from_classes == from_cls) & (to_classes == to_cls))
                            if count > 0:
                                class_transitions[f"class_{map_biomas.class_names[from_cls]}_to_{map_biomas.class_names[to_cls]}"] = count
                    
                    # Calculate area of transitions
                    pixel_size = src_1985.res[0] * src_1985.res[1]  # in square units of the CRS
                    total_transition_area = total_transitions * pixel_size
                    
                    # Add results for this polygon - now simplified to category level
                    result_row = {
                        'polygon_id': idx,
                        'geometry': row.geometry,
                        'total_transitions': total_transitions,
                        'total_transition_area': total_transition_area,
                        'natural_to_human_area': total_transition_area,  # Main category transition
                        'natural_to_human_count': total_transitions,     # Main category transition count
                    }
                    
                    # Add detailed class transitions if needed (optional)
                    result_row.update(class_transitions)
                    
                    results.append(result_row)
                    
                except Exception as e:
                    print(f"Error processing polygon {idx}: {e}")
                    # Add a row with zero values for this polygon
                    result_row = {
                        'polygon_id': idx,
                        'geometry': row.geometry,
                        'total_transitions': 0,
                        'total_transition_area': 0,
                        'natural_to_human_area': 0,
                        'natural_to_human_count': 0,
                    }
                    results.append(result_row)
    
    # Convert results to GeoDataFrame
    results_gdf = gpd.GeoDataFrame(results, geometry='geometry')
    results_gdf.crs = gdf.crs  # Set the CRS from the original territory data
    
    # Calculate summary statistics
    total_area = results_gdf['total_transition_area'].sum()
    total_count = results_gdf['total_transitions'].sum()
    
    print(f"\nSUMMARY RESULTS:")
    print(f"Total area transitioned from natural to human use: {total_area:.2f} sq units")
    print(f"Total pixels transitioned: {total_count:,}")
    print(f"Number of polygons with transitions: {(results_gdf['total_transitions'] > 0).sum()}")
    
    # Save results
    results_gdf.to_file("data/se_rs_clip_lulc.geojson", driver='GeoJSON')
    print("Results saved to data/se_rs_clip_lulc.geojson")