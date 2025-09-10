import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import shape
import matplotlib.pyplot as plt

from models.mapbiomas import MapBiomas
from models.gpkg import GeoPackage

def calculate_forest_changes(raster_1985_path, raster_2023_path, forest_list: list, human_list: list, output_shapefile=None):
    """
    Calculate forest regeneration and degeneration percentages from 1985 to 2023.
    
    Parameters:
    raster_1985_path: Path to 1985 land cover raster
    raster_2023_path: Path to 2023 land cover raster
    output_shapefile: Optional path to save results as shapefile
    
    Returns:
    Dictionary with regeneration and degeneration statistics
    """
    
    # Read both rasters
    with rasterio.open(raster_1985_path) as src_1985:
        data_1985 = src_1985.read(1)
        profile_1985 = src_1985.profile
        transform_1985 = src_1985.transform
        
    with rasterio.open(raster_2023_path) as src_2023:
        data_2023 = src_2023.read(1)
        profile_2023 = src_2023.profile
        transform_2023 = src_2023.transform
    
    # Check if rasters have same dimensions
    if data_1985.shape != data_2023.shape:
        raise ValueError("Rasters must have the same dimensions")
    
    # Create masks for forest and human classes
    forest_mask_1985 = np.isin(data_1985, forest_list)
    forest_mask_2023 = np.isin(data_2023, forest_list)
    
    human_mask_1985 = np.isin(data_1985, human_list)
    human_mask_2023 = np.isin(data_2023, human_list)
    
    # Calculate total area (in pixels)
    total_pixels = data_1985.size
    
    # Calculate forest areas
    forest_area_1985 = np.sum(forest_mask_1985)
    forest_area_2023 = np.sum(forest_mask_2023)
    
    # Calculate regeneration: Human in 1985 -> Forest in 2023
    regenerated_mask = human_mask_1985 & forest_mask_2023
    regenerated_area = np.sum(regenerated_mask)
    
    # Calculate degeneration: Forest in 1985 -> Human in 2023
    degenerated_mask = forest_mask_1985 & human_mask_2023
    degenerated_area = np.sum(degenerated_mask)
    
    # Calculate percentages
    regeneration_percentage = (regenerated_area / human_mask_1985.sum()) * 100 if human_mask_1985.sum() > 0 else 0
    degeneration_percentage = (degenerated_area / forest_area_1985) * 100 if forest_area_1985 > 0 else 0
    
    # Net forest change
    net_forest_change = forest_area_2023 - forest_area_1985
    net_change_percentage = (net_forest_change / forest_area_1985) * 100 if forest_area_1985 > 0 else 0
    
    # Create results dictionary
    results = {
        'total_area_pixels': total_pixels,
        'forest_1985_area': forest_area_1985,
        'forest_2023_area': forest_area_2023,
        'regenerated_area': regenerated_area,
        'degenerated_area': degenerated_area,
        'regeneration_percentage': regeneration_percentage,
        'degeneration_percentage': degeneration_percentage,
        'net_forest_change': net_forest_change,
        'net_change_percentage': net_change_percentage,
        'forest_cover_1985_percentage': (forest_area_1985 / total_pixels) * 100,
        'forest_cover_2023_percentage': (forest_area_2023 / total_pixels) * 100
    }
    
    # Create change map if output shapefile is requested
    if output_shapefile:
        create_change_shapefile(regenerated_mask, degenerated_mask, transform_1985, output_shapefile)
    
    return results

def create_change_shapefile(regenerated_mask, degenerated_mask, transform, output_path):
    """
    Create a shapefile showing regeneration and degeneration areas.
    """
    # Create change classification raster
    change_raster = np.zeros_like(regenerated_mask, dtype=np.int8)
    change_raster[regenerated_mask] = 1  # Regenerated areas
    change_raster[degenerated_mask] = 2  # Degenerated areas
    
    # Convert raster to vector features
    features = []
    for geom, value in shapes(change_raster, transform=transform, connectivity=8):
        if value > 0:  # Only include changed areas
            feature = {
                'geometry': shape(geom),
                'class': int(value),
                'change_type': 'Regenerated' if value == 1 else 'Degenerated'
            }
            features.append(feature)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(features)
    gdf.crs = "EPSG:4326"  # Adjust CRS as needed
    
    # Save to shapefile
    gdf.to_file(output_path)
    
    return gdf

def print_results(results):
    """
    Print formatted results.
    """
    print("=" * 60)
    print("FOREST CHANGE ANALYSIS (1985 - 2023)")
    print("=" * 60)
    print(f"Total area: {results['total_area_pixels']:,} pixels")
    print(f"Forest cover 1985: {results['forest_1985_area']:,} pixels ({results['forest_cover_1985_percentage']:.2f}%)")
    print(f"Forest cover 2023: {results['forest_2023_area']:,} pixels ({results['forest_cover_2023_percentage']:.2f}%)")
    print(f"Net forest change: {results['net_forest_change']:+,} pixels ({results['net_change_percentage']:+.2f}%)")
    print("-" * 60)
    print(f"Regenerated area (Human → Forest): {results['regenerated_area']:,} pixels")
    print(f"Regeneration percentage: {results['regeneration_percentage']:.2f}% of 1985 human areas")
    print(f"Degenerated area (Forest → Human): {results['degenerated_area']:,} pixels")
    print(f"Degeneration percentage: {results['degeneration_percentage']:.2f}% of 1985 forest areas")
    print("=" * 60)

# Example usage
if __name__ == "__main__":
    # Replace with your actual raster file paths
    raster_1985 = r"data\ti_lulc_1985_reproject.tif"
    raster_2023 = r"data\ti_lulc_2023_reproject.tif"
    output_shp = "forest_changes.shp"
    mp = MapBiomas()
    forest_list = mp.natural_classes
    human_list = mp.human_classes
    
    try:
        # Calculate forest changes
        results = calculate_forest_changes(raster_1985, raster_2023, forest_list, human_list)
        
        # Print results
        print_results(results)
        
        # Optional: Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot regeneration areas
        ax1.set_title('Regenerated Areas (Human → Forest)')
        ax1.imshow(results.get('regenerated_mask', np.zeros((100, 100))), cmap='Greens')
        
        # Plot degeneration areas
        ax2.set_title('Degenerated Areas (Forest → Human)')
        ax2.imshow(results.get('degenerated_mask', np.zeros((100, 100))), cmap='Reds')
        
        plt.tight_layout()
        plt.savefig('forest_change_map.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the file paths for your raster data.")
    except Exception as e:
        print(f"An error occurred: {e}")