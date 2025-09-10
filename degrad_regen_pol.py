import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from rasterio.mask import mask
from rasterio.features import geometry_mask
import warnings
warnings.filterwarnings('ignore')


from models.mapbiomas import MapBiomas
from models.gpkg import GeoPackage

def calculate_forest_changes_by_polygon(raster_1985_path, raster_2023_path, polygons_gdf, forest_list, human_list, output_shapefile=None):
    """
    Calculate forest regeneration and degeneration percentages for each polygon.
    
    Parameters:
    raster_1985_path: Path to 1985 land cover raster
    raster_2023_path: Path to 2023 land cover raster
    polygons_gdf: GeoDataFrame with polygons to analyze
    polygon_id_column: Column name containing unique polygon identifiers
    output_shapefile: Optional path to save results as shapefile
    
    Returns:
    GeoDataFrame with added columns for regeneration and degeneration statistics
    """
        # Create a copy of the input GeoDataFrame to avoid modifying the original
    results_gdf = polygons_gdf.copy()
    
    # Initialize new columns
    results_gdf['total_area_px'] = 0
    results_gdf['forest_1985_px'] = 0
    results_gdf['forest_2023_px'] = 0
    results_gdf['human_1985_px'] = 0
    results_gdf['human_2023_px'] = 0
    results_gdf['regenerated_px'] = 0
    results_gdf['degenerated_px'] = 0
    results_gdf['regeneration_%'] = 0.0
    results_gdf['degeneration_%'] = 0.0
    results_gdf['net_forest_change'] = 0
    results_gdf['net_change_%'] = 0.0
    results_gdf['forest_cover_1985_%'] = 0.0
    results_gdf['forest_cover_2023_%'] = 0.0
    
    # Open both rasters
    with rasterio.open(raster_1985_path) as src_1985, rasterio.open(raster_2023_path) as src_2023:
        
        # Check if CRS match
        if src_1985.crs != polygons_gdf.crs:
            print("Warning: Raster and polygon CRS don't match. Reprojecting polygons...")
            polygons_gdf = polygons_gdf.to_crs(src_1985.crs)
        
        # Process each polygon
        for idx, polygon in polygons_gdf.iterrows():
            try:
                # Get polygon geometry
                geom = [polygon.geometry]
                
                # Mask both rasters with the polygon
                masked_1985, transform_1985 = mask(src_1985, geom, crop=True, all_touched=True)
                masked_2023, transform_2023 = mask(src_2023, geom, crop=True, all_touched=True)
                
                # Get the data arrays (remove extra dimensions if needed)
                if len(masked_1985.shape) == 3:
                    data_1985 = masked_1985[0]
                else:
                    data_1985 = masked_1985
                
                if len(masked_2023.shape) == 3:
                    data_2023 = masked_2023[0]
                else:
                    data_2023 = masked_2023
                
                # Create masks for valid data (excluding NoData values)
                valid_mask_1985 = data_1985 != src_1985.nodata
                valid_mask_2023 = data_2023 != src_2023.nodata
                valid_mask = valid_mask_1985 & valid_mask_2023
                
                # Apply valid mask
                data_1985_valid = data_1985[valid_mask]
                data_2023_valid = data_2023[valid_mask]
                
                # Skip if no valid data
                if data_1985_valid.size == 0:
                    continue
                
                # Create masks for forest and human classes
                forest_mask_1985 = np.isin(data_1985_valid, forest_list)
                forest_mask_2023 = np.isin(data_2023_valid, forest_list)
                
                human_mask_1985 = np.isin(data_1985_valid, human_list)
                human_mask_2023 = np.isin(data_2023_valid, human_list)
                
                # Calculate areas
                total_pixels = data_1985_valid.size
                forest_area_1985 = np.sum(forest_mask_1985)
                forest_area_2023 = np.sum(forest_mask_2023)
                human_area_1985 = np.sum(human_mask_1985)
                human_area_2023 = np.sum(human_mask_2023)
                
                # Calculate regeneration: Human in 1985 -> Forest in 2023
                regenerated_mask = human_mask_1985 & forest_mask_2023
                regenerated_area = np.sum(regenerated_mask)
                
                # Calculate degeneration: Forest in 1985 -> Human in 2023
                degenerated_mask = forest_mask_1985 & human_mask_2023
                degenerated_area = np.sum(degenerated_mask)
                
                # Calculate percentages
                regeneration_percentage = (regenerated_area / human_area_1985) * 100 if human_area_1985 > 0 else 0
                degeneration_percentage = (degenerated_area / forest_area_1985) * 100 if forest_area_1985 > 0 else 0
                
                # Net forest change
                net_forest_change = forest_area_2023 - forest_area_1985
                net_change_percentage = (net_forest_change / forest_area_1985) * 100 if forest_area_1985 > 0 else 0
                
                # Store results
                results_gdf.loc[idx, 'total_area_px'] = total_pixels
                results_gdf.loc[idx, 'forest_1985_px'] = forest_area_1985
                results_gdf.loc[idx, 'forest_2023_px'] = forest_area_2023
                results_gdf.loc[idx, 'human_1985_px'] = human_area_1985
                results_gdf.loc[idx, 'human_2023_px'] = human_area_2023
                results_gdf.loc[idx, 'regenerated_px'] = regenerated_area
                results_gdf.loc[idx, 'degenerated_px'] = degenerated_area
                results_gdf.loc[idx, 'regeneration_%'] = regeneration_percentage
                results_gdf.loc[idx, 'degeneration_%'] = degeneration_percentage
                results_gdf.loc[idx, 'net_forest_change'] = net_forest_change
                results_gdf.loc[idx, 'net_change_%'] = net_change_percentage
                results_gdf.loc[idx, 'forest_cover_1985_%'] = (forest_area_1985 / total_pixels) * 100
                results_gdf.loc[idx, 'forest_cover_2023_%'] = (forest_area_2023 / total_pixels) * 100
                
            except Exception as e:
                print(f"Error processing polygon {idx}: {e}")
                continue
    
    # Save results if output path is provided
    if output_shapefile:
        results_gdf.to_file(output_shapefile)
        print(f"Results saved to: {output_shapefile}")
    
    return results_gdf

def create_summary_statistics(results_gdf):
    """
    Create summary statistics for the polygon analysis.
    """
    summary = {
        'total_polygons': len(results_gdf),
        'total_area_pixels': results_gdf['total_area_px'].sum(),
        'total_regenerated_px': results_gdf['regenerated_px'].sum(),
        'total_degenerated_px': results_gdf['degenerated_px'].sum(),
        'avg_regeneration_%': results_gdf['regeneration_%'].mean(),
        'avg_degeneration_%': results_gdf['degeneration_%'].mean(),
        'max_regeneration_%': results_gdf['regeneration_%'].max(),
        'max_degeneration_%': results_gdf['degeneration_%'].max(),
        'polygons_with_regeneration': (results_gdf['regeneration_%'] > 0).sum(),
        'polygons_with_degeneration': (results_gdf['degeneration_%'] > 0).sum()
    }
    
    return summary

def print_polygon_results(results_gdf, summary):
    """
    Print formatted results for polygon analysis.
    """
    print("=" * 80)
    print("FOREST CHANGE ANALYSIS BY POLYGON (1985 - 2023)")
    print("=" * 80)
    print(f"Total polygons analyzed: {summary['total_polygons']}")
    print(f"Total area: {summary['total_area_pixels']:,} pixels")
    print(f"Total regenerated area: {summary['total_regenerated_px']:,} pixels")
    print(f"Total degenerated area: {summary['total_degenerated_px']:,} pixels")
    print(f"Average regeneration: {summary['avg_regeneration_%']:.2f}%")
    print(f"Average degeneration: {summary['avg_degeneration_%']:.2f}%")
    print(f"Polygons with regeneration: {summary['polygons_with_regeneration']}")
    print(f"Polygons with degeneration: {summary['polygons_with_degeneration']}")
    print("=" * 80)
    
    # Print top 5 polygons by regeneration
    print("\nTop 5 polygons by regeneration percentage:")
    top_regeneration = results_gdf.nlargest(5, 'regeneration_%')[
        [results_gdf.columns[0], 'regeneration_%', 'regenerated_px', 'total_area_px']
    ]
    print(top_regeneration.to_string(index=False))
    
    # Print top 5 polygons by degeneration
    print("\nTop 5 polygons by degeneration percentage:")
    top_degeneration = results_gdf.nlargest(5, 'degeneration_%')[
        [results_gdf.columns[0], 'degeneration_%', 'degenerated_px', 'total_area_px']
    ]
    print(top_degeneration.to_string(index=False))

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    raster_1985 = r"data\ti_lulc_1985_reproject.tif"
    raster_2023 = r"data\ti_lulc_2023_reproject.tif"
    output_shp = "forest_changes.shp"
    mp = MapBiomas()
    gpkg = GeoPackage()
    gdf = gpkg.read_layer('ti_ac_preliminar')
    forest_list = mp.natural_classes
    human_list = mp.human_classes
    
    try:        
        # Calculate forest changes for each polygon
        results_gdf = calculate_forest_changes_by_polygon(
            raster_1985, raster_2023, gdf, forest_list, human_list
        )
        # results_gdf.to_file('data/teste_degen.geojson')
        
        # Create summary statistics
        summary = create_summary_statistics(results_gdf)
        
        # Print results
        print_polygon_results(results_gdf, summary)
        
        # Optional: Save results to CSV
        csv_path = "forest_changes_by_polygon.csv"
        results_gdf.drop(columns='geometry').to_csv(csv_path, index=False)
        print(f"\nTabular results saved to: {csv_path}")
        
        # Optional: Create a simple plot
        if len(results_gdf) > 0:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot regeneration percentages
            results_gdf.plot(column='regeneration_%', ax=ax1, legend=True,
                           cmap='Greens', scheme='quantiles',
                           legend_kwds={'title': 'Regeneration %'})
            ax1.set_title('Forest Regeneration by Polygon (%)')
            
            # Plot degeneration percentages
            results_gdf.plot(column='degeneration_%', ax=ax2, legend=True,
                           cmap='Reds', scheme='quantiles',
                           legend_kwds={'title': 'Degeneration %'})
            ax2.set_title('Forest Degeneration by Polygon (%)')
            
            plt.tight_layout()
            plt.savefig('forest_changes_by_polygon.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the file paths for your data.")
    except Exception as e:
        print(f"An error occurred: {e}")