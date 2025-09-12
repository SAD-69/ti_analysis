import os
import rasterio
import numpy as np
from pandas import concat

from rasterio.mask import mask
from rasterstats import zonal_stats
from pandas import DataFrame
from geopandas import GeoDataFrame

def lulc_percentage(gdf: GeoDataFrame, raster: str, class_dict: dict[int, str]) -> GeoDataFrame:
    stats = zonal_stats(
        gdf,
        raster,
        categorical=True,
        nodata=0
    )
    df_stats = DataFrame(stats)
    cols = {k: v for k, v in class_dict.items() if k in df_stats.columns}
    df_stats.rename(columns=cols, inplace=True)
    df_stats_pct = df_stats.div(df_stats.sum(axis=1), axis=0) * 100
    joined_gdf = gdf.join(df_stats_pct)
    return joined_gdf

def lulc_percentage_enhanced(
    gdf: GeoDataFrame, 
    raster: str, 
    class_dict: dict[int, str],
    aggregate_classes: dict[str, list[int]] = None,
    keep_original: bool = True
) -> GeoDataFrame:
    """
    Calcula porcentagens de classes de uso do solo com opção de agregação.
    
    Parameters:
    -----------
    keep_original : bool
        Se True, mantém ambas as classes originais e agregadas
    """
    stats = zonal_stats(
        gdf,
        raster,
        categorical=True,
        nodata=0
    )
    
    df_stats = DataFrame(stats)
    cols = {k: v for k, v in class_dict.items() if k in df_stats.columns}
    df_stats.rename(columns=cols, inplace=True)
    
    total_pixels = df_stats.sum(axis=1)
    
    # Calcular porcentagens das classes originais
    df_stats_pct_original = df_stats.div(total_pixels, axis=0) * 100
    
    # Processar agregações se especificado
    if aggregate_classes:
        df_aggregated = DataFrame(index=df_stats.index)
        
        for aggregate_name, class_values in aggregate_classes.items():
            # Encontrar colunas correspondentes aos valores
            cols_to_aggregate = []
            for class_val in class_values:
                # Verificar se o valor existe no class_dict e encontrar nome correspondente
                if class_val in class_dict:
                    class_name = class_dict[class_val]
                    if class_name in df_stats.columns:
                        cols_to_aggregate.append(class_name)
            
            if cols_to_aggregate:
                # Somar as colunas selecionadas
                df_aggregated[aggregate_name] = df_stats[cols_to_aggregate].sum(axis=1)
            else:
                # Se não encontrar colunas, criar coluna com zeros
                df_aggregated[aggregate_name] = 0
        
        # Calcular porcentagens CORRETAMENTE sobre o total de pixels
        df_aggregated_pct = df_aggregated.div(total_pixels, axis=0) * 100
        
        # Combinar resultados baseado na opção keep_original
        if keep_original:
            # Manter ambas as classes originais e agregadas
            final_df = concat([df_stats_pct_original, df_aggregated_pct], axis=1)
        else:
            # Manter apenas as classes agregadas
            final_df = df_aggregated_pct
            
    else:
        # Se não houver agregação, usar apenas classes originais
        final_df = df_stats_pct_original
    
    final_df = final_df.fillna(0)  # Preencher NaN com 0
    
    # Juntar com GeoDataFrame original
    joined_gdf = gdf.join(final_df)
    
    return joined_gdf


def calc_forest_changes(raster_0: str, raster_f: str, input_gdf: GeoDataFrame, 
                        forest_list: list[int], non_forest_list: list[int], year_0: int = 1985, year_f: int = 2023):
    gdf = input_gdf.copy()
    # Initialize new columns
    gdf['total_area_px'] = 0
    gdf[f'forest_{year_0}_px'] = 0
    gdf[f'forest_{year_f}_px'] = 0
    gdf[f'non_forest_{year_0}_px'] = 0
    gdf[f'non_forest_{year_f}_px'] = 0
    gdf['regenerated_px'] = 0
    gdf['degenerated_px'] = 0
    gdf['regeneration_%'] = 0.0
    gdf['degeneration_%'] = 0.0
    gdf['net_forest_change'] = 0
    gdf['net_change_%'] = 0.0
    gdf[f'forest_cover_{year_0}_%'] = 0.0
    gdf[f'forest_cover_{year_f}_%'] = 0.0

    with rasterio.open(raster_0) as src_0, rasterio.open(raster_f) as src_f:
        if src_0.crs != gdf.crs:
            print("Warning: Raster and polygon CRS don't match. Reprojecting polygons...")
            gdf.to_crs(src_0.crs, inplace=True)

        for idx, pol in gdf.iterrows():
            try:
                geom = [pol.geometry]
                mask_0, _ = mask(src_0, geom, crop=True, all_touched=True)
                mask_f, _ = mask(src_f, geom, crop=True, all_touched=True)

                if len(mask_0.shape) ==3:
                    data_0 = mask_0[0]
                else:
                    data_0 = mask_0
                if len(mask_f.shape) ==3:
                    data_f = mask_f[0]
                else:
                    data_f = mask_f

                valid_mask_0 = data_0 != src_0.nodata
                valid_mask_f = data_f != src_f.nodata
                valid_mask = valid_mask_0 & valid_mask_f

                data_0_valid = data_0[valid_mask]
                data_f_valid = data_f[valid_mask]

                if data_0_valid.size == 0:
                    continue

                forest_mask_0 = np.isin(data_0_valid, forest_list)
                non_forest_mask_0 = np.isin(data_0_valid, non_forest_list)

                forest_mask_f = np.isin(data_f_valid, forest_list)
                non_forest_mask_f = np.isin(data_f_valid, non_forest_list)

                total_pixels = data_0_valid.size
                
                forest_area_0 = np.sum(forest_mask_0)
                forest_area_f = np.sum(forest_mask_f)

                non_forest_area_0 = np.sum(non_forest_mask_0)
                non_forest_area_f = np.sum(non_forest_mask_f)

                regen_mask = non_forest_mask_0 & forest_mask_f
                regen_area = np.sum(regen_mask)

                degen_mask = forest_mask_0 & non_forest_mask_f
                degen_area = np.sum(degen_mask)
                
                regen_perc = (regen_area / non_forest_area_0) * 100 if non_forest_area_0 > 0 else 0
                degen_perc = (degen_area / forest_area_0) * 100 if forest_area_0 > 0 else 0

                net_forest_change = forest_area_f - forest_area_0
                net_change_perc = (net_forest_change / forest_area_0) * 100 if forest_area_0 > 0 else 0

                # Store results
                gdf.loc[idx, 'total_area_px'] = total_pixels
                gdf.loc[idx, f'forest_{year_0}_px'] = forest_area_0
                gdf.loc[idx, f'forest_{year_f}_px'] = forest_area_f
                gdf.loc[idx, f'non_forest_{year_0}_px'] = non_forest_area_0
                gdf.loc[idx, f'non_forest_{year_f}_px'] = non_forest_area_f
                gdf.loc[idx, 'regenerated_px'] = regen_area
                gdf.loc[idx, 'degenerated_px'] = degen_area
                gdf.loc[idx, 'regeneration_%'] = regen_perc
                gdf.loc[idx, 'degeneration_%'] = degen_perc
                gdf.loc[idx, 'net_forest_change'] = net_forest_change
                gdf.loc[idx, 'net_change_%'] = net_change_perc
                gdf.loc[idx, f'forest_cover_{year_0}_%'] = (forest_area_0 / total_pixels) * 100
                gdf.loc[idx, f'forest_cover_{year_f}_%'] = (forest_area_f / total_pixels) * 100
            except Exception as e:
                print(f"Error processing polygon {idx}: {e}")
                continue
    return gdf


if __name__ == '__main__':
    from models.gpkg import GeoPackage
    from models.mapbiomas import MapBiomas

    raster_1985 = r"data\ti_lulc_1985_reproject.tif"
    raster_2023 = r"data\ti_lulc_2023_reproject.tif"
    mp = MapBiomas()
    gpkg = GeoPackage()
    gdf = gpkg.read_layer('ti_ac_preliminar')
    gdf['pol_id'] = gdf.index
    forest_list = mp.natural_classes
    non_forest_list = mp.human_classes

    degen_gdf = calc_forest_changes(raster_1985, raster_2023, gdf, forest_list, non_forest_list)
    lulc_perc_gdf = lulc_percentage(gdf, raster_2023, mp.class_names)

    final_gdf = degen_gdf.join(lulc_perc_gdf, on='pol_id')
    print(final_gdf)
    print(final_gdf.describe())