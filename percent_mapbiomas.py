from rasterstats import zonal_stats
from models.gpkg import GeoPackage
from models.mapbiomas import MapBiomas
from pandas import DataFrame

if __name__ == '__main__':
    gpkg = GeoPackage()
    gdf = gpkg.read_layer("ti_stats_sistemas_ecologicos_ipcc")
    mp_bioma = MapBiomas()
    class_dict = mp_bioma.class_names
    raster = 'data/ti_lulc_2023_reproject.tif'
    stats = zonal_stats(
        gdf,
        raster,
        categorical=True,
        nodata=0
    )
    df_stats = DataFrame(stats)
    df_stats = df_stats.rename(columns={k: v for k, v in class_dict.items() if k in df_stats.columns})
    df_stats_pct = df_stats.div(df_stats.sum(axis=1), axis=0) * 100
    gdf_final = gdf.join(df_stats_pct)

    gdf_final.to_file("data/ti_stats_se_mapbiomas.geojson")

    print(gdf_final.head())