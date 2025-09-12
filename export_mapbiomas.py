from models.mapbiomas import MapBiomas
from models.gpkg import GeoPackage

if __name__ == '__main__':
    db = GeoPackage()
    mp = MapBiomas()
    gdf = db.read_layer('rs_buffer')
    mp.clip_by_year(2010, gdf, f"data/lulc_mapbiomas_{2010}.tif")
    # for year in (2000, 2011):
        # mp.clip_by_year(year, gdf, f"data/lulc_mapbiomas_{year}.tif")