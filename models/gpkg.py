from geopandas import GeoDataFrame, read_file
from fiona import listlayers

# GPKG_PATH = r'c:\Users\adminitsd\Documents\ufrgs\mestrado\qgz\final_master.gpkg'
GPKG_PATH = r'c:\Users\adminitsd\Documents\ufrgs\mestrado\qgz\ti_db.gpkg'
EPSG_CODE = 5880 # SIRGAS 2000/Brazil Polyconic

class GeoPackage:
    """Class to interact with GeoPackages
    """
    def __init__(self, gpkg_path: str = GPKG_PATH, epsg: int = EPSG_CODE):
        self.gpkg_path = gpkg_path
        self.epsg_code = epsg

    def read_layer(self, layer_name: str) -> GeoDataFrame:
        return read_file(self.gpkg_path, layer=layer_name).to_crs(epsg=EPSG_CODE)
    
    def save_layer(self, gdf: GeoDataFrame, layer_name) -> None:
        return gdf.to_file(self.gpkg_path, layer=layer_name)
    
    @property
    def layer_list(self) -> list[str, str]:
        return listlayers(self.gpkg_path)