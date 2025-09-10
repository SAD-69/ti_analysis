import os
import rasterio as rst
import numpy as np

from rasterio.mask import mask
from rasterio import DatasetReader
from geopandas import GeoDataFrame
from pandas import DataFrame, read_csv
from functools import cached_property


MAPBIOMAS_TIF_URL = 'https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection_10/lulc/coverage/brazil_coverage_{0}.tif'
MAPBIOMAS_CSV_URL = 'https://brasil.mapbiomas.org/wp-content/uploads/sites/4/2025/08/Codigos-da-legenda-colecao-10.zip'
CLASS_ID_COL = "Class_ID"
DESCRIPTION_COL = "Description"

class MapBiomas:
    def __init__(self):
        self.url = MAPBIOMAS_TIF_URL
        self.df_dict = read_csv(
            MAPBIOMAS_CSV_URL, 
            encoding='utf-8', 
            sep='\t', 
            compression='zip', 
            engine='python'
        )

    @cached_property
    def classes(self) -> DataFrame:
        df = self.df_dict.copy()
        natural_classes = (
            "Forest",
            "Forest Formation",
            "Savanna Formation",
            "Mangrove",
            "Floodable Forest",
            "Wooded Sandbank Vegetation",
            "Herbaceous and Shrubby Vegetation",
            "Wetland",
            "Grassland",
            "Herbaceous Sandbank Vegetation"
        )
        condition = df['Description'].isin(natural_classes)
        df['cover_landuse'] = np.where(condition, "cobertura_natural", "uso_solo")
        if not os.path.exists('data/cd_legenda_mapbiomas.csv'):
            df.to_csv('data/cd_legenda_mapbiomas.csv')
        return df
    
    @cached_property
    def class_names(self) -> dict:
        return dict(zip(self.classes[CLASS_ID_COL], self.classes[DESCRIPTION_COL]))
    
    @cached_property
    def natural_classes(self) -> np.ndarray:
        return self.classes[CLASS_ID_COL][self.classes['cover_landuse'] == 'cobertura_natural'].values
    
    @cached_property
    def human_classes(self) -> np.ndarray:
        return self.classes[CLASS_ID_COL][self.classes['cover_landuse'] != 'cobertura_natural'].values

    def clip_by_year(self, year: int, gdf: GeoDataFrame, filename: str = None):
        with rst.open(self.url.format(year)) as src:
            src: DatasetReader
            geom = gdf.to_crs(src.crs).geometry
            out_img, out_transf = mask(
                src,
                geom,
                crop=True,
                all_touched=True,
                filled=False
            )
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transf
            })
            if filename:
                with rst.open(filename, "w", **out_meta) as clipped_src:
                    clipped_src.write(out_img)

if __name__ == '__main__':
    mp = MapBiomas()
    print(mp.class_names)