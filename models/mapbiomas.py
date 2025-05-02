import rasterio as rst
import requests
from io import BytesIO


class MapBiomas:
    def __init__(self):
        self.url = 'https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection_10/lulc/coverage/brazil_coverage_{0}.tif'

    def lulc_by_year(self, year: int) -> rst.DatasetReader:
        return rst.open(self.url.format(year))
    
    def download_lulc(self, year: int, path: str) -> None:
        r = requests.get(self.url.format(year))
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
        