import geopandas as gpd
from owslib.wfs import WebFeatureService
from sqlalchemy import create_engine
from io import BytesIO

# -----------------------
# CONFIGURATION
# -----------------------
WFS_URL = "https://terrabrasilis.dpi.inpe.br/queimadas/geoserver/ows"  # Your GeoServer WFS endpoint
# FILTER = "estado = 'RIO GRANDE DO SUL'"          # CQL filter

# PostgreSQL connection string
PG_CONN_STR = "postgresql://postgres:postgres@mod-pais-srv01.nuvem.ufrgs.br:5432/mapa_gis"
TABLE_NAME = "your_table"

engine = create_engine(PG_CONN_STR)


ogc_filter = """
<ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">
  <ogc:PropertyIsEqualTo>
    <ogc:PropertyName>estado</ogc:PropertyName>
    <ogc:Literal>RIO GRANDE DO SUL</ogc:Literal>
  </ogc:PropertyIsEqualTo>
</ogc:Filter>
"""
# list_layer_name = [f"dados_abertos:focos_{year}_br_todosats" for year in range(1998, 2025)]
# -----------------------
# FETCH DATA FROM WFS
# -----------------------
print("Connecting to WFS...")
wfs = WebFeatureService(url=WFS_URL, version="1.1.0")

for year in range(1998, 2025):
    lyr_name = f"dados_abertos:focos_{year}_br_todosats"
    print(f"Fetching {lyr_name}")
    res = wfs.getfeature(
        lyr_name,
        filter=ogc_filter,
        outputFormat='application/json'
    )
    gdf = gpd.read_file(BytesIO(res.read()))
    
    print(f"Fetched {len(gdf)} features")

    # -----------------------
    # SAVE TO POSTGIS
    # -----------------------
    print("Saving to PostGIS...")

    # Writes to PostGIS (replaces if exists)
    gdf.to_postgis(f"focos_{year}_br_todosats", engine, if_exists="replace", index=False, schema="bd_queimadas")

    print(f"Data written to table focos_{year}_br_todosats")
