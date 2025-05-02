import geopandas as gpd
import os
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv

PG_USER = os.getenv('PG_USER')
PG_PASS = os.getenv('PG_PASS')
PG_HOST = os.getenv('PG_HOST')
PG_DB = os.getenv('PG_DB')
PG_PORT = os.getenv('PG_PORT')
load_dotenv()
engine = create_engine(f'postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}')

def read_gdb(path: str, layer: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path, layer=layer, driver='OpenFileGDB')


if __name__ == '__main__':
    lista_gdb = os.listdir('fgb')
    layer_list = ['SUB', 'SSDMT', 'SSDAT', 'SSDBT']
    
    for gdb in lista_gdb:
        gdb_path = os.path.join('fgb', gdb)
        try:
            for layer in layer_list:
                table_name = gdb.split('_')[0] + f'_{layer}'
                gdf = read_gdb(gdb_path, layer)
                gdf.to_postgis(table_name, engine.connect(), 'aneel')
                print(f"Table {table_name} saved succesfully on db")
        except Exception as e:
            print(f"Error: {str(e)}")
