from geopandas import GeoDataFrame, read_file, sjoin
import matplotlib.pyplot as plt
import numpy as np

GPKG_PATH = r'c:\Users\adminitsd\Documents\ufrgs\mestrado\qgz\final_master.gpkg'
EPSG_CODE = 5880 # SIRGAS 2000/Brazil Polyconic

se_gdf = read_file(GPKG_PATH, layer='se_rs_clip')
ti_gdf = read_file(GPKG_PATH, layer='ti_inside_pampa')

se_gdf.to_crs(epsg=EPSG_CODE, inplace=True)
ti_gdf.to_crs(epsg=EPSG_CODE, inplace=True)

join_gdf = sjoin(ti_gdf, se_gdf, predicate='intersects')
guarani_list = ['Guarani', 'Guaraní', 'Guarani Mbya']
condition = join_gdf['etnia_nome'].isin(guarani_list)

join_gdf['etnia'] = np.where(condition, 'Guarani', join_gdf['etnia_nome'])
category_count = join_gdf['SISTEMA'].value_counts()
etnia_count = join_gdf['etnia'].value_counts()


print(category_count)
print(etnia_count)

etnia_count.plot.pie()
plt.show()

# join_gdf['SISTEMA'].hist(bins=5, edgecolor='black')
# plt.title('Distribution of Column Name')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.grid(axis='y', alpha=0.75) # Add grid for better readability
# plt.show() # Display the plot