from tools.raster import calc_forest_changes
from models.gpkg import GeoPackage
from models.mapbiomas import MapBiomas
import pandas as pd


if __name__ == '__main__':
    gpkg = GeoPackage()
    mp = MapBiomas()
    non_forest = mp.human_classes
    forest = mp.natural_classes
    gdf = gpkg.read_layer('bioma_lulc')
    gdf.head()
    start = [1985, 2000]
    middle = [2000, 2010]
    end = [2010, 2023]
    year_chunks = {
        0: start,
        1: middle,
        2: end
    }
    df_list = []
    for k, v in year_chunks.items():

        raster_1985 = f"data/ti_lulc_{v[0]}_reproject.tif"
        raster_2023 = f"data/ti_lulc_{v[1]}_reproject.tif"

        df_list.append(
            calc_forest_changes(
                raster_1985, 
                raster_2023, 
                gdf, 
                forest_list=forest, 
                non_forest_list=non_forest, 
                year_0=v[0], 
                year_f=v[1]
                )
            )
    king_biomas = pd.concat(df_list, ignore_index=True)
    gpkg.save_layer(king_biomas, 'bioma_net_change')
    # print(test.head())
    # gpkg.save_layer(test, 'bioma_lulc')



    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd

    # # Prepare data for grouped bar
    # df_plot = gdf[['bioma', 'forest_cover_1985_%', 'forest_cover_2023_%']].melt(
    #     id_vars='bioma',
    #     var_name='Ano',
    #     value_name='Forest Cover %'
    # )
    # # df_plot['Year'] = 0
    # # Rename for nicer labels
    # df_plot['Ano'] = df_plot['Ano'].replace({
    #     'forest_cover_1985_%': '1985',
    #     'forest_cover_2023_%': '2023'
    # })
    # sns.color_palette()
    # plt.figure(figsize=(8,5))
    # sns.barplot(
    #     data=df_plot, 
    #     x='bioma', 
    #     y='Forest Cover %', 
    #     hue='Ano',
    #     palette=["#77db56", "#1a8038"])
    # plt.title('Cobertura Florestal 1985 - 2023')
    # plt.ylabel('Cobertura Florestal (%)')
    # plt.xlabel('Bioma')
    # plt.legend(title='Ano')
    # plt.show()

    # plt.figure(figsize=(7,5))

    # for _, row in gdf.iterrows():
    #     plt.plot([1985, 2023],
    #             [row['forest_cover_1985_%'], row['forest_cover_2023_%']],
    #             marker='o', label=row['bioma'])

    # plt.title('Slope Graph: Forest Cover Change by Biome')
    # plt.ylabel('Forest Cover (%)')
    # plt.xticks([1985, 2023])
    # plt.legend()
    # plt.show()


    # import matplotlib.pyplot as plt
    # import numpy as np

    # # Example: assume your gdf already has regeneration_% and degeneration_% columns
    # # If not, you can calculate them (regen = gain / area *100, degen = loss / area *100)
    # # Here I just pull them out:
    # biomes = gdf['bioma']
    # regen = gdf['regeneration_%']
    # degen = gdf['degeneration_%']
    # net = gdf['net_change_%']

    # x = np.arange(len(biomes))

    # plt.figure(figsize=(8,5))

    # # Lines for regen and degen
    # plt.plot(x, regen, marker='o', color='green', label='Regeneração %')
    # plt.plot(x, degen, marker='o', color='red', label='Degeneração %')

    # # Bars for net change (can be negative)
    # plt.bar(x, net, color=['#2ca02c' if v >= 0 else '#d62728' for v in net],
    #         alpha=0.4, label='Net Change % (regen - degen)')

    # # X-axis
    # plt.xticks(x, biomes, rotation=20)
    # plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

    # plt.title('Forest Regeneration vs Degeneration vs Net Change by Biome')
    # plt.ylabel('Percentage (%)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
