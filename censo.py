import os
import zipfile
import psycopg2

import geopandas as gpd
import polars as pl

from sqlalchemy import create_engine
from functools import reduce

pg = create_engine('postgresql://postgres:postgres@143.54.25.37:5432/mapa_gis')

gdf = gpd.read_file(r'files\geo\BR_setores_CD2022.gpkg', layer='BR_setores_CD2022')
# print(gdf.head())

def get_range_cols(df: pl.DataFrame, start: str, end: str) -> list[str]:
    return [col for col in df.columns if start <= col <= end]

# Safe wrapper to avoid ComputeError if range is empty
def safe_sum_horizontal(df: pl.DataFrame, start: str, end: str) -> pl.Expr:
    cols = get_range_cols(df, start, end)
    return pl.sum_horizontal([pl.col(c) for c in cols]) if cols else pl.lit(0.0)


def unzip(file: str, folder: str):
    with zipfile.ZipFile(file, 'r') as f:
        f.extractall(folder)

if __name__ == '__main__':
    zip_folder = r'files\csv\zip'
    csv_folder = r'files\csv'
    lista_zip = os.listdir(zip_folder)
    if not os.listdir(csv_folder):
        for file in lista_zip:
            file = os.path.join(zip_folder, file)
            unzip(file, csv_folder)

    def rename_cod_setor(dfs: list[pl.DataFrame]):
        for i in range(1, len(dfs)):
            dfs[i] = dfs[i].rename({"CD_SETOR": f"CD_SETOR_{i}"})
        return dfs
    
    list_df: list[pl.DataFrame] = []
    for csv in os.listdir(csv_folder):
        csv = os.path.join(csv_folder, csv)
        if os.path.isfile(csv):
            df = pl.read_csv(csv, encoding='ISO-8859-1', separator=';', null_values=['.', 'X'])
            df = df.rename({df.columns[0]: df.columns[0].upper()})
            list_df.append(df)
    
    # list_df = rename_cod_setor(list_df)
    
    # list_df[:1] = [data.drop("CD_SETOR") for data in list_df[1:]] 

    merged_df = reduce(lambda  left, right: left.join(right, on="CD_SETOR", how="left", suffix=''), list_df)
    # print(merged_df.head())
    # print(merged_df.columns)
    # merged_df.write_parquet(r'D:\Vicente\dados_agregados_censo2022.parquet')
    # merged_df = merged_df.to_pandas()
    # merged_df.to_sql('dados_agregados_setores', con=pg.connect(), schema='lim_pol_ibge')
    # merged_gdf = gdf.merge(merged_df, how='outer', on='CD_SETOR')
    # merged_gdf.to_parquet('br_setores_all.parquet')
    # merged_df.to_csv('merged.csv')
    # Get a list of all V-prefixed columns that are digits (e.g., V0001, V0002...)
    merged_df = merged_df.rename({col: col.upper() for col in merged_df.columns})
    v_cols = [col for col in merged_df.columns if col.startswith('V') and col[1:].isdigit()]
    merged_df = merged_df.with_columns([
    pl.col("V0005")
    .cast(str)                      # make sure it's a string
    .str.replace(",", ".")         # replace , with .
    .cast(pl.Float64)              # convert to float
])
    merged_df = merged_df.with_columns([
    pl.col("V0006")
    .cast(str)                      # make sure it's a string
    .str.replace(",", ".")         # replace , with .
    .cast(pl.Float64)              # convert to float
])
    # Cast those columns to Float64
    merged_df = merged_df.with_columns([
        pl.col(col).cast(pl.Float64) for col in v_cols
    ])
    merged_df = merged_df.with_columns([
    pl.col("AREA_KM2")
    .cast(str)                      # make sure it's a string
    .str.replace(",", ".")         # replace , with .
    .cast(pl.Float64)              # convert to float
])
    merged_df = merged_df.with_columns()
    
    print(merged_df.columns)

    new_df = pl.DataFrame()

  # Create the new DataFrame
    new_df = merged_df.select([
        pl.col("AREA_KM2"),
        pl.col("V0001").alias("Pop_tot"),
        (pl.col("V0001") / pl.col("AREA_KM2")).alias("Dens_pop"),
        (pl.col("V0001") / pl.col("V0001") * 100).alias("r_sex"),

        (safe_sum_horizontal(merged_df, "V0023", "V0038") /
        safe_sum_horizontal(merged_df, "V0023", "V00134") * 100).alias("p_0-4"),

        (safe_sum_horizontal(merged_df, "V0023", "V0048") /
        safe_sum_horizontal(merged_df, "V0023", "V00134") * 100).alias("p_0-14"),

        (safe_sum_horizontal(merged_df, "V0049", "V0093") /
        safe_sum_horizontal(merged_df, "V0023", "V00134") * 100).alias("p_15-59"),

        (safe_sum_horizontal(merged_df, "V0094", "V00134") /
        safe_sum_horizontal(merged_df, "V0023", "V00134") * 100).alias("p_60+"),

        ((safe_sum_horizontal(merged_df, "V0023", "V0048") +
        safe_sum_horizontal(merged_df, "V0094", "V00134")) /
        safe_sum_horizontal(merged_df, "V0049", "V0093") * 100).alias("r_dep"),

        # pl.col("V0005").alias("V0005 (renda)"),

        (((safe_sum_horizontal(merged_df, "V0044", "V00134") -
        (pl.col("V0001") - safe_sum_horizontal(merged_df, "V0002", "V0006"))) /
        safe_sum_horizontal(merged_df, "V0044", "V00134")) * 100).alias("_tx_analf"),

        # rn_x_y
        # pl.col("V0001").alias("rn_0_05"),
        # pl.col("V0002").alias("rn_05_1"),
        # pl.col("V0003").alias("rn_1_2"),
        # pl.col("V0004").alias("rn_2_3"),
        # pl.col("V0005").alias("rn_3_5"),
        # pl.col("V0006").alias("rn_5_10"),
        # pl.col("V0007").alias("rn_10_15"),
        # pl.col("V0008").alias("rn_15_20"),
        # pl.col("V0009").alias("rn_20"),

        # race proportions
        (pl.col("V0002") / pl.col("V0001") * 100).alias("_p_branc"),
        (pl.col("V0003") / pl.col("V0001") * 100).alias("_p_pret"),
        (pl.col("V0004") / pl.col("V0001") * 100).alias("_p_amar"),
        (pl.col("V0005") / pl.col("V0001") * 100).alias("_p_pard"),
        (pl.col("V0006") / pl.col("V0001") * 100).alias("_p_ind"),
        ((pl.col("V0003") + pl.col("V0005")) / pl.col("V0001") * 100).alias("_p_par+pret"),
        (safe_sum_horizontal(merged_df, "V0002", "V0005") / pl.col("V0001") * 100).alias("_p_nao_ind"),

        # gender
        # pl.col("V0001").alias("homens"),
        # pl.col("V0001").alias("mulheres"),
        # (pl.col("V0001") / pl.col("V0001") * 100).alias("p_hom"),
        # (pl.col("V0001") / pl.col("V0001") * 100).alias("p_mul"),

        # age groups
        safe_sum_horizontal(merged_df, "V0023", "V0038").alias("a0_4"),
        safe_sum_horizontal(merged_df, "V0039", "V0043").alias("a5_9"),
        safe_sum_horizontal(merged_df, "V0044", "V0048").alias("a10_14"),
        safe_sum_horizontal(merged_df, "V0049", "V0053").alias("a15_19"),
        safe_sum_horizontal(merged_df, "V0054", "V0058").alias("a20_24"),
        safe_sum_horizontal(merged_df, "V0059", "V0063").alias("a25_29"),
        safe_sum_horizontal(merged_df, "V0064", "V0068").alias("a30_34"),
        safe_sum_horizontal(merged_df, "V0069", "V0073").alias("a35_39"),
        safe_sum_horizontal(merged_df, "V0074", "V0078").alias("a40_44"),
        safe_sum_horizontal(merged_df, "V0079", "V0083").alias("a45_49"),
        safe_sum_horizontal(merged_df, "V0084", "V0088").alias("a50_54"),
        safe_sum_horizontal(merged_df, "V0089", "V0093").alias("a55_59"),
        safe_sum_horizontal(merged_df, "V0094", "V00103").alias("a60_69"),
        safe_sum_horizontal(merged_df, "V00104", "V00113").alias("a70_79"),
        safe_sum_horizontal(merged_df, "V00114", "V00134").alias("a80_100"),
    ])
    
    new_df.write_csv('merge.csv')
    # new_df.write_database('lim_pol_ibge.dados_tratados_censo2022', connection=pg.connect())