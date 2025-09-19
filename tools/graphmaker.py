import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from matplotlib import rcParams
from geopandas import GeoDataFrame
from functools import cached_property

plt.style.use('default')
rcParams['font.family'] = 'DejaVu Sans'
sns.set_palette("husl")

class GraphMaker:
    def __init__(self, 
            gdf: GeoDataFrame, 
            columns: list[str] = [
                     'ex_score', 
                     'se_score', 
                     'ca_score', 
                     'vulnerability_index'
                     ],
            group: str = 'etnia_nome',
            region: str = 'bioma'
        ):
        self.gdf = gdf
        self.cols = columns
        self.group = group
        self.region = region

    def plot_etnia(self):
        available_cols = [col for col in self.cols if col in self.gdf.columns]
        if not available_cols:
            raise ValueError("Nenhuma coluna de vulnerabilidade encontrada no GeoDataFrame")
        
        # Criar figura com subplots
        n_cols = len(available_cols)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        # 1. Gráfico de barras com médias por etnia
        for i, col in enumerate(available_cols):
            if i >= len(axes):
                break
                
            # Calcular estatísticas por etnia
            stats_by_ethnicity = self.gdf.groupby(self.group)[col].agg([
                'mean', 'std', 'count', 'min', 'max', 'median'
            ]).round(3)
            
            # Ordenar por média (maior para menor)
            stats_by_ethnicity = stats_by_ethnicity.sort_values('mean', ascending=False)
            
            # Plotar gráfico de barras
            bars = axes[i].bar(stats_by_ethnicity.index, stats_by_ethnicity['mean'],
                            yerr=stats_by_ethnicity['std'], 
                            capsize=5, alpha=0.8,
                            color=plt.cm.Set3(np.arange(len(stats_by_ethnicity))))
            
            axes[i].set_title(f'{col.upper()} - Média por Etnia\n(Ordenado por maior vulnerabilidade)')
            axes[i].set_ylabel('Valor Médio')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Adicionar valores nas barras
            for j, (idx, row) in enumerate(stats_by_ethnicity.iterrows()):
                axes[i].text(j, row['mean'] + row['std'] + 0.01, 
                        f'{row["mean"]:.3f}\n(n={row["count"]})', 
                        ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # 2. Boxplots detalhados por etnia
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, col in enumerate(available_cols):
            if i >= len(axes):
                break
                
            # Boxplot por etnia_nome
            boxplot_data = []
            ethnicities = []
            
            # Ordenar etnias pela mediana
            ethnicity_medians =self.gdf.groupby(self.group)[col].median().sort_values(ascending=False)
            
            for ethnicity in ethnicity_medians.index:
                data = self.gdf[self.gdf[self.group] == ethnicity][col].dropna()
                if len(data) > 0:
                    boxplot_data.append(data)
                    ethnicities.append(ethnicity)
            
            box = axes[i].boxplot(boxplot_data, labels=ethnicities, patch_artist=True)
            
            # Colorir boxes
            colors = plt.cm.Set3(np.arange(len(ethnicities)))
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
            
            axes[i].set_title(f'{col.upper()} - Distribuição por Etnia\n(Ordenado por mediana)')
            axes[i].set_ylabel('Valor')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Adicionar informação de sample size
            for j, ethnicity in enumerate(ethnicities):
                n = len(self.gdf[self.gdf[self.group] == ethnicity])
                axes[i].text(j + 1, axes[i].get_ylim()[0] - 0.05, 
                        f'n={n}', ha='center', va='top', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # 3. Heatmap de correlações entre componentes por etnia
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Heatmap 1: Médias por etnia
        ethnicity_means =self.gdf.groupby(self.group)[available_cols].mean()
        sns.heatmap(ethnicity_means, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                    center=ethnicity_means.values.mean(), ax=axes[0])
        axes[0].set_title('Médias de Vulnerabilidade por Etnia')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)
        
        # Heatmap 2: Contagem por etnia
        ethnicity_counts =self.gdf.groupby([self.group, self.group]).size().unstack(fill_value=0)
        sns.heatmap(ethnicity_counts, annot=True, fmt='.0f', cmap='Blues', ax=axes[1])
        axes[1].set_title('Número de Observações por Etnia')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        # 4. Ranking detalhado
        print("=" * 80)
        print("RANKING DETALHADO POR ETNIA")
        print("=" * 80)
        
        for col in available_cols:
            print(f"\n{col.upper()} - Ranking por Etnia:")
            print("-" * 40)
            
            stats =self.gdf.groupby(self.group)[col].agg(['mean', 'std', 'count', 'median']).round(4)
            stats = stats.sort_values('mean', ascending=False)
            
            for i, (ethnicity, row) in enumerate(stats.iterrows(), 1):
                rank = f"{i}º"
                print(f"{rank:4} {ethnicity:20}: {row['mean']:.4f} ± {row['std']:.4f} "
                    f"(mediana: {row['median']:.4f}, n={row['count']})")
        
        return stats

