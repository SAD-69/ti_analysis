import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
import contextily as ctx

def plot_gwr_results(gdf, gwr_model, variable_names, dep_var, figsize=(18, 14), add_basemap=True, bw = None):
    """
    Função para plotar resultados do modelo GWR com basemap do OpenStreetMap.
    """
    # Configuração do estilo dos plots
    plt.style.use('default')
    
    # Verificar se o GeoDataFrame está em coordenadas geográficas
    if gdf.crs is None:
        print("Aviso: O GeoDataFrame não tem CRS definido. Definindo como WGS84 (EPSG:4326).")
        gdf = gdf.set_crs(epsg=4326)
    
    # Reprojetar para Web Mercator (EPSG:3857) para compatibilidade com contextily
    gdf_web_mercator = gdf.to_crs(epsg=3857)
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Resultados da Regressão Geograficamente Ponderada (GWR)', fontsize=16, fontweight='bold')
    
    # 1. MAPA DOS COEFICIENTES PARA UMA VARIÁVEL EXEMPLAR
    if gwr_model.params.shape[1] > 1:
        coef_var = gwr_model.params[:, 1]  # Coeficientes para a primeira variável (após intercepto)
    else:
        coef_var = gwr_model.params[:, 0]  # Apenas intercepto
    
    gdf_web_mercator['coef_var1'] = coef_var
    
    ax1 = axes[0, 0]
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    # Plotar os dados
    gdf_web_mercator.plot(column='coef_var1', ax=ax1, legend=True, cax=cax,
                         cmap='RdYlBu_r', edgecolor='black', linewidth=0.8,
                         alpha=0.8, legend_kwds={'label': 'Coeficiente'})
    
    # Adicionar basemap
    if add_basemap:
        try:
            ctx.add_basemap(ax1, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
        except Exception as e:
            print(f"Erro ao adicionar basemap: {e}")
    
    ax1.set_title('Coeficiente da Variável', fontweight='bold')
    ax1.set_axis_off()
    
    # 2. MAPA DOS VALORES AJUSTADOS (Y hat)
    gdf_web_mercator['y_pred'] = gwr_model.predy
    ax2 = axes[0, 1]
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    gdf_web_mercator.plot(column='y_pred', ax=ax2, legend=True, cax=cax,
                         cmap='viridis', edgecolor='black', linewidth=0.8,
                         alpha=0.8, legend_kwds={'label': 'Valores Preditos'})
    
    if add_basemap:
        try:
            ctx.add_basemap(ax2, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
        except:
            pass
    
    ax2.set_title('Valores Preditos (Ŷ)', fontweight='bold')
    ax2.set_axis_off()
    
    # 3. MAPA DOS RESÍDUOS
    gdf_web_mercator['residuals'] = gwr_model.resid_response
    ax3 = axes[0, 2]
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    gdf_web_mercator.plot(column='residuals', ax=ax3, legend=True, cax=cax,
                         cmap='RdBu_r', edgecolor='black', linewidth=0.8,
                         alpha=0.8, legend_kwds={'label': 'Resíduos'})
    
    if add_basemap:
        try:
            ctx.add_basemap(ax3, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
        except:
            pass
    
    ax3.set_title('Resíduos do Modelo', fontweight='bold')
    ax3.set_axis_off()
    
    # 4. MAPA DO R² LOCAL
    try:
        local_r2 = gwr_model.localR2
    except AttributeError:
        try:
            # Tentar método alternativo para obter R² local
            y_mean = np.mean(gwr_model.predy)
            SST_local = np.sum((gdf[dep_var].values - y_mean)**2)
            SSE_local = np.sum(gwr_model.resid_response**2)
            local_r2 = 1 - (SSE_local / SST_local)
            local_r2 = np.full(gwr_model.n, local_r2)  # Criar array com valor único
        except:
            local_r2 = np.full(gwr_model.n, gwr_model.R2)  # Usar R² global
    
    gdf_web_mercator['local_r2'] = local_r2
    ax4 = axes[1, 0]
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    gdf_web_mercator.plot(column='local_r2', ax=ax4, legend=True, cax=cax,
                         cmap='RdYlGn', vmin=0, vmax=1, edgecolor='black', linewidth=0.8,
                         alpha=0.8, legend_kwds={'label': 'R² Local'})
    
    if add_basemap:
        try:
            ctx.add_basemap(ax4, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
        except:
            pass
    
    ax4.set_title('R² Local', fontweight='bold')
    ax4.set_axis_off()
    
    # 5. MAPA DA INFLUÊNCIA LOCAL (leverage)
    try:
        influence = gwr_model.influ
        gdf_web_mercator['influence'] = influence
        ax5 = axes[1, 1]
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        gdf_web_mercator.plot(column='influence', ax=ax5, legend=True, cax=cax,
                             cmap='OrRd', edgecolor='black', linewidth=0.8,
                             alpha=0.8, legend_kwds={'label': 'Influência'})
        
        if add_basemap:
            try:
                ctx.add_basemap(ax5, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.7)
            except:
                pass
        
        ax5.set_title('Influência Local (Leverage)', fontweight='bold')
        ax5.set_axis_off()
    except AttributeError:
        ax5 = axes[1, 1]
        ax5.text(0.5, 0.5, 'Informação de influência\nnão disponível', 
                ha='center', va='center', fontsize=12, transform=ax5.transAxes)
        ax5.set_title('Influência Local', fontweight='bold')
        ax5.set_axis_off()
    
    # 6. ESTATÍSTICAS DO MODELO E BANDWIDTH
    ax6 = axes[1, 2]
    
    try:
        bandwidth = bw
    except:
        bandwidth = 'N/A'
    
    model_stats = f"""
    Estatísticas do Modelo:
    -----------------------
    R² Global: {gwr_model.R2:.3f}
    R² Ajustado: {getattr(gwr_model, 'adj_R2', 'N/A'):.3f}
    AICc: {getattr(gwr_model, 'aicc', 'N/A'):.2f}
    Bandwidth: {bandwidth}
    Nº Observações: {gwr_model.n}
    """
    
    ax6.text(0.5, 0.5, model_stats, ha='center', va='center', 
             fontsize=10, transform=ax6.transAxes, 
             bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'))
    
    ax6.set_title('Estatísticas do Modelo', fontweight='bold')
    ax6.set_axis_off()
    
    plt.tight_layout()
    plt.show()
    
    # Plotar coeficientes para todas as variáveis com basemap
    plot_all_coefficients(gdf, gwr_model, variable_names, add_basemap)

def plot_all_coefficients(gdf, gwr_model, variable_names, add_basemap=True):
    """
    Plota mapas de coeficientes para todas as variáveis do modelo GWR com basemap.
    """
    # Reprojetar para Web Mercator
    gdf_web_mercator = gdf.to_crs(epsg=3857)
    
    n_vars = gwr_model.params.shape[1]  # Número de variáveis baseado nos parâmetros
    n_cols = min(3, n_vars)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    fig.suptitle('Coeficientes GWR por Variável com OpenStreetMap', fontsize=16, fontweight='bold')
    
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    
    axes = axes.flatten()
    
    for i in range(n_vars):
        if i >= len(axes):
            break
            
        ax = axes[i]
        coef_values = gwr_model.params[:, i]
        
        gdf_temp = gdf_web_mercator.copy()
        gdf_temp['coef'] = coef_values
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        # Plotar coeficientes
        var_name = variable_names[i] if i < len(variable_names) else f'Var_{i}'
        plot = gdf_temp.plot(column='coef', ax=ax, legend=True, cax=cax,
                           cmap='RdYlBu_r', edgecolor='black', linewidth=0.8,
                           alpha=0.8, legend_kwds={'label': f'Coef: {var_name}'})
        
        # Adicionar basemap
        if add_basemap:
            try:
                ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, alpha=0.6)
            except Exception as e:
                print(f"Erro ao adicionar basemap: {e}")
        
        ax.set_title(f'Coeficiente: {var_name}', fontweight='bold')
        ax.set_axis_off()
        
        # Adicionar estatísticas descritivas
        stats_text = f'Média: {coef_values.mean():.3f}\nStd: {coef_values.std():.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Ocultar eixos extras
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def select_bandwidth_safe(coords, y, X, kernel='bisquare', criterion='AICc'):
    """
    Função segura para seleção de bandwidth que funciona em diferentes versões do mgwr.
    """
    from mgwr.sel_bw import Sel_BW
    
    try:
        selector = Sel_BW(coords, y, X, kernel=kernel)
        
        # Tentar diferentes abordagens para evitar o IndexError
        try:
            # Método 1: search() sem parâmetros
            bw = selector.search()
        except (IndexError, TypeError):
            try:
                # Método 2: search() com verbose=False
                bw = selector.search(verbose=False)
            except (IndexError, TypeError):
                try:
                    # Método 3: search() com número fixo de iterações
                    bw = selector.search(search_method='golden_section')
                except:
                    # Método 4: Usar bandwidth fixo como fallback
                    print("Usando bandwidth fixo como fallback")
                    n = len(y)
                    bw = int(n * 0.2)  # 20% das observações
        
        # Verificar se bw é uma tupla e extrair o valor
        if isinstance(bw, tuple):
            bw = bw[0]
            
        print(f"Bandwidth selecionado: {bw}")
        return bw
        
    except Exception as e:
        print(f"Erro na seleção de bandwidth: {e}")
        # Fallback para bandwidth fixo
        n = len(y)
        bw = max(5, int(n * 0.15))  # Mínimo 5, máximo 15% das observações
        print(f"Usando bandwidth fixo: {bw}")
        return bw

def run_gwr_safe(coords, y, X, fixed_bw=None):
    """
    Função segura para executar GWR com tratamento de erros.
    """
    from mgwr.gwr import GWR
    
    try:
        if fixed_bw is not None:
            # Usar bandwidth fixo se fornecido
            bw = fixed_bw
        else:
            # Selecionar bandwidth automaticamente
            bw = select_bandwidth_safe(coords, y, X)
        
        # Ajustar modelo GWR
        gwr_model = GWR(coords, y, X, bw, fixed=True if fixed_bw else False).fit()
        
        return gwr_model, bw
        
    except Exception as e:
        print(f"Erro ao ajustar modelo GWR: {e}")
        raise

def run_complete_gwr_analysis(gdf, dependent_var, independent_vars, coords, add_basemap=True, fixed_bw=None):
    """
    Função completa para executar análise GWR e visualizar resultados com basemap.
    """
    # Verificar e definir CRS se necessário
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
        print("CRS definido como WGS84 (EPSG:4326)")
    
    # Preparar dados
    y = gdf[dependent_var].values.reshape(-1, 1)
    X = gdf[independent_vars].values
    
    # Executar GWR com tratamento de erros
    gwr_model, bw = run_gwr_safe(coords, y, X, fixed_bw)
    
    # Nomes das variáveis
    variable_names = ['Intercepto'] + independent_vars
    
    print(f"\n=== RESULTADOS DO MODELO GWR ===")
    print(f"Bandwidth: {bw}")
    print(f"R²: {gwr_model.R2:.3f}")
    print(f"R² Ajustado: {getattr(gwr_model, 'adj_R2', 'N/A'):.3f}")
    print(f"AICc: {getattr(gwr_model, 'aicc', 'N/A'):.2f}")
    
    # Plotar resultados
    plot_gwr_results(gdf, gwr_model, variable_names, dependent_var, add_basemap=add_basemap, bw=bw)
    
    return gwr_model, bw

# Exemplo de uso CORRETO:
def example_usage():
    """
    Exemplo de como usar as funções corretamente.
    """
    # Supondo que você tenha seus dados:
    # gdf = GeoDataFrame com seus dados espaciais
    # coords = np.array([[x1, y1], [x2, y2], ...])  # Coordenadas dos pontos
    
    # Método 1: Bandwidth automático
    try:
        gwr_model, bw = run_complete_gwr_analysis(
            gdf=gdf,
            dependent_var='nome_variavel_dependente',
            independent_vars=['var1', 'var2', 'var3'],
            coords=coords,
            add_basemap=True
        )
    except Exception as e:
        print(f"Erro no método automático: {e}")
        
        # Método 2: Bandwidth fixo (fallback)
        print("Tentando com bandwidth fixo...")
        try:
            gwr_model, bw = run_complete_gwr_analysis(
                gdf=gdf,
                dependent_var='nome_variavel_dependente',
                independent_vars=['var1', 'var2', 'var3'],
                coords=coords,
                add_basemap=True,
                fixed_bw=50  # Bandwidth fixo
            )
        except Exception as e2:
            print(f"Erro também com bandwidth fixo: {e2}")
            return None

# Instalação das dependências necessárias
# pip install contextily mgwr geopandas matplotlib numpy

if __name__ == '__main__':
    from models.gpkg import GeoPackage
    gpkg = GeoPackage()
    gdf = gpkg.read_layer('vindex_1985_2023_entrega')
    gdf.geometry = gdf.geometry.buffer(gdf.dist_buf)
    gdf['soybean'] = gdf['soybean_ex'] + gdf['soybean_se']
    # gdf = gdf[gdf['vulnerability_index'] > 0.5]
    coords = np.array(list(zip(gdf.centroid.x, gdf.centroid.y)))
    run_complete_gwr_analysis(gdf, 'vulnerability_index', ['est_fundiaria', 'soybean'], coords)