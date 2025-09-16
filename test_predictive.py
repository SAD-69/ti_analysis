import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from libpysal.weights import DistanceBand
from esda.moran import Moran, Moran_Local
from spreg import OLS, ML_Lag, ML_Error
import pandas as pd
from models.gpkg import GeoPackage
from pprint import pprint
# Configuração dos plots
plt.style.use('default')

# Load data
gpkg = GeoPackage()
gdf = gpkg.read_layer('vindex_1985_2023_differential')
y_col = 'txalfabeti_imputed'
cols = ["ex_score", "se_score"]

# Variáveis dependente (y) e independentes (X)
y = gdf[y_col].values.reshape(-1,1)
X = gdf[cols].values

# Criar matriz de pesos espaciais
w = DistanceBand.from_dataframe(gdf, threshold=40000, silence_warnings=True)
w.transform = "r"

# 1. Rodar OLS primeiro
ols = OLS(y, X, name_y=y_col, name_x=cols)

# 2. Testar autocorrelação espacial nos resíduos
w.plot(gdf)
moran_res = Moran(ols.u, w)

moran_loc = Moran_Local(y, w)
print(moran_loc.p_sim)
print(f"Moran I: {moran_res.I}")
# moran_res.plot_scatter()
moran_loc.plot_combination(gdf, 'vulnerability_index')
# 3. Selecionar o melhor modelo
if moran_res.p_sim < 0.05:
    # Rodar modelos espaciais
    sar = ML_Lag(y, X, w=w, name_y=y_col, name_x=cols)
    sem = ML_Error(y, X, w=w, name_y=y_col, name_x=cols)
    
    # Comparar por AIC
    aic_values = {'OLS': ols.aic, 'SAR': sar.aic, 'SEM': sem.aic}
    pprint(aic_values)
    best_model_name = min(aic_values, key=aic_values.get)
    
    if best_model_name == 'SAR':
        best_model = sar
        model_type = 'SAR'
    elif best_model_name == 'SEM':
        best_model = sem
        model_type = 'SEM'
    else:
        best_model = ols
        model_type = 'OLS'
else:
    best_model = ols
    model_type = 'OLS'

print(f"Melhor modelo selecionado: {model_type}")
print(f"AIC: {best_model.aic:.2f}")

def predict_future_years(model: OLS | ML_Lag | ML_Error, X_future, model_type='OLS', w=None, gdf=None):
    """
    Faz previsões para anos futuros usando o modelo selecionado
    
    Parameters:
    model: modelo treinado (OLS, ML_Lag, ou ML_Error)
    X_future: array com valores das variáveis independentes para anos futuros
    model_type: tipo do modelo ('OLS', 'SAR', 'SEM')
    w: matriz de pesos espaciais (necessária para modelos espaciais)
    gdf: GeoDataFrame original (necessário para modelos espaciais)
    
    Returns:
    predictions: array com as previsões
    """
    
    if model_type == 'OLS':
        # Para OLS: y_pred = Xβ
        predictions = np.dot(X_future, model.betas[1:]) + model.betas[0]
    
    elif model_type == 'SAR':
        # Para SAR: y = ρWy + Xβ + ε
        # Precisamos resolver iterativamente ou usar matriz inversa
        n = X_future.shape[0]
        I = np.eye(n)
        rho = model.betas[-1]  # coeficiente espacial
        beta = model.betas[:-1]  # outros coeficientes
        
        # Termo Xβ
        XB = np.dot(X_future, beta[1:]) + beta[0]
        
        # Para previsão, assumimos que a estrutura espacial se mantém
        # Esta é uma aproximação simplificada
        predictions = np.linalg.solve(I - rho * w.full()[0], XB)
    
    elif model_type == 'SEM':
        # Para SEM: y = Xβ + u, u = λWu + ε
        # Esta implementação é mais complexa e pode requerer aproximações
        beta = model.betas[:-1]  # coeficientes das variáveis
        lambda_val = model.betas[-1]  # coeficiente espacial do erro
        
        # Previsão direta (aproximada)
        predictions = np.dot(X_future, beta[1:]) + beta[0]
    
    return predictions

# 4. Gerar cenários futuros (exemplo)
# Supondo que queremos prever para os próximos 5 anos
future_years = 5

# Aqui você precisa fornecer os valores futuros das variáveis independentes
# Este é um exemplo - você precisa obter dados reais ou criar cenários
X_future = np.array([
    [X[-1, 0] * 1.05, X[-1, 1] * 1.02],  # ano 1: +5% e +2%
    [X[-1, 0] * 1.08, X[-1, 1] * 1.04],  # ano 2: +8% e +4%
    [X[-1, 0] * 1.12, X[-1, 1] * 1.06],  # ano 3: +12% e +6%
    [X[-1, 0] * 1.15, X[-1, 1] * 1.08],  # ano 4: +15% e +8%
    [X[-1, 0] * 1.18, X[-1, 1] * 1.10],  # ano 5: +18% e +10%
])

# Fazer previsões
predictions = predict_future_years(best_model, X_future, model_type, w, gdf)

# 5. Visualizar resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Previsões futuras
years = list(range(1, future_years + 1))
ax1.plot(years, predictions, 'o-', linewidth=2, markersize=8, color='red')
ax1.set_xlabel('Anos Futuros')
ax1.set_ylabel('Índice de Vulnerabilidade Previsto')
ax1.set_title(f'Previsões para os próximos {future_years} anos\n(Modelo: {model_type})')
ax1.grid(True, alpha=0.3)

# Adicionar valores nos pontos
for i, (x, y_val) in enumerate(zip(years, predictions)):
    ax1.annotate(f'{y_val[0]:.2f}', (x, y_val), textcoords="offset points", 
                xytext=(0,10), ha='center')

# Gráfico 2: Comparação com dados históricos
historical_years = list(range(len(y) - 10, len(y)))  # últimos 10 anos
historical_values = y[historical_years].flatten()

ax2.plot(historical_years, historical_values, 'o-', linewidth=2, 
         markersize=6, color='blue', label='Histórico')
ax2.plot([len(y)] + list(range(len(y), len(y) + future_years)), 
         [y[-1]] + list(predictions), 'o--', linewidth=2, 
         markersize=8, color='red', label='Previsão')
ax2.set_xlabel('Período')
ax2.set_ylabel('Índice de Vulnerabilidade')
ax2.set_title('Comparação: Histórico vs Previsão')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Criar DataFrame com resultados
results_df = pd.DataFrame({
    'Ano': [f'Ano +{i+1}' for i in range(future_years)],
    'degeneration_%_ex': X_future[:, 0],
    'degeneration_%_se': X_future[:, 1],
    'vulnerability_index_pred': predictions[:, 0]
})

print("\n" + "="*50)
print("RESULTADOS DAS PREVISÕES")
print("="*50)
print(f"Modelo utilizado: {model_type}")
print(f"AIC do modelo: {best_model.aic:.2f}")
print("\nPrevisões:")
print(results_df.to_string(index=False))

# 7. Salvar resultados (opcional)
# results_df.to_csv('previsoes_vulnerabilidade.csv', index=False)
print(f"\nResultados salvos em 'previsoes_vulnerabilidade.csv'")