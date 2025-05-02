import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import geopandas as gpd

# Create synthetic dataset for Brazilian municipalities
def create_sample_data():
    """Create a sample dataset with indicators for Exposure, Sensitivity, and Adaptive Capacity"""
    np.random.seed(42)
    
    municipalities = [f'Municipality_{i}' for i in range(1, 101)]
    states = ['RS', 'SC', 'PR', 'SP', 'RJ', 'MG', 'ES', 'BA', 'SE', 'AL'] * 10
    
    data = {
        'municipality': municipalities,
        'state': states,
        # Exposure indicators (higher = more exposed)
        'temperature_change': np.random.uniform(0.5, 3.0, 100),  # °C increase
        'precipitation_change': np.random.uniform(-30, 30, 100), # % change
        'flood_risk': np.random.uniform(0, 1, 100),              # 0-1 scale
        'drought_frequency': np.random.uniform(0, 5, 100),       # events/year
        
        # Sensitivity indicators (higher = more sensitive)
        'poverty_rate': np.random.uniform(5, 40, 100),           # %
        'elderly_population': np.random.uniform(5, 25, 100),     # %
        'agriculture_dependency': np.random.uniform(0, 0.8, 100),# proportion
        'infrastructure_quality': np.random.uniform(0, 1, 100),  # 0-1 scale (lower = worse)
        
        # Adaptive Capacity indicators (higher = better capacity)
        'education_index': np.random.uniform(0.4, 0.9, 100),     # 0-1 scale
        'healthcare_access': np.random.uniform(0.3, 0.95, 100),  # 0-1 scale
        'gdp_per_capita': np.random.uniform(10000, 50000, 100),  # R$
        'governance_index': np.random.uniform(0.3, 0.9, 100)     # 0-1 scale
    }
    
    return pd.DataFrame(data)

# Normalization functions
def normalize_indicators(df, indicators, method='minmax', direction='positive'):
    """
    Normalize indicators to 0-1 scale
    direction: 'positive' if higher values are better, 'negative' if higher values are worse
    """
    normalized_df = df.copy()
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    
    for indicator in indicators:
        if direction == 'negative':
            # Invert so higher values always mean worse outcome for vulnerability
            normalized_df[indicator] = 1 - scaler.fit_transform(df[[indicator]].values.reshape(-1, 1))
        else:
            normalized_df[indicator] = scaler.fit_transform(df[[indicator]].values.reshape(-1, 1))
    
    return normalized_df

def calculate_ipcc_vulnerability(df, weights=None):
    """
    Calculate IPCC Vulnerability Index
    V = (Exposure + Sensitivity) - Adaptive Capacity
    """
    if weights is None:
        # Default equal weights - YOU SHOULD ADJUST THESE BASED ON EXPERT KNOWLEDGE
        weights = {
            'exposure': {'temperature_change': 0.25, 'precipitation_change': 0.25, 
                        'flood_risk': 0.25, 'drought_frequency': 0.25},
            'sensitivity': {'poverty_rate': 0.25, 'elderly_population': 0.25,
                           'agriculture_dependency': 0.25, 'infrastructure_quality': 0.25},
            'adaptive_capacity': {'education_index': 0.25, 'healthcare_access': 0.25,
                                 'gdp_per_capita': 0.25, 'governance_index': 0.25}
        }
    
    # Calculate component scores
    df['exposure_score'] = sum(df[indicator] * weight for indicator, weight 
                              in weights['exposure'].items())
    df['sensitivity_score'] = sum(df[indicator] * weight for indicator, weight 
                                 in weights['sensitivity'].items())
    df['adaptive_capacity_score'] = sum(df[indicator] * weight for indicator, weight 
                                      in weights['adaptive_capacity'].items())
    
    # Calculate vulnerability (normalized to 0-1)
    df['vulnerability_raw'] = (df['exposure_score'] + df['sensitivity_score']) - df['adaptive_capacity_score']
    df['vulnerability_index'] = MinMaxScaler().fit_transform(df[['vulnerability_raw']])
    
    return df

# Main analysis
def run_ipcc_vulnerability_analysis():
    # Create sample data
    df = create_sample_data()
    
    # Normalize indicators (ensure higher values = worse outcome for vulnerability)
    exposure_indicators = ['temperature_change', 'precipitation_change', 'flood_risk', 'drought_frequency']
    sensitivity_indicators = ['poverty_rate', 'elderly_population', 'agriculture_dependency', 'infrastructure_quality']
    adaptive_capacity_indicators = ['education_index', 'healthcare_access', 'gdp_per_capita', 'governance_index']
    
    df_normalized = normalize_indicators(df, exposure_indicators, direction='negative')
    df_normalized = normalize_indicators(df_normalized, sensitivity_indicators, direction='negative')
    df_normalized = normalize_indicators(df_normalized, adaptive_capacity_indicators, direction='positive')
    
    # Calculate vulnerability
    df_results = calculate_ipcc_vulnerability(df_normalized)
    
    # Classify vulnerability levels
    df_results['vulnerability_category'] = pd.cut(df_results['vulnerability_index'],
                                                 bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
                                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    return df_results

# Visualization functions
def plot_vulnerability_results(df_results):
    """Create visualization of vulnerability results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Vulnerability distribution
    axes[0,0].hist(df_results['vulnerability_index'], bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[0,0].set_title('Distribution of Vulnerability Index')
    axes[0,0].set_xlabel('Vulnerability Index (0-1)')
    axes[0,0].set_ylabel('Frequency')
    
    # Component scores
    components = ['exposure_score', 'sensitivity_score', 'adaptive_capacity_score']
    component_data = [df_results[comp] for comp in components]
    axes[0,1].boxplot(component_data, labels=['Exposure', 'Sensitivity', 'Adaptive Capacity'])
    axes[0,1].set_title('Distribution of Component Scores')
    axes[0,1].set_ylabel('Score (0-1)')
    
    # Vulnerability by state
    state_vulnerability = df_results.groupby('state')['vulnerability_index'].mean().sort_values()
    axes[1,0].barh(state_vulnerability.index, state_vulnerability.values)
    axes[1,0].set_title('Average Vulnerability by State')
    axes[1,0].set_xlabel('Vulnerability Index')
    
    # Category distribution
    category_counts = df_results['vulnerability_category'].value_counts()
    axes[1,1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    axes[1,1].set_title('Vulnerability Categories Distribution')
    
    plt.tight_layout()
    plt.show()

# Run the analysis
if __name__ == "__main__":
    results = run_ipcc_vulnerability_analysis()
    
    # Display top 10 most vulnerable municipalities
    print("Top 10 Most Vulnerable Municipalities:")
    print(results.nlargest(10, 'vulnerability_index')[['municipality', 'state', 'vulnerability_index', 'vulnerability_category']])
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean Vulnerability: {results['vulnerability_index'].mean():.3f}")
    print(f"Std Deviation: {results['vulnerability_index'].std():.3f}")
    print(f"Minimum: {results['vulnerability_index'].min():.3f}")
    print(f"Maximum: {results['vulnerability_index'].max():.3f}")
    
    # Plot results
    plot_vulnerability_results(results)
    
    # Save results
    results.to_csv('ipcc_vulnerability_assessment.csv', index=False)
    print("\nResults saved to 'ipcc_vulnerability_assessment.csv'")