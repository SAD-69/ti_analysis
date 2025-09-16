import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import warnings
warnings.filterwarnings('ignore')

# Set the style for better visualizations
plt.style.use('default')

class LandUseVisualizer:
    def __init__(self, df_1985_2000, df_2000_2010, df_2010_2023, df_1985_2023):
        """
        Initialize with the four dataframes
        """
        self.df_periods = {
            '1985-2000': df_1985_2000,
            '2000-2010': df_2000_2010,
            '2010-2023': df_2010_2023,
            '1985-2023': df_1985_2023
        }
        
        # Define your specific color mapping
        self.land_cover_colors = {
            'forest_formation': '#006400',    # Dark green
            'forest_plantation': '#D3D3D3',   # Light gray
            'grassland': '#90EE90',           # Light green
            'soybean': '#FF0000',             # Red
            'urban_area': '#FFC0CB',          # Pink
            'rice': '#000000',                # Black
            'wetland': '#00008B'              # Dark blue
        }
        
        # List of columns to keep (your specified classes)
        self.cols_to_keep = [
            'forest_formation',
            'forest_plantation',
            'grassland',
            'soybean',
            'urban_area',
            'rice',
            'wetland'
        ]
        
        # Identify numeric columns from the specified list
        self.numeric_columns = self._identify_numeric_columns()
        print(f"Found {len(self.numeric_columns)} numeric columns: {list(self.numeric_columns)}")
        
    def _identify_numeric_columns(self):
        """Identify columns that contain numeric data from the specified list"""
        numeric_cols = []
        
        for col in self.cols_to_keep:
            found_in_all = True
            for period, df in self.df_periods.items():
                if col not in df.columns:
                    print(f"Warning: Column '{col}' not found in period '{period}'")
                    found_in_all = False
                    break
                try:
                    pd.to_numeric(df[col], errors='raise')
                except (ValueError, TypeError):
                    print(f"Warning: Column '{col}' in period '{period}' contains non-numeric data")
                    found_in_all = False
                    break
            
            if found_in_all:
                numeric_cols.append(col)
        
        return numeric_cols
    
    def _get_numeric_data(self, df):
        """Extract only specified numeric columns from dataframe"""
        numeric_cols = [col for col in self.numeric_columns if col in df.columns]
        return df[numeric_cols]
    
    def plot_stacked_area(self):
        """
        Plot stacked area chart showing composition over time with your specified colors
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Prepare data
        periods = ['1985-2000', '2000-2010', '2010-2023']
        
        # Calculate average percentages using only specified numeric data
        avg_data = []
        for period in periods:
            numeric_df = self._get_numeric_data(self.df_periods[period])
            avg_data.append(numeric_df.mean().values)
        
        land_cover_types = self._get_numeric_data(self.df_periods['1985-2000']).columns
        
        # Create color list in the order of the columns
        colors = [self.land_cover_colors[lc_type] for lc_type in land_cover_types]
        
        avg_df = pd.DataFrame(avg_data, index=periods, columns=land_cover_types)
        
        # Create stacked area plot with your specified colors
        ax.stackplot(range(len(periods)), avg_df.T, labels=land_cover_types, 
                    colors=colors, alpha=0.85)
        
        ax.set_title('Land Use/Cover Composition Over Time (1985-2023)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Percentage Composition (%)', fontsize=12)
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, fontsize=11)
        
        # Create legend with color boxes
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10,
                 frameon=True, fancybox=True, shadow=True)
        
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)  # Set y-axis to percentage scale
        plt.tight_layout()
        
        return fig
    
    def plot_stacked_area_with_percentage_labels(self):
        """
        Enhanced stacked area plot with percentage labels
        """
        fig, ax = plt.subplots(figsize=(18, 12))
        
        # Prepare data
        periods = ['1985-2000', '2000-2010', '2010-2023']
        
        # Calculate average percentages
        avg_data = []
        for period in periods:
            numeric_df = self._get_numeric_data(self.df_periods[period])
            avg_data.append(numeric_df.mean().values)
        
        land_cover_types = self._get_numeric_data(self.df_periods['1985-2000']).columns
        colors = [self.land_cover_colors[lc_type] for lc_type in land_cover_types]
        
        avg_df = pd.DataFrame(avg_data, index=periods, columns=land_cover_types)
        
        # Create stacked area plot
        stacks = ax.stackplot(range(len(periods)), avg_df.T, labels=land_cover_types, 
                             colors=colors, alpha=0.8)
        
        # Add percentage labels at the end of each period
        cumulative = np.zeros(len(periods))
        for i, (lc_type, color) in enumerate(zip(land_cover_types, colors)):
            values = avg_df[lc_type].values
            cumulative += values
            
            # Add text labels at the end of each stack
            for j, period_idx in enumerate(range(len(periods))):
                if values[j] > 1:  # Only label if percentage is > 1%
                    y_pos = cumulative[j] - values[j]/2
                    text_color = 'white' if color in ['#000000', '#00008B', '#006400'] else 'black'
                    ax.text(period_idx, y_pos, f'{values[j]:.1f}%', 
                           ha='center', va='center', fontweight='bold',
                           fontsize=9, color=text_color)
        
        ax.set_title('Land Use/Cover Composition with Percentage Labels', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Percentage Composition (%)', fontsize=12)
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, fontsize=11)
        ax.set_ylim(0, 100)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_individual_trends(self):
        """
        Plot individual trend lines for each land cover class
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        periods = ['1985-2000', '2000-2010', '2010-2023']
        
        # Calculate average percentages
        avg_data = {}
        for period in periods:
            numeric_df = self._get_numeric_data(self.df_periods[period])
            avg_data[period] = numeric_df.mean()
        
        avg_df = pd.DataFrame(avg_data)
        
        # Plot each land cover type in its own subplot
        for i, lc_type in enumerate(avg_df.index):
            if i < len(axes):
                ax = axes[i]
                color = self.land_cover_colors[lc_type]
                
                ax.plot(periods, avg_df.loc[lc_type], marker='o', linewidth=3, 
                       markersize=8, color=color, label=lc_type)
                
                ax.set_title(f'{lc_type.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                ax.set_ylabel('Percentage (%)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on points
                for j, period in enumerate(periods):
                    value = avg_df.loc[lc_type, period]
                    ax.text(j, value + 0.5, f'{value:.1f}%', 
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Hide unused subplots
        for i in range(len(avg_df.index), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Individual Land Use/Cover Trends (1985-2023)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_composition_pie_charts(self):
        """
        Plot pie charts showing composition for each time period
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        
        periods = ['1985-2000', '2000-2010', '2010-2023']
        
        for i, period in enumerate(periods):
            numeric_df = self._get_numeric_data(self.df_periods[period])
            avg_values = numeric_df.mean()
            land_cover_types = numeric_df.columns
            
            colors = [self.land_cover_colors[lc_type] for lc_type in land_cover_types]
            
            # Only show labels for significant portions (>5%)
            labels = [lc_type if avg_values[lc_type] > 5 else '' for lc_type in land_cover_types]
            
            wedges, texts, autotexts = axes[i].pie(avg_values, labels=labels, colors=colors,
                                                  autopct='%1.1f%%', startangle=90)
            
            # Improve text appearance
            for autotext in autotexts:
                autotext.set_color('white' if autotext.get_text() != '' else 'black')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            axes[i].set_title(f'{period}\nTotal: {avg_values.sum():.1f}%', fontsize=12, fontweight='bold')
        
        # Create a custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=self.land_cover_colors[lc_type], 
                               label=lc_type.replace('_', ' ').title())
                         for lc_type in self.numeric_columns]
        
        fig.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.5), 
                  loc='center left', fontsize=10)
        
        plt.suptitle('Land Use/Cover Composition by Time Period', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig

    # Keep other methods (calculate_percentage_changes, etc.) from previous version

def create_sample_data():
    """
    Create sample data for demonstration
    Replace this with your actual data loading
    """
    from models.gpkg import GeoPackage
    gpkg = GeoPackage()
    cols_to_keep = [
        'forest_formation',
        'forest_plantation',
        'grassland',
        'soybean',
        'urban_area',
        'rice',
        'wetland'
        ]
    gdf_1 = gpkg.read_layer('biomas_rs_lulc_85')[cols_to_keep]
    gdf_2 = gpkg.read_layer('biomas_rs_lulc_2000')[cols_to_keep]
    gdf_3 = gpkg.read_layer('biomas_rs_lulc_2010')[cols_to_keep]
    gdf_4 = gpkg.read_layer('biomas_rs_lulc')[cols_to_keep]
    return gdf_1, gdf_2, gdf_3, gdf_4
def main():
    """
    Main function to run the visualization with your specified colors
    """
    # Load your data
    print("Loading data...")
    df_1985_2000, df_2000_2010, df_2010_2023, df_1985_2023 = create_sample_data()
    
    # Initialize visualizer
    print("Creating visualizations...")
    visualizer = LandUseVisualizer(df_1985_2000, df_2000_2010, df_2010_2023, df_1985_2023)
    
    # Create all visualizations with your specified colors
    fig1 = visualizer.plot_stacked_area()
    fig2 = visualizer.plot_stacked_area_with_percentage_labels()
    fig3 = visualizer.plot_individual_trends()
    fig4 = visualizer.plot_composition_pie_charts()
    
    # Save figures
    fig1.savefig('stacked_area_colors.png', dpi=300, bbox_inches='tight')
    fig2.savefig('stacked_area_with_labels.png', dpi=300, bbox_inches='tight')
    fig3.savefig('individual_trends.png', dpi=300, bbox_inches='tight')
    fig4.savefig('pie_charts_composition.png', dpi=300, bbox_inches='tight')
    
    print("Visualizations created with your specified colors!")
    print("Color mapping used:")
    for lc_type, color in visualizer.land_cover_colors.items():
        print(f"  {lc_type}: {color}")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()