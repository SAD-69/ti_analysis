import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import warnings
warnings.filterwarnings('ignore')

# Set the style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

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
        
    def calculate_percentage_changes(self):
        """
        Calculate percentage changes between periods
        """
        # Assuming each dataframe has the same structure with land cover types as columns
        # and percentage values as data
        
        # Get all land cover types (assuming they're the same across all periods)
        land_cover_types = self.df_periods['1985-2000'].columns
        
        # Create a dataframe to store percentage changes
        changes_data = {}
        
        # Calculate changes for each transition period
        # 1985-2000 to 2000-2010
        changes_data['2000-2010_vs_1985-2000'] = (
            (self.df_periods['2000-2010'].mean() - self.df_periods['1985-2000'].mean()) / 
            self.df_periods['1985-2000'].mean() * 100
        )
        
        # 2000-2010 to 2010-2023
        changes_data['2010-2023_vs_2000-2010'] = (
            (self.df_periods['2010-2023'].mean() - self.df_periods['2000-2010'].mean()) / 
            self.df_periods['2000-2010'].mean() * 100
        )
        
        # Overall change 1985-2023
        changes_data['1985-2023_vs_1985-2000'] = (
            (self.df_periods['1985-2023'].mean() - self.df_periods['1985-2000'].mean()) / 
            self.df_periods['1985-2000'].mean() * 100
        )
        
        self.changes_df = pd.DataFrame(changes_data)
        return self.changes_df
    
    def plot_timeseries_comparison(self):
        """
        Plot the average percentage for each land cover type across all periods
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for plotting
        periods = list(self.df_periods.keys())[:-1]  # Exclude the general period
        land_cover_types = self.df_periods['1985-2000'].columns
        
        # Calculate average percentages for each period
        avg_percentages = {}
        for period in periods:
            avg_percentages[period] = self.df_periods[period].mean()
        
        avg_df = pd.DataFrame(avg_percentages)
        
        # Plot each land cover type
        for lc_type in land_cover_types:
            ax.plot(periods, avg_df.loc[lc_type], marker='o', linewidth=2, markersize=8, label=lc_type)
        
        ax.set_title('Land Use/Cover Percentage Trends (1985-2023)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Average Percentage (%)', fontsize=12)
        ax.set_xlabel('Time Period', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def plot_percentage_changes_bar(self):
        """
        Plot bar chart of percentage changes between periods
        """
        if not hasattr(self, 'changes_df'):
            self.calculate_percentage_changes()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        fig.suptitle('Percentage Changes in Land Use/Cover Between Periods', fontsize=16, fontweight='bold')
        
        periods_comparison = self.changes_df.columns
        
        for i, period in enumerate(periods_comparison):
            ax = axes[i]
            changes = self.changes_df[period].sort_values()
            
            colors = ['red' if x < 0 else 'green' for x in changes]
            bars = ax.barh(changes.index, changes, color=colors, alpha=0.7)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                label_x_pos = width + (0.01 * width) if width >= 0 else width - (0.01 * abs(width))
                ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                       f'{width:.1f}%', va='center', ha='left' if width >= 0 else 'right')
            
            ax.set_title(f'Change: {period}', fontsize=12)
            ax.set_xlabel('Percentage Change (%)')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_heatmap_changes(self):
        """
        Plot heatmap of percentage changes
        """
        if not hasattr(self, 'changes_df'):
            self.calculate_percentage_changes()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        im = ax.imshow(self.changes_df.T, cmap='RdBu_r', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(self.changes_df.index)))
        ax.set_yticks(range(len(self.changes_df.columns)))
        ax.set_xticklabels(self.changes_df.index, rotation=45, ha='right')
        ax.set_yticklabels(self.changes_df.columns)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percentage Change (%)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(self.changes_df.columns)):
            for j in range(len(self.changes_df.index)):
                text = ax.text(j, i, f'{self.changes_df.iloc[j, i]:.1f}%',
                              ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Heatmap of Land Use/Cover Percentage Changes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_stacked_area(self):
        """
        Plot stacked area chart showing composition over time
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        periods = ['1985-2000', '2000-2010', '2010-2023']
        land_cover_types = self.df_periods['1985-2000'].columns
        
        # Calculate average percentages
        avg_data = []
        for period in periods:
            avg_data.append(self.df_periods[period].mean().values)
        
        avg_df = pd.DataFrame(avg_data, index=periods, columns=land_cover_types)
        
        # Create stacked area plot
        ax.stackplot(range(len(periods)), avg_df.T, labels=land_cover_types, alpha=0.8)
        
        ax.set_title('Land Use/Cover Composition Over Time', fontsize=16, fontweight='bold')
        ax.set_ylabel('Percentage Composition (%)', fontsize=12)
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def create_summary_table(self):
        """
        Create a summary table with statistics
        """
        summary_data = {}
        
        for period, df in self.df_periods.items():
            summary_data[period] = {
                'Mean': df.mean(),
                'Std': df.std(),
                'Min': df.min(),
                'Max': df.max()
            }
        
        # Also include percentage changes
        if not hasattr(self, 'changes_df'):
            self.calculate_percentage_changes()
        
        return summary_data, self.changes_df



def main():
    """
    Main function to run the visualization
    """
    # Load your data here - replace with your actual data loading
    print("Loading data...")
    df_1985_2000, df_2000_2010, df_2010_2023, df_1985_2023 = create_sample_data()
    
    # Initialize visualizer
    print("Creating visualizations...")
    visualizer = LandUseVisualizer(df_1985_2000, df_2000_2010, df_2010_2023, df_1985_2023)
    
    # Create all visualizations
    fig1 = visualizer.plot_timeseries_comparison()
    fig2 = visualizer.plot_percentage_changes_bar()
    fig3 = visualizer.plot_heatmap_changes()
    fig4 = visualizer.plot_stacked_area()
    
    # Calculate and display summary statistics
    summary_data, changes_df = visualizer.create_summary_table()
    
    # Save figures
    fig1.savefig('land_use_timeseries.png', dpi=300, bbox_inches='tight')
    fig2.savefig('percentage_changes_bar.png', dpi=300, bbox_inches='tight')
    fig3.savefig('changes_heatmap.png', dpi=300, bbox_inches='tight')
    fig4.savefig('stacked_area_composition.png', dpi=300, bbox_inches='tight')
    
    # Save summary data to CSV
    changes_df.to_csv('percentage_changes_summary.csv')
    
    print("Visualizations created and saved successfully!")
    print("\nPercentage Changes Summary:")
    print(changes_df)
    changes_df.to_csv("changes_use.csv")
    
    # Show plots
    plt.show()

    # Example usage and sample data creation
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

if __name__ == "__main__":
    main()