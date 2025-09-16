import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List, Union
import argparse
from models.gpkg import GeoPackage

class HistogramVisualizer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the histogram visualizer with data
        
        Args:
            data (pd.DataFrame): Input data containing the columns to visualize
        """
        self.data = data
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Setup consistent plot styling"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
    
    def create_histogram(self, 
                        x_column: str, 
                        y_column: Optional[str] = None,
                        title: Optional[str] = None,
                        xlabel: Optional[str] = None,
                        ylabel: Optional[str] = None,
                        bins: int = 30,
                        alpha: float = 0.7,
                        figsize: tuple = (12, 8)):
        """
        Create a histogram visualization
        
        Args:
            x_column (str): Column name for the x-axis (categorical or numerical)
            y_column (str, optional): Column name for numerical values to aggregate
            title (str, optional): Plot title
            xlabel (str, optional): X-axis label
            ylabel (str, optional): Y-axis label
            bins (int): Number of bins for histogram
            alpha (float): Transparency level
            figsize (tuple): Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine if x_column is categorical
        if self.data[x_column].dtype == 'object' or self.data[x_column].nunique() < 20:
            # Categorical data - create bar plot
            if y_column:
                # Aggregate numerical data by categorical groups
                aggregated_data = self.data.groupby(x_column)[y_column].mean()
                bars = ax.bar(aggregated_data.index.astype(str), aggregated_data.values, 
                             alpha=alpha, color='skyblue', edgecolor='black')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom')
            else:
                # Simple count of categories
                value_counts = self.data[x_column].value_counts()
                bars = ax.bar(value_counts.index.astype(str), value_counts.values,
                             alpha=alpha, color='lightcoral', edgecolor='black')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            
        else:
            # Numerical data - create histogram
            if y_column:
                # For numerical x and y, we might want a different visualization
                ax.scatter(self.data[x_column], self.data[y_column], alpha=0.6)
                ax.set_ylabel(y_column)
            else:
                # Standard histogram
                ax.hist(self.data[x_column].dropna(), bins=bins, alpha=alpha, 
                       color='lightgreen', edgecolor='black')
        
        # Set labels and title
        ax.set_xlabel(xlabel or x_column)
        ax.set_ylabel(ylabel or (y_column if y_column else 'Count'))
        ax.set_title(title or f'Distribution of {y_column if y_column else x_column}')
        
        plt.tight_layout()
        return fig, ax
    
    def create_grouped_histogram(self,
                               x_column: str,
                               group_column: str,
                               y_column: Optional[str] = None,
                               title: Optional[str] = None,
                               figsize: tuple = (14, 10)):
        """
        Create grouped histogram by a categorical column
        
        Args:
            x_column (str): Main column to visualize
            group_column (str): Column to group by
            y_column (str, optional): Numerical column to aggregate
            title (str, optional): Plot title
            figsize (tuple): Figure size
        """
        # Get unique groups
        groups = self.data[group_column].unique()
        n_groups = len(groups)
        
        # Create subplots
        fig, axes = plt.subplots(nrows=(n_groups + 1) // 2, ncols=2, figsize=figsize)
        axes = axes.flatten()
        
        for i, group in enumerate(groups):
            if i >= len(axes):
                break
                
            group_data = self.data[self.data[group_column] == group]
            
            if y_column:
                # Histogram of numerical values for this group
                axes[i].hist(group_data[y_column].dropna(), bins=30, alpha=0.7,
                           color=f'C{i}', edgecolor='black')
                axes[i].set_ylabel('Count')
            else:
                # Count of x_column values for this group
                value_counts = group_data[x_column].value_counts()
                axes[i].bar(value_counts.index.astype(str), value_counts.values,
                          alpha=0.7, color=f'C{i}', edgecolor='black')
                plt.sca(axes[i])
                plt.xticks(rotation=45, ha='right')
            
            axes[i].set_title(f'{group_column}: {group}')
            axes[i].set_xlabel(x_column)
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        fig.suptitle(title or f'{x_column} distribution by {group_column}', fontsize=16)
        plt.tight_layout()
        return fig, axes

def load_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    
    # Sample data with vulnerability index and ethnicity
    n_samples = 1000
    data = {
        'vulnerability_index': np.random.uniform(0, 1, n_samples),
        'etnia_nome': np.random.choice(['White', 'Black', 'Asian', 'Hispanic', 'Other'], n_samples),
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'health_score': np.random.normal(75, 15, n_samples)
    }
    
    return pd.DataFrame(data)

def main():
    """Main function to demonstrate the histogram visualizer"""
    parser = argparse.ArgumentParser(description='Create histogram visualizations')
    parser.add_argument('--file', type=str, help='CSV file path (optional)')
    parser.add_argument('--x', type=str, required=True, help='X column name')
    parser.add_argument('--y', type=str, help='Y column name (numerical)')
    parser.add_argument('--group', type=str, help='Group column name (categorical)')
    parser.add_argument('--output', type=str, default='histogram.png', help='Output file name')
    
    args = parser.parse_args()
    
    # Load data
    if args.file:
        try:
            data = pd.read_csv(args.file)
            print(f"Loaded data from {args.file}")
            print(f"Columns: {list(data.columns)}")
        except FileNotFoundError:
            print(f"File {args.file} not found. Using sample data.")
            data = load_sample_data()
    else:
        print("No file provided. Using sample data.")
        data = load_sample_data()
    
    # Create visualizer instance
    visualizer = HistogramVisualizer(data)
    
    # Create appropriate visualization
    if args.group:
        # Grouped histogram
        fig, _ = visualizer.create_grouped_histogram(
            x_column=args.x,
            group_column=args.group,
            y_column=args.y,
            title=f"{args.x} by {args.group}"
        )
    else:
        # Single histogram
        fig, _ = visualizer.create_histogram(
            x_column=args.x,
            y_column=args.y,
            title=f"Distribution of {args.y if args.y else args.x}"
        )
    
    # Save and show
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {args.output}")
    plt.show()

def demonstrate_all_visualizations():
    """Demonstrate all types of visualizations with sample data"""
    data = load_sample_data()
    visualizer = HistogramVisualizer(data)
    
    print("Sample Data Overview:")
    print(data.head())
    print("\nData Info:")
    print(data.info())
    
    # 1. Basic histogram of vulnerability index
    print("\n1. Creating histogram of vulnerability index...")
    fig1, ax1 = visualizer.create_histogram('vulnerability_index', title='Vulnerability Index Distribution')
    plt.savefig('vulnerability_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Ethnicity count bar chart
    print("2. Creating ethnicity distribution...")
    fig2, ax2 = visualizer.create_histogram('etnia_nome', title='Ethnicity Distribution')
    plt.savefig('ethnicity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Average vulnerability by ethnicity
    print("3. Creating average vulnerability by ethnicity...")
    fig3, ax3 = visualizer.create_histogram('etnia_nome', 'vulnerability_index', 
                                          title='Average Vulnerability Index by Ethnicity',
                                          ylabel='Average Vulnerability Index')
    plt.savefig('vulnerability_by_ethnicity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Grouped histogram: age distribution by ethnicity
    print("4. Creating age distribution by ethnicity...")
    fig4, axes4 = visualizer.create_grouped_histogram('age', 'etnia_nome', 
                                                    title='Age Distribution by Ethnicity')
    plt.savefig('age_by_ethnicity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Income distribution by education level
    print("5. Creating income by education level...")
    fig5, ax5 = visualizer.create_histogram('education_level', 'income',
                                          title='Average Income by Education Level',
                                          ylabel='Average Income')
    plt.savefig('income_by_education.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All visualizations created successfully!")

if __name__ == "__main__":
    # Uncomment the line below for command line usage
    # main()
    
    # Demonstrate all visualizations
    gpkg = GeoPackage()
    gdf = gpkg.read_layer('vindex_1985_2023_form_revised')
    histo = HistogramVisualizer(gdf)
    histo.create_histogram(
        'etnia_nome',
        'vulnerability_index'
    )
    # histo.create_grouped_histogram(
    #     'etnia_nome',
    #     'bioma',
    #     'ca_score'
    # )
    plt.show()
    # demonstrate_all_visualizations()