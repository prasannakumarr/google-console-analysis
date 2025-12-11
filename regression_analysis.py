#!/usr/bin/env python3
"""
Regression Analysis for SEO Data
Performs regression analysis on SEO optimization opportunity data
and creates visualizations of the relationships.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import os

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create output directory
os.makedirs('regression_results', exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the SEO data for analysis"""
    print("Loading and preparing data...")
    
    # Load the data
    df = pd.read_csv('SEO optimization opportunity.csv')
    
    # Clean CTR column - remove % sign and convert to float
    df['CTR'] = df['CTR'].str.rstrip('%').astype('float') / 100.0
    
    # Convert keywords to string
    df['Keywords'] = df['Keywords'].astype('string')
    
    print(f"Loaded {len(df)} records")
    print("Data preview:")
    print(df.head())
    print("\nData description:")
    print(df.describe())
    
    return df

def create_correlation_matrix(df):
    """Create and save correlation matrix"""
    print("\nCreating correlation matrix...")
    
    # Select numeric columns
    numeric_cols = ['Clicks', 'Impressions', 'CTR', 'Position']
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                annot_kws={"size": 12}, fmt=".3f")
    plt.title('Correlation Matrix of SEO Metrics', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    corr_plot_path = 'regression_results/correlation_matrix.png'
    plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved correlation matrix to {corr_plot_path}")
    
    plt.close()
    return corr_matrix

def perform_regression_analysis(df):
    """Perform multiple regression analyses"""
    print("\nPerforming regression analysis...")
    
    results = {}
    
    # Analysis 1: Clicks ~ Impressions + CTR + Position
    print("\n1. Analyzing: Clicks ~ Impressions + CTR + Position")
    X1 = df[['Impressions', 'CTR', 'Position']]
    y1 = df['Clicks']
    X1 = sm.add_constant(X1)  # Add intercept
    
    model1 = sm.OLS(y1, X1).fit()
    results['clicks_regression'] = {
        'model': model1,
        'summary': model1.summary(),
        'r2': model1.rsquared,
        'adj_r2': model1.rsquared_adj
    }
    
    print(model1.summary())
    
    # Analysis 2: CTR ~ Position + Impressions
    print("\n2. Analyzing: CTR ~ Position + Impressions")
    X2 = df[['Position', 'Impressions']]
    y2 = df['CTR']
    X2 = sm.add_constant(X2)
    
    model2 = sm.OLS(y2, X2).fit()
    results['ctr_regression'] = {
        'model': model2,
        'summary': model2.summary(),
        'r2': model2.rsquared,
        'adj_r2': model2.rsquared_adj
    }
    
    print(model2.summary())
    
    # Analysis 3: Position ~ Impressions + Clicks
    print("\n3. Analyzing: Position ~ Impressions + Clicks")
    X3 = df[['Impressions', 'Clicks']]
    y3 = df['Position']
    X3 = sm.add_constant(X3)
    
    model3 = sm.OLS(y3, X3).fit()
    results['position_regression'] = {
        'model': model3,
        'summary': model3.summary(),
        'r2': model3.rsquared,
        'adj_r2': model3.rsquared_adj
    }
    
    print(model3.summary())
    
    return results

def create_regression_plots(df, results):
    """Create visualizations for regression analysis"""
    print("\nCreating regression plots...")
    
    # Plot 1: Clicks vs Impressions with regression line
    plt.figure(figsize=(12, 8))
    sns.regplot(x='Impressions', y='Clicks', data=df, 
                scatter_kws={'alpha':0.6, 's':60}, 
                line_kws={'color':'red', 'linewidth':2})
    plt.title('Clicks vs Impressions with Regression Line', fontsize=16)
    plt.xlabel('Impressions', fontsize=12)
    plt.ylabel('Clicks', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    clicks_vs_impressions_path = 'regression_results/clicks_vs_impressions.png'
    plt.savefig(clicks_vs_impressions_path, dpi=300, bbox_inches='tight')
    print(f"Saved clicks vs impressions plot to {clicks_vs_impressions_path}")
    plt.close()
    
    # Plot 2: CTR vs Position with regression line
    plt.figure(figsize=(12, 8))
    sns.regplot(x='Position', y='CTR', data=df, 
                scatter_kws={'alpha':0.6, 's':60}, 
                line_kws={'color':'green', 'linewidth':2})
    plt.title('CTR vs Position with Regression Line', fontsize=16)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Click-Through Rate (CTR)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ctr_vs_position_path = 'regression_results/ctr_vs_position.png'
    plt.savefig(ctr_vs_position_path, dpi=300, bbox_inches='tight')
    print(f"Saved CTR vs position plot to {ctr_vs_position_path}")
    plt.close()
    
    # Plot 3: Position vs Impressions with regression line
    plt.figure(figsize=(12, 8))
    sns.regplot(x='Impressions', y='Position', data=df, 
                scatter_kws={'alpha':0.6, 's':60}, 
                line_kws={'color':'purple', 'linewidth':2})
    plt.title('Position vs Impressions with Regression Line', fontsize=16)
    plt.xlabel('Impressions', fontsize=12)
    plt.ylabel('Position', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    position_vs_impressions_path = 'regression_results/position_vs_impressions.png'
    plt.savefig(position_vs_impressions_path, dpi=300, bbox_inches='tight')
    print(f"Saved position vs impressions plot to {position_vs_impressions_path}")
    plt.close()
    
    # Plot 4: Pairplot of all numeric variables
    plt.figure(figsize=(16, 12))
    numeric_df = df[['Clicks', 'Impressions', 'CTR', 'Position']]
    pairplot = sns.pairplot(numeric_df, diag_kind='kde', 
                           plot_kws={'alpha': 0.6, 's': 50})
    pairplot.fig.suptitle('Pairplot of SEO Metrics', y=1.02, fontsize=18)
    
    pairplot_path = 'regression_results/pairplot.png'
    pairplot.savefig(pairplot_path, dpi=300, bbox_inches='tight')
    print(f"Saved pairplot to {pairplot_path}")
    plt.close()

def save_regression_summary(results):
    """Save regression analysis summary to text file"""
    print("\nSaving regression analysis summary...")
    
    with open('regression_results/regression_analysis_summary.txt', 'w') as f:
        f.write("SEO REGRESSION ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Write correlation analysis
        f.write("CORRELATION ANALYSIS\n")
        f.write("-" * 25 + "\n")
        f.write("The correlation matrix shows relationships between variables.\n")
        f.write("Key insights from correlation analysis:\n")
        f.write("- Clicks and Impressions are positively correlated\n")
        f.write("- CTR and Position show a negative correlation (higher position = lower CTR)\n")
        f.write("- Position and Impressions may have an interesting relationship\n\n")
        
        # Write regression results
        for analysis_name, analysis_data in results.items():
            f.write(f"{analysis_name.upper()}\n")
            f.write("=" * len(analysis_name) + "\n")
            f.write(f"R-squared: {analysis_data['r2']:.4f}\n")
            f.write(f"Adjusted R-squared: {analysis_data['adj_r2']:.4f}\n")
            f.write("\n")
            f.write(str(analysis_data['summary']))
            f.write("\n\n")
            f.write("INTERPRETATION:\n")
            f.write("-" * 15 + "\n")
            
            if analysis_name == 'clicks_regression':
                f.write("This model examines how Impressions, CTR, and Position affect Clicks.\n")
                f.write("The coefficients show the expected change in clicks for each unit change\n")
                f.write("in the independent variables, holding other variables constant.\n")
            
            elif analysis_name == 'ctr_regression':
                f.write("This model examines how Position and Impressions affect CTR.\n")
                f.write("The negative coefficient for Position suggests that higher positions\n")
                f.write("(lower numbers) are associated with higher CTR.\n")
            
            elif analysis_name == 'position_regression':
                f.write("This model examines how Impressions and Clicks affect Position.\n")
                f.write("The coefficients indicate the relationship between search volume\n")
                f.write("metrics and ranking position.\n")
            
            f.write("\n\n")

def main():
    """Main function to run regression analysis"""
    print("SEO Regression Analysis")
    print("=" * 30)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Create correlation matrix
    corr_matrix = create_correlation_matrix(df)
    
    # Perform regression analysis
    results = perform_regression_analysis(df)
    
    # Create regression plots
    create_regression_plots(df, results)
    
    # Save regression summary
    save_regression_summary(results)
    
    print("\n" + "=" * 50)
    print("REGRESSION ANALYSIS COMPLETED!")
    print("=" * 50)
    print("Results saved in 'regression_results/' directory:")
    print("- correlation_matrix.png")
    print("- clicks_vs_impressions.png")
    print("- ctr_vs_position.png")
    print("- position_vs_impressions.png")
    print("- pairplot.png")
    print("- regression_analysis_summary.txt")
    print("\nKey findings:")
    print(f"- Clicks regression R²: {results['clicks_regression']['r2']:.3f}")
    print(f"- CTR regression R²: {results['ctr_regression']['r2']:.3f}")
    print(f"- Position regression R²: {results['position_regression']['r2']:.3f}")

if __name__ == "__main__":
    main()