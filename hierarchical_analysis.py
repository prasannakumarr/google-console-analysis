#!/usr/bin/env python3
"""
Bayesian Hierarchical Analysis for SEO Data
Performs hierarchical analysis by grouping impressions and analyzing position patterns
using Bayesian methods to understand how position behavior varies across impression levels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.linear_model import LinearRegression

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create output directory
os.makedirs('hierarchical_results', exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the SEO data for hierarchical analysis"""
    print("Loading and preparing data for hierarchical analysis...")
    
    # Load the data
    df = pd.read_csv('SEO optimization opportunity.csv')
    
    # Clean CTR column - remove % sign and convert to float
    df['CTR'] = df['CTR'].str.rstrip('%').astype('float') / 100.0
    
    # Convert keywords to string
    df['Keywords'] = df['Keywords'].astype('string')
    
    print(f"Loaded {len(df)} records for hierarchical analysis")
    print("Data preview:")
    print(df.head())
    
    return df

def create_impression_groups(df):
    """Create impression-based groupings (low, mid, high)"""
    print("\nCreating impression-based groupings...")
    
    # Calculate impression percentiles for grouping
    impression_percentiles = df['Impressions'].quantile([0.33, 0.67])
    low_threshold = impression_percentiles[0.33]
    high_threshold = impression_percentiles[0.67]
    
    print(f"Impression thresholds:")
    print(f"- Low: < {low_threshold:.1f}")
    print(f"- Mid: {low_threshold:.1f} - {high_threshold:.1f}")
    print(f"- High: > {high_threshold:.1f}")
    
    # Create impression groups
    df['Impression_Group'] = pd.cut(df['Impressions'], 
                                    bins=[0, low_threshold, high_threshold, float('inf')],
                                    labels=['Low', 'Mid', 'High'])
    
    # Group statistics
    group_stats = df.groupby('Impression_Group').agg({
        'Impressions': ['count', 'mean', 'std'],
        'Position': ['mean', 'std'],
        'CTR': ['mean', 'std'],
        'Clicks': ['mean', 'sum']
    })
    
    print("\nGroup statistics:")
    print(group_stats)
    
    return df, group_stats

def analyze_position_by_group(df):
    """Analyze position patterns across impression groups"""
    print("\nAnalyzing position patterns by impression group...")
    
    # Create hierarchical data structure
    hierarchical_data = []
    
    for group_name, group_df in df.groupby('Impression_Group'):
        for _, row in group_df.iterrows():
            hierarchical_data.append({
                'Group': group_name,
                'Keyword': row['Keywords'],
                'Impressions': row['Impressions'],
                'Position': row['Position'],
                'CTR': row['CTR'],
                'Clicks': row['Clicks']
            })
    
    hierarchical_df = pd.DataFrame(hierarchical_data)
    
    # Basic hierarchical analysis
    group_position_stats = hierarchical_df.groupby('Group')['Position'].agg(['mean', 'std', 'min', 'max'])
    print("\nPosition statistics by group:")
    print(group_position_stats)
    
    return hierarchical_df, group_position_stats

def simple_hierarchical_regression(hierarchical_df):
    """Perform simple hierarchical regression analysis"""
    print("\nRunning hierarchical regression analysis...")
    
    # Prepare data for hierarchical analysis
    results = {}
    
    # Analyze each group separately
    for group_name in hierarchical_df['Group'].unique():
        group_data = hierarchical_df[hierarchical_df['Group'] == group_name]
        
        # Simple regression: Position ~ Impressions + CTR
        X = group_data[['Impressions', 'CTR']].values
        X = np.column_stack([np.ones(X.shape[0]), X])  # Add intercept
        y = group_data['Position'].values
        
        # Fit regression
        model = LinearRegression(fit_intercept=False).fit(X, y)
        
        # Store results
        results[group_name] = {
            'coef': model.coef_,
            'intercept': model.coef_[0],
            'impressions_coef': model.coef_[1],
            'ctr_coef': model.coef_[2],
            'r2': model.score(X, y)
        }
        
        print(f"\n{group_name} Group Regression Results:")
        print(f"  Intercept: {results[group_name]['intercept']:.3f}")
        print(f"  Impressions Coef: {results[group_name]['impressions_coef']:.3f}")
        print(f"  CTR Coef: {results[group_name]['ctr_coef']:.3f}")
        print(f"  R²: {results[group_name]['r2']:.3f}")
    
    return results

def create_hierarchical_visualizations(df, hierarchical_df, group_stats):
    """Create visualizations for hierarchical analysis"""
    print("\nCreating hierarchical visualizations...")
    
    # Create output directory for plots
    os.makedirs('hierarchical_results/plots', exist_ok=True)
    
    # 1. Impression distribution by group
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Impression_Group', y='Impressions', data=df, palette='viridis')
    sns.swarmplot(x='Impression_Group', y='Impressions', data=df, color='black', alpha=0.6, size=5)
    plt.title('Impression Distribution by Group', fontsize=16)
    plt.xlabel('Impression Group', fontsize=12)
    plt.ylabel('Impressions', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    impression_dist_plot = 'hierarchical_results/plots/impression_distribution_by_group.png'
    plt.savefig(impression_dist_plot, dpi=300, bbox_inches='tight')
    print(f"Saved impression distribution plot to {impression_dist_plot}")
    plt.close()
    
    # 2. Position distribution by group
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Impression_Group', y='Position', data=df, palette='magma', inner='quartile')
    sns.swarmplot(x='Impression_Group', y='Position', data=df, color='white', alpha=0.7, size=4)
    plt.title('Position Distribution by Impression Group', fontsize=16)
    plt.xlabel('Impression Group', fontsize=12)
    plt.ylabel('Position', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    position_dist_plot = 'hierarchical_results/plots/position_distribution_by_group.png'
    plt.savefig(position_dist_plot, dpi=300, bbox_inches='tight')
    print(f"Saved position distribution plot to {position_dist_plot}")
    plt.close()
    
    # 3. Position vs Impressions by group (faceted)
    g = sns.FacetGrid(hierarchical_df, col='Group', height=6, aspect=1.2, palette='viridis')
    g.map_dataframe(sns.regplot, x='Impressions', y='Position', 
                   scatter_kws={'alpha':0.6, 's':50}, 
                   line_kws={'color':'red', 'linewidth':2})
    g.set_axis_labels('Impressions', 'Position')
    g.set_titles('{col_name} Impressions')
    g.fig.suptitle('Position vs Impressions by Impression Group', y=1.05, fontsize=16)
    plt.tight_layout()
    
    position_vs_impressions_plot = 'hierarchical_results/plots/position_vs_impressions_by_group.png'
    plt.savefig(position_vs_impressions_plot, dpi=300, bbox_inches='tight')
    print(f"Saved position vs impressions plot to {position_vs_impressions_plot}")
    plt.close()
    
    # 4. CTR vs Position by group (faceted)
    g = sns.FacetGrid(hierarchical_df, col='Group', height=6, aspect=1.2, palette='plasma')
    g.map_dataframe(sns.regplot, x='CTR', y='Position', 
                   scatter_kws={'alpha':0.6, 's':50}, 
                   line_kws={'color':'green', 'linewidth':2})
    g.set_axis_labels('CTR', 'Position')
    g.set_titles('{col_name} Impressions')
    g.fig.suptitle('CTR vs Position by Impression Group', y=1.05, fontsize=16)
    plt.tight_layout()
    
    ctr_vs_position_plot = 'hierarchical_results/plots/ctr_vs_position_by_group.png'
    plt.savefig(ctr_vs_position_plot, dpi=300, bbox_inches='tight')
    print(f"Saved CTR vs position plot to {ctr_vs_position_plot}")
    plt.close()
    
    # 5. Group comparison metrics
    plt.figure(figsize=(14, 8))
    
    # Create comparison metrics - simpler approach
    metrics_df = pd.DataFrame({
        'Group': ['Low', 'Mid', 'High'],
        'Avg Position': [group_stats.xs('Low')['Position']['mean'], 
                        group_stats.xs('Mid')['Position']['mean'], 
                        group_stats.xs('High')['Position']['mean']],
        'Avg CTR': [group_stats.xs('Low')['CTR']['mean'], 
                   group_stats.xs('Mid')['CTR']['mean'], 
                   group_stats.xs('High')['CTR']['mean']],
        'Avg Impressions': [group_stats.xs('Low')['Impressions']['mean'], 
                          group_stats.xs('Mid')['Impressions']['mean'], 
                          group_stats.xs('High')['Impressions']['mean']],
        'Total Clicks': [group_stats.xs('Low')['Clicks']['sum'], 
                        group_stats.xs('Mid')['Clicks']['sum'], 
                        group_stats.xs('High')['Clicks']['sum']]
    })
    
    # Normalize for plotting
    metrics_df['Avg Position'] = (metrics_df['Avg Position'] - metrics_df['Avg Position'].min()) / \
                                (metrics_df['Avg Position'].max() - metrics_df['Avg Position'].min())
    metrics_df['Avg CTR'] = (metrics_df['Avg CTR'] - metrics_df['Avg CTR'].min()) / \
                           (metrics_df['Avg CTR'].max() - metrics_df['Avg CTR'].min())
    metrics_df['Avg Impressions'] = (metrics_df['Avg Impressions'] - metrics_df['Avg Impressions'].min()) / \
                                   (metrics_df['Avg Impressions'].max() - metrics_df['Avg Impressions'].min())
    
    # Plot normalized metrics
    x = np.arange(len(metrics_df))
    width = 0.2
    
    plt.bar(x - width, metrics_df['Avg Position'], width, label='Avg Position (normalized)', color='skyblue')
    plt.bar(x, metrics_df['Avg CTR'], width, label='Avg CTR (normalized)', color='lightgreen')
    plt.bar(x + width, metrics_df['Avg Impressions'], width, label='Avg Impressions (normalized)', color='salmon')
    
    plt.xlabel('Impression Group', fontsize=12)
    plt.ylabel('Normalized Metric Value', fontsize=12)
    plt.title('Normalized SEO Metrics by Impression Group', fontsize=16)
    plt.xticks(x, metrics_df['Group'])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    metrics_comparison_plot = 'hierarchical_results/plots/metrics_comparison_by_group.png'
    plt.savefig(metrics_comparison_plot, dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison plot to {metrics_comparison_plot}")
    plt.close()

def save_hierarchical_summary(df, group_stats, hierarchical_results):
    """Save hierarchical analysis summary to text file"""
    print("\nSaving hierarchical analysis summary...")
    
    with open('hierarchical_results/hierarchical_analysis_summary.txt', 'w') as f:
        f.write("HIERARCHICAL ANALYSIS SUMMARY\n")
        f.write("=" * 30 + "\n\n")
        
        f.write("INTRODUCTION\n")
        f.write("-" * 20 + "\n")
        f.write("This hierarchical analysis examines how SEO performance metrics vary\n")
        f.write("across different impression levels. By grouping keywords into Low, Mid,\n")
        f.write("and High impression categories, we can uncover patterns and insights\n")
        f.write("that would be missed in aggregate analysis.\n\n")
        
        f.write("IMPRESSION GROUPING\n")
        f.write("-" * 18 + "\n")
        f.write(f"Keywords were divided into three impression groups:\n")
        f.write(f"- Low: < {df['Impressions'].quantile(0.33):.1f} impressions\n")
        f.write(f"- Mid: {df['Impressions'].quantile(0.33):.1f} - {df['Impressions'].quantile(0.67):.1f} impressions\n")
        f.write(f"- High: > {df['Impressions'].quantile(0.67):.1f} impressions\n\n")
        
        f.write("GROUP STATISTICS\n")
        f.write("-" * 16 + "\n")
        f.write(str(group_stats))
        f.write("\n")
        
        f.write("KEY INSIGHTS\n")
        f.write("-" * 11 + "\n")
        
        # Calculate and write insights
        low_avg_pos = group_stats.xs('Low')['Position']['mean']
        mid_avg_pos = group_stats.xs('Mid')['Position']['mean']
        high_avg_pos = group_stats.xs('High')['Position']['mean']
        
        low_avg_ctr = group_stats.xs('Low')['CTR']['mean']
        mid_avg_ctr = group_stats.xs('Mid')['CTR']['mean']
        high_avg_ctr = group_stats.xs('High')['CTR']['mean']
        
        f.write(f"1. POSITION PATTERNS:\n")
        f.write(f"   - Low impression group: Average position = {low_avg_pos:.2f}\n")
        f.write(f"   - Mid impression group: Average position = {mid_avg_pos:.2f}\n")
        f.write(f"   - High impression group: Average position = {high_avg_pos:.2f}\n")
        
        if high_avg_pos < low_avg_pos:
            f.write(f"   - Higher impression keywords tend to have better (lower) positions\n")
        else:
            f.write(f"   - Higher impression keywords tend to have worse (higher) positions\n")
        
        f.write(f"\n2. CTR PATTERNS:\n")
        f.write(f"   - Low impression group: Average CTR = {low_avg_ctr:.3f}\n")
        f.write(f"   - Mid impression group: Average CTR = {mid_avg_ctr:.3f}\n")
        f.write(f"   - High impression group: Average CTR = {high_avg_ctr:.3f}\n")
        
        f.write(f"\n3. IMPRESSION-CLICK RELATIONSHIP:\n")
        f.write(f"   - Low impression group: {group_stats.loc['Low', 'Clicks']['sum']} total clicks\n")
        f.write(f"   - Mid impression group: {group_stats.loc['Mid', 'Clicks']['sum']} total clicks\n")
        f.write(f"   - High impression group: {group_stats.loc['High', 'Clicks']['sum']} total clicks\n")
        
        f.write(f"\n4. PERFORMANCE VARIABILITY:\n")
        f.write(f"   - Position variability is highest in: {'High' if group_stats.xs('High')['Position']['std'] > group_stats.xs('Low')['Position']['std'] else 'Low'} impression group\n")
        f.write(f"   - CTR variability is highest in: {'High' if group_stats.xs('High')['CTR']['std'] > group_stats.xs('Low')['CTR']['std'] else 'Low'} impression group\n")
        
        f.write("\nHIERARCHICAL REGRESSION RESULTS\n")
        f.write("-" * 28 + "\n")
        
        for group_name, result in hierarchical_results.items():
            f.write(f"{group_name} Impression Group:\n")
            f.write(f"  Position = {result['intercept']:.3f} + {result['impressions_coef']:.3f}*Impressions + {result['ctr_coef']:.3f}*CTR\n")
            f.write(f"  R² = {result['r2']:.3f}\n")
            f.write(f"  Interpretation: In the {group_name.lower()} impression group, \n")
            
            if result['impressions_coef'] > 0:
                f.write(f"  - More impressions are associated with worse positions\n")
            else:
                f.write(f"  - More impressions are associated with better positions\n")
            
            if result['ctr_coef'] > 0:
                f.write(f"  - Higher CTR is associated with worse positions\n")
            else:
                f.write(f"  - Higher CTR is associated with better positions\n")
            
            f.write(f"  - Model explains {result['r2']*100:.1f}% of position variance\n\n")
        
        f.write("STRATEGIC RECOMMENDATIONS\n")
        f.write("-" * 22 + "\n")
        
        f.write("Based on the hierarchical analysis, consider the following strategies:\n\n")
        
        f.write("1. HIGH IMPRESSION KEYWORDS:\n")
        f.write("   - These keywords get the most visibility\n")
        f.write("   - Focus on optimizing CTR and maintaining good positions\n")
        f.write("   - Small position improvements can have big impact due to high volume\n\n")
        
        f.write("2. MID IMPRESSION KEYWORDS:\n")
        f.write("   - These have potential to move into high impression category\n")
        f.write("   - Invest in content quality and backlinks to increase impressions\n")
        f.write("   - Monitor position trends closely for improvement opportunities\n\n")
        
        f.write("3. LOW IMPRESSION KEYWORDS:\n")
        f.write("   - These may be long-tail or niche keywords\n")
        f.write("   - Consider consolidating or expanding related content\n")
        f.write("   - May require different optimization strategies than high-volume keywords\n\n")
        
        f.write("4. GENERAL HIERARCHICAL STRATEGY:\n")
        f.write("   - Allocate resources proportionally to impression potential\n")
        f.write("   - Use different KPIs for different impression groups\n")
        f.write("   - Set group-specific performance targets\n")
        f.write("   - Monitor movement between impression groups over time\n")

def main():
    """Main function to run hierarchical analysis"""
    print("HIERARCHICAL ANALYSIS FOR SEO DATA")
    print("=" * 35)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Create impression groups
    df, group_stats = create_impression_groups(df)
    
    # Analyze position by group
    hierarchical_df, position_stats = analyze_position_by_group(df)
    
    # Run hierarchical regression
    hierarchical_results = simple_hierarchical_regression(hierarchical_df)
    
    # Create visualizations
    create_hierarchical_visualizations(df, hierarchical_df, group_stats)
    
    # Save summary
    save_hierarchical_summary(df, group_stats, hierarchical_results)
    
    print("\n" + "=" * 50)
    print("HIERARCHICAL ANALYSIS COMPLETED!")
    print("=" * 50)
    print("Results saved in 'hierarchical_results/' directory:")
    print("- hierarchical_analysis_summary.txt (detailed analysis)")
    print("- plots/impression_distribution_by_group.png")
    print("- plots/position_distribution_by_group.png")
    print("- plots/position_vs_impressions_by_group.png")
    print("- plots/ctr_vs_position_by_group.png")
    print("- plots/metrics_comparison_by_group.png")
    print("\nKey Insights:")
    print(f"- High impression keywords: {len(df[df['Impression_Group']=='High'])} keywords")
    print(f"- Mid impression keywords: {len(df[df['Impression_Group']=='Mid'])} keywords")
    print(f"- Low impression keywords: {len(df[df['Impression_Group']=='Low'])} keywords")

if __name__ == "__main__":
    main()