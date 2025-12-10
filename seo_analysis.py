# SEO Optimization Opportunity Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better looking plots
sns.set(style="whitegrid")

# 1. Load the dataset using Pandas
print("Loading dataset...")
df = pd.read_csv('SEO optimization opportunity.csv')
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# 2. Display the cleaned data as a table
print("\n" + "="*50)
print("CLEANED DATA TABLE")
print("="*50)
print(df.to_string(index=False))

# 3. Perform basic exploratory analysis with summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(df.describe())

# Check for missing values
print("\n" + "="*50)
print("MISSING VALUES")
print("="*50)
print(df.isnull().sum())

# Data types
print("\n" + "="*50)
print("DATA TYPES")
print("="*50)
print(df.dtypes)

# 4. Create relevant plots to understand the data
print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

# Clean CTR column - remove % sign and convert to float
print("Cleaning CTR data...")
df['CTR_clean'] = df['CTR'].str.rstrip('%').astype('float') / 100.0

# Create figure directory
import os
os.makedirs('figures', exist_ok=True)

# Plot 1: Impressions vs Position
print("Creating Impressions vs Position plot...")
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Position', y='Impressions', hue='Clicks', size='Clicks', sizes=(50, 200), palette='viridis')
plt.title('Impressions vs Position (Size = Clicks, Color = Clicks)', fontsize=16)
plt.xlabel('Average Position', fontsize=14)
plt.ylabel('Impressions', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/impressions_vs_position.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Clicks vs Position
print("Creating Clicks vs Position plot...")
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Position', y='Clicks', hue='Impressions', size='Impressions', sizes=(50, 200), palette='plasma')
plt.title('Clicks vs Position (Size = Impressions, Color = Impressions)', fontsize=16)
plt.xlabel('Average Position', fontsize=14)
plt.ylabel('Clicks', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/clicks_vs_position.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: CTR Distribution
print("Creating CTR Distribution plot...")
plt.figure(figsize=(12, 8))
sns.histplot(data=df, x='CTR_clean', bins=20, kde=True, color='skyblue')
plt.title('CTR Distribution', fontsize=16)
plt.xlabel('Click-Through Rate', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/ctr_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3b: CTR vs Impressions Scatter Plot
print("Creating CTR vs Impressions plot...")
plt.figure(figsize=(14, 10))
sizes = df['Impressions'] / 50  # Scale down for bubble size
scatter = plt.scatter(data=df, x='CTR_clean', y='Impressions', 
                      s=sizes, c=df['Position'], cmap='viridis', alpha=0.7)

# Add labels and annotations
plt.title('CTR vs Impressions with Position Color Coding (Bubble Size = Impressions)', fontsize=16)
plt.xlabel('Click-Through Rate', fontsize=14)
plt.ylabel('Impressions', fontsize=14)
plt.grid(True, alpha=0.3)

# Add colorbar for position reference
cbar = plt.colorbar(scatter)
cbar.set_label('Average Position', fontsize=12)

# Add annotations for top opportunities
for i, row in df.iterrows():
    if row['Impressions'] > 800 or row['CTR_clean'] > 0.01:
        plt.text(row['CTR_clean'], row['Impressions'], row['Keywords'],
                 fontsize=9, ha='center', va='center', color='black', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

plt.tight_layout()
plt.savefig('figures/ctr_vs_impressions.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3c: CTR Distribution by Impression Range
print("Creating CTR Distribution by Impression Range plot...")
# Create impression ranges
df['Impression_Range'] = pd.cut(df['Impressions'], 
                                bins=[0, 300, 600, 900, 1500], 
                                labels=['Low (0-300)', 'Medium (300-600)', 'High (600-900)', 'Very High (900+)'])

plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='Impression_Range', y='CTR_clean', hue='Impression_Range')
plt.title('CTR Distribution by Impression Range', fontsize=16)
plt.xlabel('Impression Range', fontsize=14)
plt.ylabel('Click-Through Rate', fontsize=14)
plt.grid(True, alpha=0.3)

# Add scatter points to show individual data points
for impression_range in df['Impression_Range'].unique():
    subset = df[df['Impression_Range'] == impression_range]
    plt.scatter(x=[impression_range]*len(subset), 
                y=subset['CTR_clean'], 
                color='black', alpha=0.5, s=50)

plt.tight_layout()
plt.savefig('figures/ctr_by_impression_range.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Top Opportunity Pages by Impressions - Bar Chart with Labels
print("Creating Top Opportunity Pages bar plot...")
top_pages = df.nlargest(10, 'Impressions')
plt.figure(figsize=(14, 10))
ax = sns.barplot(data=top_pages, y='Keywords', x='Impressions', palette='rocket', hue='Position', dodge=False)
plt.title('Top 10 Opportunity Pages by Impressions (Color = Position)', fontsize=16)
plt.xlabel('Impressions', fontsize=14)
plt.ylabel('Keywords', fontsize=14)
plt.grid(True, alpha=0.3)

# Add position labels inside each bar
for i, (index, row) in enumerate(top_pages.iterrows()):
    position = row['Position']
    impressions = row['Impressions']
    # Position the label in the middle of the bar
    ax.text(impressions/2, i, f'Pos: {position}', 
            ha='center', va='center', color='white', 
            fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/top_opportunity_pages_with_labels.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Alternative Visualization - Bubble Chart
print("Creating Bubble Chart visualization...")
plt.figure(figsize=(14, 10))
sizes = top_pages['Impressions'] / 10  # Scale down for bubble size
colors = top_pages['Position']
scatter = plt.scatter(data=top_pages, x='Position', y='Impressions', 
                      s=sizes, c=colors, cmap='viridis', alpha=0.7)

# Add labels and annotations
plt.title('Top 10 Opportunity Pages: Position vs Impressions (Bubble Size = Impressions)', fontsize=16)
plt.xlabel('Average Position', fontsize=14)
plt.ylabel('Impressions', fontsize=14)
plt.grid(True, alpha=0.3)

# Add colorbar for position reference
cbar = plt.colorbar(scatter)
cbar.set_label('Position Value', fontsize=12)

# Add keyword labels to each bubble
for i, row in top_pages.iterrows():
    plt.text(row['Position'], row['Impressions'], row['Keywords'],
             fontsize=10, ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig('figures/top_opportunity_bubble_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Alternative Visualization - Heatmap
print("Creating Heatmap visualization...")
# Create a pivot table for heatmap
heatmap_data = top_pages.pivot_table(values='Impressions', index='Keywords', columns='Position', aggfunc='first')
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap='YlOrRd', 
            linewidths=0.5, linecolor='black', cbar_kws={'label': 'Impressions'})
plt.title('Top 10 Opportunity Pages: Impressions Heatmap by Position', fontsize=16)
plt.xlabel('Average Position', fontsize=14)
plt.ylabel('Keywords', fontsize=14)
plt.tight_layout()
plt.savefig('figures/top_opportunity_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 7: Alternative Visualization - 3D Bar Chart
print("Creating 3D Bar Chart visualization...")
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Create 3D bars
x_pos = range(len(top_pages))
y_pos = top_pages['Position']
z_pos = [0] * len(top_pages)
dx = [0.8] * len(top_pages)
dy = [0.8] * len(top_pages)
dz = top_pages['Impressions']

# Normalize impressions for better color mapping
norm_impressions = (top_pages['Impressions'] - top_pages['Impressions'].min()) / \
                   (top_pages['Impressions'].max() - top_pages['Impressions'].min())

colors = plt.cm.viridis(norm_impressions)

# Plot 3D bars
bar3d = ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, shade=True)

# Add labels
ax.set_xlabel('Keywords', fontsize=14)
ax.set_ylabel('Position', fontsize=14)
ax.set_zlabel('Impressions', fontsize=14)
ax.set_title('Top 10 Opportunity Pages: 3D Visualization', fontsize=16)

# Add keyword labels
ax.set_xticks(range(len(top_pages)))
ax.set_xticklabels(top_pages['Keywords'], rotation=45, ha='right', fontsize=10)

# Add colorbar for impressions
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=top_pages['Impressions'].min(), 
                                                             vmax=top_pages['Impressions'].max()))
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Impressions', pad=0.1)

plt.tight_layout()
plt.savefig('figures/top_opportunity_3d_chart.png', dpi=300, bbox_inches='tight')
plt.close()

print("Creating CTR vs Position detailed analysis plot...")
plt.figure(figsize=(14, 10))

# Create position bands for analysis
position_bands = [(11, 15), (16, 20), (21, 25), (26, 30)]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Plot each position band
for i, ((min_pos, max_pos), color) in enumerate(zip(position_bands, colors)):
    band_data = df[(df['Position'] >= min_pos) & (df['Position'] < max_pos)]
    plt.scatter(band_data['Position'], band_data['CTR_clean'] * 100, 
                s=band_data['Impressions'] / 20, 
                color=color, alpha=0.7, 
                label=f'Positions {min_pos}-{max_pos-1}')

# Add trendline
x = df['Position']
y = df['CTR_clean'] * 100
z = np.polyfit(x, y, 2)
p = np.poly1d(z)
xp = np.linspace(10, 30, 100)
plt.plot(xp, p(xp), 'k--', linewidth=2, label='Polynomial Trend')

# Add annotations for key insights
plt.annotate('CTR Sweet Spot: 2-4× higher CTR', xy=(13, 0.25), xytext=(15, 0.4),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, ha='center')
plt.annotate('Sharp Drop-off Point', xy=(18, 0.15), xytext=(20, 0.3),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, ha='center')

plt.title('CTR vs Position Detailed Analysis with Performance Zones', fontsize=16)
plt.xlabel('Average Position', fontsize=14)
plt.ylabel('Click-Through Rate (%)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(title='Position Bands', fontsize=12)
plt.tight_layout()
plt.savefig('figures/ctr_position_detailed_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations created and saved successfully!")

# 5. Display the tables and plots as output
print("\n" + "="*50)
print("DISPLAYING KEY INSIGHTS")
print("="*50)

# Display top opportunity pages
print("\nTop 10 Opportunity Pages by Impressions:")
top_pages_display = top_pages[['Keywords', 'Impressions', 'Position', 'Clicks', 'CTR']]
print(top_pages_display.to_string(index=False))

# Display pages with highest CTR
print("\nTop 10 Pages by CTR:")
high_ctr_pages = df.nlargest(10, 'CTR_clean')[['Keywords', 'Impressions', 'Position', 'Clicks', 'CTR']]
print(high_ctr_pages.to_string(index=False))

# Display pages with best position but low clicks (high opportunity)
print("\nHigh Opportunity Pages (Good Position but Low Clicks):")
high_opportunity = df[(df['Position'] <= 15) & (df['Clicks'] <= 2)][['Keywords', 'Impressions', 'Position', 'Clicks', 'CTR']]
print(high_opportunity.to_string(index=False))

# Display correlation analysis
print("\nCorrelation Analysis:")
correlation_matrix = df[['Clicks', 'Impressions', 'Position', 'CTR_clean']].corr()
print(correlation_matrix)

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print("All plots have been saved in the 'figures' directory.")
print("Key insights displayed above show the best optimization opportunities.")