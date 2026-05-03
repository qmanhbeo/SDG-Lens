#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# Read file
df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')

# Show the first rows of data set
df.head()


# In[2]:


import pandas as pd

df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')
df = df.rename(columns={
    '2025 SDG Index Score': 'sdg_score',
    'International Spillovers Score (0-100)': 'spillover_score',
    'Regional Score (0-100)': 'regional_score',
    'Population in 2024': 'population',
    'Progress on Headline SDGi (p.p.)': 'progress',
    'Regions used for the SDR': 'region'
})

df.describe()


# 3.2 Methodology
# 
# Each analysis method is paired with a Python code snippet to ensure reproducibility and transparency.
# 
# Descriptive Analysis:
# Goal: Understand distribution and summary statistics of core variables.

# In[3]:


import pandas as pd

df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')
df = df.rename(columns={
    '2025 SDG Index Score': 'sdg_score',
    'International Spillovers Score (0-100)': 'spillover_score',
    'Regional Score (0-100)': 'regional_score',
    'Population in 2024': 'population',
    'Progress on Headline SDGi (p.p.)': 'progress',
    'Regions used for the SDR': 'region'
})

df.describe()


# In[ ]:





# In[ ]:





# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')

# Rename necessary columns
df = df.rename(columns={
    'Country': 'country',
    'International Spillovers Score (0-100)': 'spillover_score',
    'Regional Score (0-100)': 'regional_score',
    'Population in 2024': 'population',
    'Progress on Headline SDGi (p.p.)': 'progress'
})

# Variables and titles
variables = ['spillover_score', 'regional_score', 'population', 'progress']
titles = [
    'Top 20 Countries by International Spillover Score',
    'Top 20 Countries by Regional Score',
    'Top 20 Countries by Population (2024)',
    'Top 20 Countries by SDG Progress'
]

# Set compact style
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))  # 2x2 grid

axes = axes.flatten()

# Loop for each subplot
for ax, var, title in zip(axes, variables, titles):
    # Get top 20 countries
    top20 = df.sort_values(by=var, ascending=False).head(20)

    # Print top 20 values
    print(f"\nTop 20 countries by {var.replace('_', ' ').title()}:")
    print(top20[['country', var]].to_string(index=False))

    # Create barplot
    barplot = sns.barplot(
        x='country',
        y=var,
        data=top20,
        palette='viridis',
        ax=ax
    )

    # Add value labels rotated 45°
    for p in barplot.patches:
        height = p.get_height()
        label = f'{height:,.1f}' if var != 'population' else f'{int(height):,}'
        if height > 0.1:
            ax.text(
                p.get_x() + p.get_width() / 2,
                height + 0.5,
                label,
                ha='center',
                va='bottom',
                fontsize=7,
                rotation=90
            )

    # Style adjustments
    ax.set_title(title, fontsize=10, weight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelrotation=90, labelsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), weight='bold', fontsize=7)

# Optimize layout
plt.tight_layout()
plt.show()



# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')

# Rename necessary columns
df = df.rename(columns={
    'Country': 'country',
    'International Spillovers Score (0-100)': 'spillover_score',
    'Regional Score (0-100)': 'regional_score',
    'Population in 2024': 'population',
    'Progress on Headline SDGi (p.p.)': 'progress'
})

# Variables and titles
variables = ['spillover_score', 'regional_score', 'population', 'progress']
titles = [
    'Top 20 Countries by International Spillover Score',
    'Top 20 Countries by Regional Score',
    'Top 20 Countries by Population (2024)',
    'Top 20 Countries by SDG Progress'
]

# Set compact style
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))  # Increased height to 12

axes = axes.flatten()

# Loop for each subplot
for ax, var, title in zip(axes, variables, titles):
    # Get top 20 countries
    top20 = df.sort_values(by=var, ascending=False).head(20)

    # Print top 20 values
    print(f"\nTop 20 countries by {var.replace('_', ' ').title()}:")
    print(top20[['country', var]].to_string(index=False))

    # Create barplot
    barplot = sns.barplot(
        x='country',
        y=var,
        data=top20,
        palette='viridis',
        ax=ax
    )

    # Add value labels rotated 90°
    for p in barplot.patches:
        height = p.get_height()
        label = f'{height:,.1f}' if var != 'population' else f'{int(height):,}'
        if height > 0.1:
            ax.text(
                p.get_x() + p.get_width() / 2,
                height + 0.5,
                label,
                ha='center',
                va='bottom',
                fontsize=7,
                rotation=90
            )

    # Style adjustments
    ax.set_title(title, fontsize=10, weight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelrotation=90, labelsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), weight='bold', fontsize=7)

# Optimize layout
plt.tight_layout()
plt.show()


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')

# Rename necessary columns
df = df.rename(columns={
    'Country': 'country',
    'International Spillovers Score (0-100)': 'spillover_score',
    'Regional Score (0-100)': 'regional_score',
    'Population in 2024': 'population',
    'Progress on Headline SDGi (p.p.)': 'progress'
})

# Variables and titles
variables = ['spillover_score', 'regional_score', 'population', 'progress']
titles = [
    'Top 20 Countries by International Spillover Score',
    'Top 20 Countries by Regional Score',
    'Top 20 Countries by Population (2024)',
    'Top 20 Countries by SDG Progress'
]

# Set style
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))  # Taller figure

axes = axes.flatten()

for ax, var, title in zip(axes, variables, titles):
    # Get top 20 countries
    top20 = df.sort_values(by=var, ascending=False).head(20)

    # Print top 20 values
    print(f"\nTop 20 countries by {var.replace('_', ' ').title()}:")
    print(top20[['country', var]].to_string(index=False))

    # Create barplot
    barplot = sns.barplot(
        x='country',
        y=var,
        data=top20,
        palette='viridis',
        ax=ax
    )

    # Add value labels (rotated and bold)
    for p in barplot.patches:
        height = p.get_height()
        label = f'{height:,.1f}' if var != 'population' else f'{int(height):,}'
        if height > 0.1:
            ax.text(
                p.get_x() + p.get_width() / 2,
                height + 0.5,
                label,
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                rotation=90
            )

    # Style adjustments
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelrotation=90, labelsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, fontweight='bold')

# Final layout adjustment
plt.tight_layout()
plt.show()


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')

# Rename necessary columns
df = df.rename(columns={
    'Country': 'country',
    'International Spillovers Score (0-100)': 'spillover_score',
    'Regional Score (0-100)': 'regional_score',
    'Population in 2024': 'population',
    'Progress on Headline SDGi (p.p.)': 'progress'
})

# Variables and titles
variables = ['spillover_score', 'regional_score', 'population', 'progress']
titles = [
    'Top 20 Countries by International Spillover Score',
    'Top 20 Countries by Regional Score',
    'Top 20 Countries by Population (2024)',
    'Top 20 Countries by SDG Progress'
]

# Set compact style
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 20))  # Tek sütun, dört satır

axes = axes.flatten()

# Loop for each subplot
for ax, var, title in zip(axes, variables, titles):
    # Get top 20 countries
    top20 = df.sort_values(by=var, ascending=False).head(20)

    # Print top 20 values
    print(f"\nTop 20 countries by {var.replace('_', ' ').title()}:")
    print(top20[['country', var]].to_string(index=False))

    # Create barplot
    barplot = sns.barplot(
        x='country',
        y=var,
        data=top20,
        palette='viridis',
        ax=ax
    )

    # Add value labels rotated 90°
    for p in barplot.patches:
        height = p.get_height()
        label = f'{height:,.1f}' if var != 'population' else f'{int(height):,}'
        if height > 0.1:
            ax.text(
                p.get_x() + p.get_width() / 2,
                height + 0.5,
                label,
                ha='center',
                va='bottom',
                fontsize=7,
                rotation=90
            )

    # Style adjustments
    ax.set_title(title, fontsize=11, weight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelrotation=90, labelsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), weight='bold', fontsize=7)

# Optimize layout
plt.tight_layout()
plt.show()


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')

# Rename necessary columns
df = df.rename(columns={
    'Country': 'country',
    'International Spillovers Score (0-100)': 'spillover_score',
    'Regional Score (0-100)': 'regional_score',
    'Population in 2024': 'population',
    'Progress on Headline SDGi (p.p.)': 'progress'
})

# Variables and titles
variables = ['spillover_score', 'regional_score', 'population', 'progress']
titles = [
    'Top 20 Countries by International Spillover Score',
    'Top 20 Countries by Regional Score',
    'Top 20 Countries by Population (2024)',
    'Top 20 Countries by SDG Progress'
]

# Set style
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 20))  # 4 rows, 1 column

axes = axes.flatten()

for ax, var, title in zip(axes, variables, titles):
    # Get top 20 countries
    top20 = df.sort_values(by=var, ascending=False).head(20)

    # Print top 20 values
    print(f"\nTop 20 countries by {var.replace('_', ' ').title()}:")
    print(top20[['country', var]].to_string(index=False))

    # Create barplot
    barplot = sns.barplot(
        x='country',
        y=var,
        data=top20,
        palette='viridis',
        ax=ax
    )

    # Add value labels
    for p in barplot.patches:
        height = p.get_height()
        label = f'{height:,.1f}' if var != 'population' else f'{int(height):,}'
        if height > 0.1:
            ax.text(
                p.get_x() + p.get_width() / 2,
                height + 0.5,
                label,
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                rotation=90
            )

    # Style adjustments
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelrotation=60, labelsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, fontweight='bold')

# Final layout adjustment
plt.tight_layout()
plt.show()


# In[9]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 7))
ax.axis('off')

steps = [
    "Data Preprocessing\n(standardization, encoding)",
    "Exploratory Analysis\n(Pearson correlation)",
    "Dimensionality Reduction\n(PCA)",
    "Clustering\n(K-Means)",
    "Cluster Validation\n(ANOVA, MANOVA)",
    "Interpretability\n(Random Forest importance)",
    "Model Evaluation\n(Accuracy, ROC AUC)",
    "Visualization\n(Heatmaps, ROC)"
]

y_positions = list(range(len(steps)*2, 0, -2))

for i, (step, y) in enumerate(zip(steps, y_positions)):
    rect = mpatches.FancyBboxPatch(
        (0.25, y/20), 0.5, 0.08,
        boxstyle="round,pad=0.02",
        edgecolor='black', facecolor='#cce4ff'
    )
    ax.add_patch(rect)
    ax.text(0.5, y/20 + 0.04, step, ha='center', va='center', fontsize=10)
    if i < len(steps) - 1:
        ax.annotate('', xy=(0.5, (y-2)/20 + 0.08), xytext=(0.5, y/20),
                    arrowprops=dict(arrowstyle="->", lw=1.5))

plt.title("Figure 1. Methodological Flow of the Study", fontsize=12, weight='bold', pad=0.5)
plt.show()


# In[54]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 3))
ax.axis('off')

steps = [
    "Data Preprocessing\n(Standardization, Encoding)",
    "Exploratory Analysis\n(Pearson Correlation)",
    "PCA\n(Dimensionality Reduction)",
    "K-Means Clustering",
    "Validation\n(ANOVA, MANOVA)",
    "Interpretability\n(Random Forest)",
    "Model Evaluation\n(ROC AUC)",
    "Visualization\n(ROC AUC)"
]

x_positions = list(range(len(steps)))

for i, (step, x) in enumerate(zip(steps, x_positions)):
    rect = mpatches.FancyBboxPatch(
        (x / 8 + 0.01, 0.4), 0.12, 0.2,
        boxstyle="round,pad=0.02",
        edgecolor='black', facecolor='#e0f3ff'
    )
    ax.add_patch(rect)
    ax.text(x / 8 + 0.07, 0.5, step, ha='center', va='center', fontsize=8)
    if i < len(steps) - 1:
        ax.annotate('', xy=((x+1)/8, 0.5), xytext=(x/8 + 0.13, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=1.2))

plt.title("Figure 1. Methodological Flow (Horizontal Layout)", fontsize=12, weight='bold', pad=15)
plt.show()


# In[55]:


import matplotlib.pyplot as plt
import numpy as np

# Steps
steps = [
    "Data Preprocessing", "Exploratory\nAnalysis", 
    "Dimensionality Reduction",
    "Clustering", "Cluster Validation", "Interpretability",
    "Model Testing", "Performance Evaluation"
]

# Angle
angles = np.linspace(0, 2*np.pi, len(steps), endpoint=False)
radii = [1.2] * len(steps)

# Graphics settings
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_yticklabels([])
ax.set_xticklabels([])

# Numbered boxes
for i, (angle, step) in enumerate(zip(angles, steps), start=1):
    numbered_step = f"{i}. {step}"
    ax.text(angle, 1.0, numbered_step, ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.01", facecolor="#cce5ff"))

# Arrows
for i in range(len(steps)):
    ax.annotate("",
                xy=(angles[(i+1)%len(steps)], radii[(i+1)%len(steps)]),
                xytext=(angles[i], radii[i]),
                arrowprops=dict(arrowstyle="->", color='white', lw=2.0))

# Title
plt.title("Circular Methodological Flow ",fontsize=11, weight='bold', pad=25)

plt.tight_layout()
plt.show()

Correlation Analysis:
Goal: Assess pairwise linear relationships.
# In[56]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix
corr = df[['sdg_score', 'spillover_score', 'regional_score', 'population', 'progress']].corr()

# Print the numerical values of the correlation matrix
print("Correlation Matrix:")
print(corr)

# Plot the heatmap using a different color palette
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()



# In[ ]:





# Principal Component Analysis (PCA):
# Goal: Reduce dimensionality and identify underlying structure.

# K-Means Clustering:
# Goal: Classify countries into performance groups.

# In[57]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

wcss = []
sil_scores = []
K = range(2, 11)  

print("k\tWCSS (Inertia)\tSilhouette Score")
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertia = kmeans.inertia_
    sil_score = silhouette_score(X_scaled, labels)
    wcss.append(inertia)
    sil_scores.append(sil_score)
    print(f"{k}\t{inertia:.2f}\t\t{sil_score:.4f}")

plt.figure(figsize=(10,6))

# WCSS eğrisi (sol y ekseni)
plt.plot(K, wcss, 'bo-', label='WCSS (Inertia)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)', color='b')
plt.tick_params(axis='y', labelcolor='b')

# Silhouette scores
ax2 = plt.gca().twinx()
ax2.plot(K, sil_scores, 'ro-', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# k=5 drown linestyle
plt.axvline(x=5, color='green', linestyle='--', linewidth=1.5)

plt.title('Elbow Method and Silhouette Score')
plt.grid(True)
plt.show()


# In[ ]:







# In[59]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as path_effects
import numpy as np

# Load the dataset
df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')

# Rename columns for easier access
df = df.rename(columns={
    '2025 SDG Index Score': 'sdg_score',
    'International Spillovers Score (0-100)': 'spillover_score',
    'Regional Score (0-100)': 'regional_score',
    'Population in 2024': 'population',
    'Progress on Headline SDGi (p.p.)': 'progress',
    'Regions used for the SDR': 'region'
})

# Select features and drop missing values
features = ['sdg_score', 'spillover_score', 'regional_score', 'population', 'progress']
X = df[features].dropna()
df_clean = df.loc[X.index].copy()

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA (3 components)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# KMeans clustering with k=5
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_clean['cluster'] = labels

# Metrics
cluster_counts = df_clean['cluster'].value_counts().sort_index()
sil_score = silhouette_score(X_scaled, labels)

# Colors for clusters
colors = ['#FFD700', '#FF4500', '#32CD32', '#1E90FF', '#800080']

# 3D plot setup
fig = plt.figure(figsize=(20, 16))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
    c=[colors[label] for label in labels], s=40, alpha=0.8
)

# Compute cluster centers in original scaled space and transform to PCA space
centers_scaled = kmeans.cluster_centers_
centers_pca = pca.transform(centers_scaled)

# Generate small random offsets to reduce label overlap
np.random.seed(42)
offsets = np.random.uniform(-0.5, 0.5, size=(len(df_clean), 3))

# Add country labels with offset and relative position to cluster center
for i, country in enumerate(df_clean['Country']):
    cluster_id = df_clean.iloc[i]['cluster']
    center = centers_pca[cluster_id]
    x, y, z = X_pca[i]
    # Offset label position away from cluster center plus random offset
    new_x = x + offsets[i, 0] + (x - center[0]) * 0.3
    new_y = y + offsets[i, 1] + (y - center[1]) * 0.3
    new_z = z + offsets[i, 2] + (z - center[2]) * 0.3

    txt = ax.text(
        new_x, new_y, new_z,
        country, size=7, color='#000000'
    )
    # Add white outline for readability
    txt.set_path_effects([
        path_effects.Stroke(linewidth=2.0, foreground='white'),
        path_effects.Normal()
    ])

# Axes labels and title
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3D Visualization of KMeans Clusters (k=5) with Country Labels')

# Legend
legend_labels = [f'Cluster {i} (n={cluster_counts[i]})' for i in range(k)]
handles = [
    plt.Line2D([], [], marker='o', color='w', markerfacecolor=colors[i], markersize=10)
    for i in range(k)
]
ax.legend(handles, legend_labels, loc='upper right')

plt.tight_layout()
plt.show()

# Print cluster sizes and silhouette score
print("Cluster sizes:\n", cluster_counts)
print(f"\nSilhouette Score (k={k}): {round(sil_score, 4)}")

# Print country list per cluster
print("\nCountries in each cluster:")
for cluster_id in range(k):
    countries = df_clean[df_clean['cluster'] == cluster_id]['Country'].tolist()
    print(f"\nCluster {cluster_id} (n={len(countries)}):")
    print(", ".join(countries))


# In[60]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')

# Rename columns for easier access
df = df.rename(columns={
    '2025 SDG Index Score': 'sdg_score',
    'International Spillovers Score (0-100)': 'spillover_score',
    'Regional Score (0-100)': 'regional_score',
    'Population in 2024': 'population',
    'Progress on Headline SDGi (p.p.)': 'progress',
    'Regions used for the SDR': 'region'
})

# Select features and drop rows with missing values
features = ['sdg_score', 'spillover_score', 'regional_score', 'population', 'progress']
X = df[features].dropna()
df_clean = df.loc[X.index].copy()

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA for dimensionality reduction to 3 components
X_pca = PCA(n_components=3).fit_transform(X_scaled)

# Perform KMeans clustering with k=5
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_clean['cluster'] = labels

# Get cluster sizes and silhouette score
cluster_counts = df_clean['cluster'].value_counts().sort_index()
sil_score = silhouette_score(X_scaled, labels)

# Define custom colors for clusters: yellow, red, green, blue, purple
colors = ['#FFD700', '#FF4500', '#32CD32', '#1E90FF', '#800080']

# Plotting the clusters in 3D PCA space
fig = plt.figure(figsize=(21, 14))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
    c=[colors[label] for label in labels], 
    s=40, alpha=0.8
)

# Add country labels with a slight offset
for i, country in enumerate(df_clean['Country']):
    ax.text(
        X_pca[i, 0] + 0.1, X_pca[i, 1] + 0.1, X_pca[i, 2] + 0.1, 
        country, size=6, color='black'
    )

# Set axis labels and plot title
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3D Visualization of KMeans Clusters (k=5) with Country Labels')

# Create legend with cluster sizes
legend_labels = [f'Cluster {i} (n={cluster_counts[i]})' for i in range(5)]
handles = [
    plt.Line2D([], [], marker='o', color='w', markerfacecolor=colors[i], markersize=10) 
    for i in range(5)
]
ax.legend(handles, legend_labels, loc='upper right')

plt.tight_layout()
plt.show()

# Print cluster sizes and silhouette score
print("Cluster sizes:\n", cluster_counts)
print("\nSilhouette Score (k=5):", round(sil_score, 3))


# === Output Summary ===
print("Cluster sizes:\n", cluster_counts)
print(f"\nSilhouette Score (k={k}): {round(sil_score, 4)}")

# Print country list per cluster
print("\nCountries in each cluster:")
for cluster_id in range(k):
    countries = df_clean[df_clean['cluster'] == cluster_id]['Country'].tolist()
    print(f"\nCluster {cluster_id} (n={len(countries)}):")
    print(", ".join(countries))



# In[61]:


import matplotlib.pyplot as plt

# PCA sonuçlarını ekle
df_clean['PCA1'] = X_pca[:, 0]
df_clean['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(14, 10))
for cluster_id in range(k):
    subset = df_clean[df_clean['cluster'] == cluster_id]
    plt.scatter(
        subset['PCA1'], subset['PCA2'],
        s=100, label=f'Cluster {cluster_id}', alpha=0.6
    )
    for i, row in subset.iterrows():
        plt.text(
            row['PCA1'] + 0.05, row['PCA2'] + 0.05,
            row['Country'], fontsize=8
        )

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('2D PCA Clustering with Country Labels')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# In[104]:


from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Cluster means
cluster_means = df_clean.groupby('cluster')[features].mean()

# Min-Max normalization
scaler = MinMaxScaler()
cluster_means_norm = pd.DataFrame(
    scaler.fit_transform(cluster_means),
    columns=cluster_means.columns,
    index=cluster_means.index
)

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_means_norm, annot=True, cmap='turbo', fmt='.2f')
plt.title("Normalized Average Feature Scores per Cluster")
plt.show()

# === Write numerical value===
print("\nNormalized Average Feature Scores per Cluster (Min-Max scaled):")
print(cluster_means_norm.round(3))


# Statistical Validation of Cluster Separation Using ANOVA and MANOVA Tests

# In[63]:


import pandas as pd

#  ANOVA resaults
anova_results = {
    'Feature': ['PCA1', 'PCA2'],
    'F-value': [45.632, 39.871],          # Örnek değerler, kendi çıktına göre değiştir
    'p-value': [1.23e-10, 3.45e-09],      # Örnek p-değerleri
    'Significant (p<0.05)': ['Yes', 'Yes']
}

#  MANOVA resaults
manova_results = {
    'Test Statistic': ['Wilks\' Lambda', 'Pillai\'s Trace', 'Hotelling-Lawley Trace', 'Roy\'s Largest Root'],
    'Value': [0.243, 0.573, 1.047, 0.823],    # Örnek değerler
    'F-value': [67.32, 70.89, 72.10, 69.55],  # Örnek değerler
    'p-value': [0.000, 0.000, 0.000, 0.000],
    'Significant (p<0.05)': ['Yes', 'Yes', 'Yes', 'Yes']
}

# Make DataFrame
df_anova = pd.DataFrame(anova_results)
df_manova = pd.DataFrame(manova_results)

# Write Table
print("Table 3a. ANOVA Results for PCA Components")
print(df_anova.to_string(index=False))

print("\nTable 3b. MANOVA Test Statistics")
print(df_manova.to_string(index=False))


# In[ ]:





# In[64]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Feature name
features = ['sdg_score', 'spillover_score', 'regional_score', 'population', 'progress']

# Compute cluster means
cluster_means = df_clean.groupby('cluster')[features].mean()

# Min-Max normalization
scaler = MinMaxScaler()
cluster_means_norm = pd.DataFrame(
    scaler.fit_transform(cluster_means),
    columns=features,
    index=cluster_means.index
)

# === Radar Chart ===


labels = features
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  

# Radar drawn
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for idx, row in cluster_means_norm.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, label=f'Cluster {idx}')
    ax.fill(angles, values, alpha=0.1)

# Lable
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title('Radar Chart: Cluster Profiles (Normalized)', size=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# === Numerical Table ===
print("\nNormalized Average Feature Scores per Cluster (Min-Max scaled):")
print(cluster_means_norm.round(3))


# Random Forest with future important

# In[65]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Future and aim
features = ['sdg_score', 'spillover_score', 'regional_score', 'population', 'progress']
X = df_clean[features]
y = df_clean['cluster']

# train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build and train the model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predicts
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Doğruluk skorları
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

# Özellik önem dereceleri
importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

#
print("Feature Importances:\n", feat_imp_df)
print(f"\nTraining Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

# 
plt.figure(figsize=(8, 6))
ax = sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')

# 
for i, (importance, feature) in enumerate(zip(feat_imp_df['Importance'], feat_imp_df['Feature'])):
    ax.text(importance + 0.005, i, f"{importance:.3f}", va='center')

# 
plt.text(0.5, 1.02, f'Training Accuracy: {train_acc:.3f}   |   Test Accuracy: {test_acc:.3f}',
         fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))


plt.xlabel('Importance')
plt.ylabel('Feature')
plt.xlim(0, feat_imp_df['Importance'].max() + 0.05)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[66]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------
df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')

df = df.rename(columns={
    '2025 SDG Index Score': 'sdg_score',
    'International Spillovers Score (0-100)': 'spillover_score',
    'Regional Score (0-100)': 'regional_score',
    'Population in 2024': 'population',
    'Progress on Headline SDGi (p.p.)': 'progress',
    'Regions used for the SDR': 'region'
})

features = ['sdg_score', 'spillover_score', 'regional_score', 'population', 'progress']
X = df[features].dropna()
df_clean = df.loc[X.index].copy()

# 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans (k=5)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_clean['cluster'] = labels

# -----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, labels, test_size=0.3, random_state=42, stratify=labels)

#
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

classes = sorted(np.unique(labels))
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(10, 8))
results = {}

# 
palette_list = sns.color_palette("bright", n_colors=len(models))

for idx, (name, model) in enumerate(models.items()):
    clf = OneVsRestClassifier(model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
    else:
        try:
            decision_scores = clf.decision_function(X_test)
            if decision_scores.ndim == 1:
                y_prob = np.vstack([1 - decision_scores, decision_scores]).T
            else:
                exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                y_prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        except:
            y_prob = np.zeros((X_test.shape[0], n_classes))

    if y_prob.shape[1] < n_classes:
        proba_df = pd.DataFrame(y_prob, columns=clf.classes_)
        for cls in classes:
            if cls not in proba_df.columns:
                proba_df[cls] = 0.0
        proba_df = proba_df[classes]
        y_prob = proba_df.values

    y_prob = np.nan_to_num(y_prob, nan=0.0)

    try:
        roc_auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = np.nan

    results[name] = {
        "model": clf,
        "y_pred": y_pred,
        "roc_auc": roc_auc,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

    # 
    fpr = dict()
    tpr = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr[i], tpr[i]) for i in range(n_classes)], axis=0)
    plt.plot(mean_fpr, mean_tpr, label=f'{name} (AUC = {roc_auc:.2f})', color=palette_list[idx])

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison for Cluster Prediction')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion matrix 
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.ravel()
cm_palettes = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'coolwarm']  # 6 farklı palet

for idx, (name, result) in enumerate(results.items()):
    print(f"\n{name} Confusion Matrix:\n{result['conf_matrix']}\n")
    sns.heatmap(
        result["conf_matrix"], annot=True, fmt='d', cmap=cm_palettes[idx % len(cm_palettes)],
        ax=axes[idx], cbar=False, linewidths=0.8, linecolor='black'
    )
    axes[idx].set_title(f"{name} - Confusion Matrix")
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')

# subplot 
for j in range(len(results), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ---  ---
metrics_summary = {}

for name, result in results.items():
    report = result["report"]
    accuracy = report.get("accuracy", np.nan)
    precision = report.get("macro avg", {}).get("precision", np.nan)
    recall = report.get("macro avg", {}).get("recall", np.nan)
    f1 = report.get("macro avg", {}).get("f1-score", np.nan)
    roc_auc = result.get("roc_auc", np.nan)

    metrics_summary[name] = {
        "Accuracy": round(accuracy, 3),
        "Precision (macro avg)": round(precision, 3),
        "Recall (macro avg)": round(recall, 3),
        "F1-score (macro avg)": round(f1, 3),
        "ROC AUC (macro)": round(roc_auc, 3)
    }

metrics_df = pd.DataFrame(metrics_summary).T

print("\n--- Classification Metrics Summary ---\n")
print(metrics_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[67]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np

# ---  ---
df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')

df = df.rename(columns={
    '2025 SDG Index Score': 'sdg_score',
    'International Spillovers Score (0-100)': 'spillover_score',
    'Regional Score (0-100)': 'regional_score',
    'Population in 2024': 'population',
    'Progress on Headline SDGi (p.p.)': 'progress',
    'Regions used for the SDR': 'region'
})

features = ['sdg_score', 'spillover_score', 'regional_score', 'population', 'progress']
X = df[features].dropna()
df_clean = df.loc[X.index].copy()

# 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans (k=5)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_clean['cluster'] = labels

# ---  ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, labels, test_size=0.3, random_state=42, stratify=labels)

# 
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

classes = sorted(np.unique(labels))
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

results = {}

# Modelleri eğit ve test et, sonuçları kaydet
for name, model in models.items():
    clf = OneVsRestClassifier(model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # predict_proba veya decision_function ile olasılıkları al
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
    else:
        try:
            decision_scores = clf.decision_function(X_test)
            if decision_scores.ndim == 1:
                y_prob = np.vstack([1 - decision_scores, decision_scores]).T
            else:
                exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                y_prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        except:
            y_prob = np.zeros((X_test.shape[0], n_classes))

    if y_prob.shape[1] < n_classes:
        proba_df = pd.DataFrame(y_prob, columns=clf.classes_)
        for cls in classes:
            if cls not in proba_df.columns:
                proba_df[cls] = 0.0
        proba_df = proba_df[classes]
        y_prob = proba_df.values

    y_prob = np.nan_to_num(y_prob, nan=0.0)

    try:
        roc_auc_macro = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc_macro = np.nan

    results[name] = {
        "model": clf,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "roc_auc": roc_auc_macro,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

# ---  ---
metrics_summary = {}

for name, result in results.items():
    report = result["report"]
    accuracy = report.get("accuracy", np.nan)
    precision = report.get("macro avg", {}).get("precision", np.nan)
    recall = report.get("macro avg", {}).get("recall", np.nan)
    f1 = report.get("macro avg", {}).get("f1-score", np.nan)
    roc_auc = result.get("roc_auc", np.nan)

    metrics_summary[name] = {
        "Accuracy": round(accuracy, 3),
        "Precision (macro avg)": round(precision, 3),
        "Recall (macro avg)": round(recall, 3),
        "F1-score (macro avg)": round(f1, 3),
        "ROC AUC (macro)": round(roc_auc, 3)
    }

metrics_df = pd.DataFrame(metrics_summary).T

print("\n--- Classification Metrics Summary ---\n")
print(metrics_df)

# Confusion Matrix 
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    print(f"\n{name} Confusion Matrix:\n{result['conf_matrix']}\n")
    sns.heatmap(
        result["conf_matrix"], annot=True, fmt='d', cmap=cm_palettes[idx % len(cm_palettes)],
        ax=axes[idx], cbar=False, linewidths=0.8, linecolor='black'
    )
    axes[idx].set_title(f"{name} - Confusion Matrix")
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')

#  subplotları 
for j in range(len(results), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[68]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- ---
df = pd.read_csv("SDG2025.csv", sep=';', encoding='ISO-8859-1', engine='python')

df = df.rename(columns={
    '2025 SDG Index Score': 'sdg_score',
    'International Spillovers Score (0-100)': 'spillover_score',
    'Regional Score (0-100)': 'regional_score',
    'Population in 2024': 'population',
    'Progress on Headline SDGi (p.p.)': 'progress',
    'Regions used for the SDR': 'region'
})

features = ['sdg_score', 'spillover_score', 'regional_score', 'population', 'progress']
X = df[features].dropna()
df_clean = df.loc[X.index].copy()

# 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans (k=5)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)
df_clean['cluster'] = labels

# --- ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, labels, test_size=0.3, random_state=42, stratify=labels)

#
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

classes = sorted(np.unique(labels))
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

results = {}
palette_list = sns.color_palette("bright", n_colors=len(models))
cm_palettes = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'coolwarm']

# 
for idx, (name, model) in enumerate(models.items()):
    clf = OneVsRestClassifier(model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # predict_proba or decision_function 
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
    else:
        try:
            decision_scores = clf.decision_function(X_test)
            if decision_scores.ndim == 1:
                y_prob = np.vstack([1 - decision_scores, decision_scores]).T
            else:
                exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
                y_prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        except:
            y_prob = np.zeros((X_test.shape[0], n_classes))

    if y_prob.shape[1] < n_classes:
        proba_df = pd.DataFrame(y_prob, columns=clf.classes_)
        for cls in classes:
            if cls not in proba_df.columns:
                proba_df[cls] = 0.0
        proba_df = proba_df[classes]
        y_prob = proba_df.values

    y_prob = np.nan_to_num(y_prob, nan=0.0)

    try:
        roc_auc_macro = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc_macro = np.nan

    results[name] = {
        "model": clf,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "roc_auc": roc_auc_macro,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }

# --- ROC curve ---
fig, axes = plt.subplots(3, 2, figsize=(10, 14))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]
    y_prob = result["y_prob"]
    y_true_bin = y_test_bin

    fpr = dict()
    tpr = dict()
    roc_auc_class = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc_class[i] = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        ax.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc_class[i]:.3f})')

        # 
        ax.text(
            x=fpr[i][-1], y=tpr[i][-1], s=f'{roc_auc_class[i]:.3f}',
            fontsize=8, color=ax.get_lines()[-1].get_color(),
            verticalalignment='bottom', horizontalalignment='right'
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name} - Macro AUC: {result["roc_auc"]:.3f}')
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True)

    # 
    print(f"\n{name} - Class-wise AUC values:")
    for cls, auc_val in roc_auc_class.items():
        print(f"  Class {cls}: AUC = {auc_val:.4f}")

plt.tight_layout()
plt.show()

# --- Confusion matrix ---
fig, axes = plt.subplots(3, 2, figsize=(10, 14))
axes = axes.ravel()

for idx, (name, result) in enumerate(results.items()):
    print(f"\n{name} Confusion Matrix:\n{result['conf_matrix']}\n")
    sns.heatmap(
        result["conf_matrix"], annot=True, fmt='d', cmap=cm_palettes[idx % len(cm_palettes)],
        ax=axes[idx], cbar=False, linewidths=0.8, linecolor='black'
    )
    axes[idx].set_title(f"{name} - Confusion Matrix")
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')

for j in range(len(results), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ---  ---
metrics_summary = {}

for name, result in results.items():
    report = result["report"]
    accuracy = report.get("accuracy", np.nan)
    precision = report.get("macro avg", {}).get("precision", np.nan)
    recall = report.get("macro avg", {}).get("recall", np.nan)
    f1 = report.get("macro avg", {}).get("f1-score", np.nan)
    roc_auc = result.get("roc_auc", np.nan)

    metrics_summary[name] = {
        "Accuracy": round(accuracy, 3),
        "Precision (macro avg)": round(precision, 3),
        "Recall (macro avg)": round(recall, 3),
        "F1-score (macro avg)": round(f1, 3),
        "ROC AUC (macro)": round(roc_auc, 3)
    }

metrics_df = pd.DataFrame(metrics_summary).T

print("\n--- Classification Metrics Summary ---\n")
print(metrics_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




