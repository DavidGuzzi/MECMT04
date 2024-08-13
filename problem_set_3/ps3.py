#%%
import pandas as pd
import numpy as np
import pyreadstat as st

path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECMT04\problem_set_3\firmas.dta"

df, meta = st.read_dta(path)
md = df.loc[:,['ebitass', 'rotc']]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
md_scaled = scaler.fit_transform(md)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=7, algorithm='lloyd')
clusters = kmeans.fit(md_scaled)

# Centroides y etiquetas
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Calcular WCSS (Within-cluster sum of squares by cluster)
wcss = np.zeros(2)
for i in range(2):
    cluster_points = md_scaled[labels == i]
    wcss[i] = np.sum((cluster_points - centroids[i]) ** 2)

print("Within-cluster sum of squares by cluster (WCSS):")
print(wcss)

# Calcular TSS (Total sum of squares)
overall_mean = np.mean(md_scaled, axis=0)
tss = np.sum((md_scaled - overall_mean) ** 2)

# Calcular BCSS (Between-cluster sum of squares)
bcss = tss - np.sum(wcss)

# Porcentaje de variación explicada
explained_variation = (bcss / tss) * 100

print(f"Total sum of squares (TSS): {tss:.2f}")
print(f"Between-cluster sum of squares (BCSS): {bcss:.2f}")
print(f"Porcentaje de variación explicada: {explained_variation:.1f}%")

# Visualización en 2D
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(md_scaled[:, 0], md_scaled[:, 1], c=labels, cmap='tab10')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='d', s=100, label='Centroids')
plt.title('KMeans Clustering')
plt.xlabel('EBITASS')
plt.ylabel('ROTC')
plt.legend()
plt.grid(True)
plt.show()