#%%
import pandas as pd
import numpy as np
import pyreadstat as st

path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECMT04\problem_set_3\firmas.dta"

df, meta = st.read_dta(path)
md = df.loc[:,['ebitass', 'rotc']]

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# md_scaled = scaler.fit_transform(md)

media = md.mean()
de = md.std()
md_scaled = ((md - media) / de).values

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


def clusternum_Ftest(data, maxclusters):
    n = data.shape[0]
    scdg = np.zeros(maxclusters)
    
    # Escalar los datos
    media = data.mean()
    de = data.std()
    md_scaled = ((data - media) / de).values
    
    # Calcular SCDG (Within-cluster sum of squares) para diferentes números de clusters
    for j in range(1, maxclusters + 1):
        kmeans = KMeans(n_clusters=j, random_state=7)
        kmeans.fit(md_scaled)
        scdg[j-1] = kmeans.inertia_
    
    # Crear DataFrame para almacenar resultados
    ftest_results = pd.DataFrame({
        'Clusters': np.arange(1, maxclusters + 1),
        'SCDG': scdg
    })
    
    # Calcular el F-test
    ftest_results['lead'] = ftest_results['SCDG'].shift(-1)
    ftest_results['Ftest'] = (ftest_results['SCDG'] - ftest_results['lead']) / (ftest_results['SCDG'] / (n - ftest_results['Clusters'] + 1))
    
    return ftest_results
#%%
import pandas as pd
import pyreadstat as st

path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECMT04\problem_set_3\firmas.dta"

df, meta = st.read_dta(path)
md = df.loc[:,['ebitass', 'rotc']]

media = md.mean()
de = md.std()
md_scaled = ((md - media) / de).values

from sklearn.metrics import pairwise_distances

# Cálculo de la distancia euclidiana
distance_matrix = pairwise_distances(md_scaled, metric='euclidean')

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


Z = linkage(distance_matrix, method='complete')  

plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.show()