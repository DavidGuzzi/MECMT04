#%%
import pandas as pd
import numpy as np
import pyreadstat as st

pd.set_option('display.float_format', '{:.4f}'.format)

path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECMT04\problem_set_2\ine.dta"

df, meta = st.read_dta(path)
ndf = df.iloc[:,1:]
cov = ndf.cov()
cor = ndf.corr()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
ndf_scaled = scaler.fit_transform(ndf)

pca = PCA()
componentes = pca.fit_transform(ndf_scaled)

varianza = pca.explained_variance_
std_componentes = np.sqrt(pca.explained_variance_)
varianza_explicada = pca.explained_variance_ratio_
varianza_explicada_acum = np.cumsum(pca.explained_variance_ratio_)
coeficientes =  pca.components_.T

#Generamos una tabla resumen
data_var = {
    'Varianza (eigenvalues)': varianza,
    'Desviación Estándar': std_componentes,
    'Varianza Explicada': varianza_explicada,
    'Varianza Explicada Acumulada': varianza_explicada_acum
}

variables_PCA = pd.DataFrame(data_var, index=[f'Componente {i+1}' for i in range(len(varianza))]).reset_index()


coeficientespca = pd.DataFrame(coeficientes, columns=[f'Coeficiente (eigenvector) {i+1} ' for i in range(coeficientes.shape[1])], index=ndf.columns).reset_index()
componentespca = pd.DataFrame(componentes, columns=[f'Componente {i+1} ' for i in range(coeficientes.shape[1])], index=df.iloc[:,0]).reset_index()

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.scatter(componentespca.iloc[:,1], componentespca.iloc[:,2], marker='o', linestyle='--')
# plt.xlabel('Componente 1')
# plt.ylabel('Componente 2')
# plt.title('Componente 1 y 2')

# for i, txt in enumerate(componentespca.iloc[:,0]):
#     plt.annotate(txt, (componentespca.iloc[i, 1], componentespca.iloc[i, 2]))

# plt.show()

# cor_ndf_with_components = pd.concat([ndf, pd.DataFrame(componentespca).iloc[:,1:3]], axis=1).corr().iloc[:-2, -2:]


# Calcular la desviación estándar de cada columna
# std_devs = np.std(componentespca.iloc[:,1:], axis=0, ddof=1)

# Crear la matriz diagonal con las desviaciones estándar
# D = np.diag(std_devs)

# # Calcular la matriz diagonal elevada a la -1/2
# D_inv_sqrt = np.diag(1 / std_devs)

# Multiplicar la matriz A por la matriz D_inv_sqrt para normalizar
# A_normalized = componentespca.iloc[:,1:] / (std_devs ^ (-0.5))
# A_normalized = pd.DataFrame(A_normalized)
import matplotlib.pyplot as plt
import seaborn as sns

# plt.figure(figsize=(10, 7))
# sns.scatterplot(x=componentespca.iloc[:,1], y=componentespca.iloc[:,2], data=componentespca, s=100, color='blue')

# for i in range(coeficientes.shape[0]):
#     plt.arrow(0, 0, coeficientes[i, 0]*max(componentespca.iloc[:,1]), coeficientes[i, 1]*max(componentespca.iloc[:,2]),
#               color='red', head_width=0.1, head_length=0.1)
#     plt.text(coeficientes[i, 0]*max(componentespca.iloc[:,1]*1.2), coeficientes[i, 1]*max(componentespca.iloc[:,2])*1.2,
#              ndf.columns[i], color='green', ha='center', va='center', fontsize=12)

# plt.axhline(0, color='gray', lw=0.5)
# plt.axvline(0, color='gray', lw=0.5)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('Biplot de las dos primeras componentes principales')
# plt.grid(True)
# plt.show()

# Configurar la figura con dos subplots en una fila
# fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# # Primer gráfico: scatterplot de las componentes principales
# sns.scatterplot(x=componentespca.iloc[:, 1], y=componentespca.iloc[:, 2], ax=axes[0], s=100, color='blue')
# axes[0].axhline(0, color='gray', lw=0.5)
# axes[0].axvline(0, color='gray', lw=0.5)
# axes[0].set_xlabel('PC1')
# axes[0].set_ylabel('PC2')
# axes[0].set_title('Componentes principales')

# # Segundo gráfico: vectores de los coeficientes (cargas)
# for i in range(coeficientes.shape[0]):
#     axes[1].arrow(0, 0, coeficientes[i, 0], coeficientes[i, 1],
#                   color='red', head_width=0.01, head_length=0.01)
#     axes[1].text(coeficientes[i, 0]*1.1, coeficientes[i, 1]*1.1,
#                  ndf.columns[i], color='green', ha='center', va='center', fontsize=12)

# axes[1].axhline(0, color='gray', lw=0.5)
# axes[1].axvline(0, color='gray', lw=0.5)
# axes[1].set_xlabel('PC1')
# axes[1].set_ylabel('PC2')
# axes[1].set_title('Vectores de cargas (loadings)')

# # Ajustar el espacio entre los subplots
# plt.tight_layout()

# plt.show()

# def biplot(score,coeff,pcax,pcay,labels=None):
#     pca1=pcax-1
#     pca2=pcay-1
#     xs = score[:,pca1]
#     ys = score[:,pca2]
#     n=score.shape[1]
#     scalex = 1.5/(xs.max()- xs.min())
#     scaley = 1.5/(ys.max()- ys.min())
#     plt.scatter(xs*scalex,ys*scaley)
#     for i in range(n):
#         plt.arrow(0, 0, coeff[i,pca1], coeff[i,pca2],color='r',alpha=0.5)
#         if labels is None:
#             plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
#         else:
#             plt.text(coeff[i,pca1]* 1.15, coeff[i,pca2] * 1.15, labels[i], color='g', ha='center', va='center')
#     plt.xlim(-1,1)
#     plt.ylim(-1,1)
#     plt.xlabel("PC{}".format(pcax))
#     plt.ylabel("PC{}".format(pcay))
#     plt.grid()


# Calcular los autovalores y autovectores
eigenvalues, eigenvectors = np.linalg.eig(cov)

# Crear una matriz diagonal con los autovalores
D = pd.DataFrame(np.diag(eigenvalues))

# Calcular la raíz cuadrada de cada valor diagonal
# D_1_2 = pd.DataFrame(np.where(
#     D != 0,
#     np.power(D, -(1/2)),
#     0  # Reemplaza los ceros con 0 en el resultado
# ))

# Z_new = pd.DataFrame(np.dot(componentespca.iloc[:,1:], D_1_2))

# D_1_2 = D_1_2[np.isinf(D_1_2)] = 0
# D_1_2 = pd.DataFrame(D_1_2)

# pd.DataFrame(diagonal_matrix)
# Calcular la media de cada columna
column_means = np.mean(ndf, axis=0)
column_means_2 = np.mean(ndf_scaled, axis=0)

# Restar la media a cada columna
A_centered = ndf - column_means
AA = np.dot(np.transpose(A_centered), A_centered)

A_centered_2 = ndf_scaled - column_means_2

AA_2 = np.dot(np.transpose(A_centered_2), A_centered_2)
eigenvalues_AA, eigenvectors_AA = np.linalg.eig(AA)
eigenvalues_AA_2, eigenvectors_AA_2 = np.linalg.eig(AA_2)


AA_eva = pd.DataFrame(np.diag(eigenvalues_AA))
AA_eva_2 = pd.DataFrame(np.diag(eigenvalues_AA_2))

# Calcular la raíz cuadrada de cada valor diagonal
AA_menos_12 = pd.DataFrame(np.where(
    AA_eva != 0,
    np.power(AA_eva, -(1/2)),
    0  # Reemplaza los ceros con 0 en el resultado
))

AA_menos_12_2 = pd.DataFrame(np.where(
    AA_eva_2 != 0,
    np.power(AA_eva_2, -(1/2)),
    0  # Reemplaza los ceros con 0 en el resultado
))

D_menos_12 = pd.DataFrame(np.where(
    D != 0,
    np.power(D, -(1/2)),
    0  # Reemplaza los ceros con 0 en el resultado
))

AA_14 = pd.DataFrame(np.where(
    AA_eva != 0,
    np.power(AA_eva, (1/4)),
    0  # Reemplaza los ceros con 0 en el resultado
))

AA_14_2 = pd.DataFrame(np.where(
    AA_eva_2 != 0,
    np.power(AA_eva_2, (1/4)),
    0  # Reemplaza los ceros con 0 en el resultado
))

AA_new = pd.DataFrame(np.dot(componentespca.iloc[:,1:], AA_menos_12))
AA_new_2 = pd.DataFrame(np.dot(componentespca.iloc[:,1:], AA_menos_12_2))
AA_neww = pd.DataFrame(np.dot(AA_new, AA_14))
AA_neww_2 = pd.DataFrame(np.dot(AA_new_2, AA_14_2))

AA_12 = pd.DataFrame(np.where(
    AA_eva != 0,
    np.power(AA_eva, (1/2)),
    0  # Reemplaza los ceros con 0 en el resultado
))

# AA_new_2 = (np.dot(AA_12, np.transpose(coeficientespca.iloc[:,1:])))
# AA_new_2_2 = np.dot(AA_14, coeficientespca.iloc[:,1:])
AA_2_coef = np.dot(AA_14_2, coeficientespca.iloc[:,1:])

#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Crear datos de ejemplo
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(x / 10)

# Crear una figura y un eje principal
fig, ax1 = plt.subplots(figsize=(10, 6))

# Graficar la primera serie de datos en el eje principal
ax1.plot(x, y1, 'b-', label='Serie 1: Sin(x)')
ax1.set_xlabel('X')
ax1.set_ylabel('Serie 1', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Crear un eje secundario en el lado derecho
ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-', label='Serie 2: Exp(x/10)')
ax2.set_ylabel('Serie 2', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Añadir leyenda y título
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Gráfico con dos series de datos en diferentes ejes')

plt.show()

