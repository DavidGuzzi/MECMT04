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
std_devs = np.std(componentespca.iloc[:,1:], axis=0, ddof=1)

# Crear la matriz diagonal con las desviaciones estándar
# D = np.diag(std_devs)

# # Calcular la matriz diagonal elevada a la -1/2
# D_inv_sqrt = np.diag(1 / std_devs)

# Multiplicar la matriz A por la matriz D_inv_sqrt para normalizar
A_normalized = componentespca.iloc[:,1:] / (std_devs ^ (-1/2))
A_normalized = pd.DataFrame(A_normalized)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 7))
sns.scatterplot(x=A_normalized.iloc[:,1], y=A_normalized.iloc[:,2], data=A_normalized, s=100, color='blue')

for i in range(coeficientes.shape[0]):
    plt.arrow(0, 0, coeficientes[i, 0]*max(componentespca.iloc[:,1]), coeficientes[i, 1]*max(componentespca.iloc[:,2]),
              color='red', head_width=0.1, head_length=0.1)
    plt.text(coeficientes[i, 0]*max(componentespca.iloc[:,1]*1.2), coeficientes[i, 1]*max(componentespca.iloc[:,2])*1.2,
             ndf.columns[i], color='green', ha='center', va='center', fontsize=12)

plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Biplot de las dos primeras componentes principales')
plt.grid(True)
plt.show()

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
