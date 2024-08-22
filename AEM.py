#%%
import pandas as pd
import numpy as np

path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECMT04\TP AEM - database.xlsx"
df = pd.read_excel(path)
ndf = df.iloc[:,1:].set_index('Country Code')

#Para todos los datos
desc = ndf.describe()
cv = pd.DataFrame(ndf.std() / ndf.mean()).T
cv.index = ['cv']
kurt = pd.DataFrame(ndf.kurt()).T
kurt.index = ['kurt']
skew = pd.DataFrame(ndf.skew()).T
skew.index = ['skew']
desc = pd.concat([desc, cv, kurt, skew], axis=0).reset_index()

#Para todos los datos sin Argentina
ndf_1 = ndf.drop('ARG')
desc_1 = ndf_1.describe()
cv_1 = pd.DataFrame(ndf_1.std() / ndf_1.mean()).T
cv_1.index = ['cv']
kurt_1 = pd.DataFrame(ndf_1.kurt()).T
kurt_1.index = ['kurt']
skew_1 = pd.DataFrame(ndf_1.skew()).T
skew_1.index = ['skew']
desc_1 = pd.concat([desc_1, cv_1, kurt_1, skew_1], axis=0).reset_index()

#Para Argentina
ndf_2 = ndf.loc[['ARG']].reset_index()

"""Ver si ARG no se encuentra muy fuera del promedio de la OCDE"""

#Matriz de covarianzas
mcov = ndf.cov()

#Matriz de correlaciones
mcor = ndf.corr()

#Medidas de variazión conjunta y medidas de correlación conjunta
varianza_total = np.trace(mcov)
varianza_media = varianza_total / len(ndf.columns)
varianza_generalizada = np.linalg.det(mcov)
varianza_efectiva = varianza_generalizada**(1/len(ndf.columns))

dependencia_conjunta = np.linalg.det(mcor)
dependencia_efectiva = dependencia_conjunta**(1/(len(ndf.columns)-1))

medidas_globales = pd.DataFrame([[varianza_total, varianza_media, varianza_generalizada, varianza_efectiva, dependencia_conjunta, dependencia_efectiva]],
                                columns=['varianza_total', 'varianza_media', 'varianza_generalizada', 'varianza_efectiva', 'dependencia_conjunta', 'dependencia_efectiva'])

# output_path = r'C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECMT04\OUTPUT_TP v2.xlsx'
# with pd.ExcelWriter(output_path) as writer:
#     desc.to_excel(writer, sheet_name='describe', index=False)
#     desc_1.to_excel(writer, sheet_name='describe 1', index=False)
#     ndf_2.to_excel(writer, sheet_name='describe 2', index=False)
#     mcov.to_excel(writer, sheet_name='covarianzas', index=False)
#     mcor.to_excel(writer, sheet_name='correlación', index=False)
#     medidas_globales.to_excel(writer, sheet_name='medidas_globales', index=False)

"""

Análisis de Componentes Principales - Matriz de Correlaciones
"""

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

variables_pca = pd.DataFrame(data_var, index=[f'Componente {i+1}' for i in range(len(varianza))]).reset_index()
coeficientes_pca = pd.DataFrame(coeficientes, columns=[f'Coeficiente (eigenvector) {i+1}' for i in range(coeficientes.shape[1])], index=ndf.columns).reset_index()
componentes_pca = pd.DataFrame(componentes, columns=[f'Componente {i+1} ' for i in range(coeficientes.shape[1])], index=df.iloc[:,0]).reset_index()

#Elección de Componentes Principales
#1. Método del Codo - Varianza
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(varianza) + 1), varianza, marker='o', linestyle='--')
# plt.xlabel('Número de componente')
# plt.ylabel('Varianza')
# plt.title('Método del Codo - Varianza')
# for i, v in enumerate(varianza):
#     plt.text(i + 1, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
# plt.show()

# #2. Método del Codo - Varianza explicada

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(varianza_explicada) + 1), varianza_explicada, marker='d', linestyle='--', color='black')
# plt.xlabel('Número de componente')
# plt.ylabel('Varianza explicada')
# plt.title('Método del Codo - Varianza explicada')
# for i, v in enumerate(varianza_explicada):
#     plt.text(i + 1, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
# plt.show()

# #3. Método del Codo - Varianza explicada acumulada
# # plt.figure(figsize=(10, 6))

# plt.plot(range(1, len(varianza_explicada_acum) + 1), varianza_explicada_acum, marker='v', linestyle='--', color='green')
# plt.xlabel('Número de componente')
# plt.ylabel('Varianza explicada acumulada')
# plt.title('Método del Codo - Varianza explicada acumulada')
# for i, v in enumerate(varianza_explicada_acum):
#     plt.text(i + 1, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
# plt.show()

"""
Análisis de Componentes Principales - Matriz de Covarianzas

"""

# pca_2 = PCA()
# pca_2.fit(ndf)

# varianza_2 = pca_2.explained_variance_
# std_componentes_2 = np.sqrt(pca_2.explained_variance_)
# varianza_explicada_2 = pca_2.explained_variance_ratio_
# varianza_explicada_acum_2 = np.cumsum(pca_2.explained_variance_ratio_)
# coeficientes_2 =  pca.components_.T

# #Generamos una tabla resumen
# data_var_2 = {
#     'Varianza': varianza_2,
#     'Desviación Estándar': std_componentes_2,
#     'Varianza Explicada': varianza_explicada_2,
#     'Varianza Explicada Acumulada': varianza_explicada_acum_2
# }

# variables_PCA_2 = pd.DataFrame(data_var_2, index=[f'Componente {i+1}' for i in range(len(varianza_2))]).reset_index()


# coeficientespca_2 = pd.DataFrame(coeficientes_2, columns=[f'Componente {i+1}' for i in range(coeficientes_2.shape[1])], index=ndf.columns).reset_index()

#Elección de Componentes Principales
#1. Método del Codo - Varianza

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(varianza) + 1), varianza, marker='o', linestyle='--')
# plt.xlabel('Número de componente')
# plt.ylabel('Varianza')
# plt.title('Método del Codo - Varianza')
# for i, v in enumerate(varianza):
#     plt.text(i + 1, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
# plt.show()

# #2. Método del Codo - Varianza explicada

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(varianza_explicada) + 1), varianza_explicada, marker='d', linestyle='--', color='black')
# plt.xlabel('Número de componente')
# plt.ylabel('Varianza explicada')
# plt.title('Método del Codo - Varianza explicada')
# for i, v in enumerate(varianza_explicada):
#     plt.text(i + 1, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
# plt.show()

# #3. Método del Codo - Varianza explicada acumulada
# # plt.figure(figsize=(10, 6))

# plt.plot(range(1, len(varianza_explicada_acum) + 1), varianza_explicada_acum, marker='v', linestyle='--', color='green')
# plt.xlabel('Número de componente')
# plt.ylabel('Varianza explicada acumulada')
# plt.title('Método del Codo - Varianza explicada acumulada')
# for i, v in enumerate(varianza_explicada_acum):
#     plt.text(i + 1, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
# plt.show()


#Método del Codo - Varianza explicada comparativa Corr vs. Cov
# plt.plot(range(1, len(varianza_explicada) + 1), varianza_explicada, marker='d', linestyle='--', color='black', label='con Matriz de Correlaciones')
# for i, v in enumerate(varianza_explicada):
#     plt.text(i + 1, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

# # Segunda variable
# plt.plot(range(1, len(varianza_explicada_2) + 1), varianza_explicada_2, marker='o', linestyle='-', color='blue', label='con Matriz de Covarianzas')
# for i, v in enumerate(varianza_explicada_2):
#     plt.text(i + 1, v - 0.03, f"{v:.2f}", ha='center', va='top')

# # Etiquetas y título
# plt.xlabel('Número de componente')
# plt.ylabel('Varianza explicada')
# plt.title('Método del Codo')

# # Leyenda
# plt.legend()

# plt.show()

