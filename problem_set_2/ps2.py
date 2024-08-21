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


coeficientespca = pd.DataFrame(coeficientes, columns=[f'Componente (eigenvector) {i+1} ' for i in range(coeficientes.shape[1])], index=ndf.columns).reset_index()