#%%
import pandas as pd
import numpy as np

path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\DiTella\MEC\Materias\2024 2T\Análisis Estadístico Multivariado [MT04]\Exámen final\OECD + Arg - 2019.xlsx"
df = pd.read_excel(path)

cols_to_drop = df.columns[df.isin(['..']).any()]
df_n = df.drop(columns=cols_to_drop)

path_2 = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\DiTella\MEC\Materias\2024 2T\Análisis Estadístico Multivariado [MT04]\Exámen final\OECD + Arg - 2019 v2.xlsx"
df_n.to_excel(path_2)

#%%
import pandas as pd
import pyreadstat as st

path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\DiTella\MEC\Materias\2024 2T\[MT04] Análisis Estadístico Multivariado\Guías prácticas\Problem Set 2-20240714\ine.dta"

df, meta = st.read_dta(path)
meta
#%%
import pandas as pd
import pyreadstat as st

path = r'C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECMT04\Hogar_t403_0.dta'

df, meta = st.read_dta(path)

agl = pd.DataFrame(list(meta.value_labels['aglomerado'].items()), columns=['id', 'location'])

agl_2 = pd.merge(df, agl, left_on='aglomerado', right_on='id')
agl_2.groupby('location').size().sort_values(ascending=False)

df[df['aglomerado'] == 8]['itf'].mean()
#%%
import pandas as pd

path = r'C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECMT04\TP AEM - FD DG.xlsx'
df = pd.read_excel(path)

df.iloc[:, 4:5].cov()
#%%
import pandas as pd
import pyreadstat as st
import numpy as np

path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\DiTella\MEC\Materias\2024 2T\[MT04] Análisis Estadístico Multivariado\Guías prácticas\Problem Set 1-20240714\eurosec.dta"

df, meta = st.read_dta(path)

# std = np.std(df.select_dtypes(include=[np.number]), axis=0, ddof=1)
# mean = np.mean(df.select_dtypes(include=[np.number]), axis=0)

# cv = std / mean

# metrics = pd.concat([std, mean, cv], axis=1)
# metrics.columns = ['desvío_estándar', 'promedio', 'coeficiente_de_variación']

# cov = df.iloc[:,1:].cov()
# pd.DataFrame(cov, index=df.select_dtypes(include=[np.number]).columns, 
#                                    columns=df.select_dtypes(include=[np.number]).columns)

# df.iloc[:,1:].cov()

# cov = df.iloc[:,1:].cov()
# np.linalg.det(cov)

# corr = df.iloc[:,1:].corr()
# eigenvalues, eigenvectors = np.linalg.eig(corr)
# eigenvalues
#%%
import pandas as pd
import pyreadstat as st
import numpy as np

path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\DiTella\MEC\Materias\2024 2T\[MT04] Análisis Estadístico Multivariado\Guías prácticas\Problem Set 1-20240714\records.dta"

df, meta = st.read_dta(path)
desc = df.describe()

dfn = df.iloc[:,1:]
mean = dfn.mean()
std = dfn.std()
cv = std / mean

desc_2 = pd.concat([mean, std, cv], axis=1)
desc_2.columns = ['mean', 'standard_deviation', 'cv']

cov = dfn.cov()
corr = dfn.corr()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
dfn_scaled = scaler.fit_transform(dfn)

pca = PCA()
a = pca.fit(dfn_scaled)

a = pca.explained_variance_
proportion_of_variance = a / np.sum(a)

pca.components_.T * np.sqrt(a)