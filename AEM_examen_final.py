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
import numpy as np

path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\DiTella\MEC\Materias\2024 2T\[MT04] Análisis Estadístico Multivariado\Exámen final\TP AEM - FD DG.xlsx"

df = pd.read_excel(path)
df