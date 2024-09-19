#%%
import pandas as pd
import pyreadstat as st
import pingouin as pg

path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECMT04\problem_set_2\ine.dta"

df, meta = st.read_dta(path)
df.columns = [f"v{i}" for i in range(1, len(df.columns)+1)]
mardia_test = pg.multivariate_normality(df, alpha=0.05)