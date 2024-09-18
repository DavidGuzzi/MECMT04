#%%
import pandas as pd
import pyreadstat as st

path = r"C:\Users\HP\OneDrive\Escritorio\David Guzzi\Github\MECMT04\problem_set_3\firmas.dta"

df, meta = st.read_dta(path)
md = df.loc[:,['ebitass', 'rotc']]