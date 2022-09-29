from tkinter.ttk import Separator
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime

df = pd.read_csv('DCOILBRENTEUv2.csv')
df.plot(x='DATE', y='DCOILBRENTEU')

## tb dá sem loop. FASTER!!!
df['VarDD'] = df.DCOILBRENTEU.diff()

def previousVariation():
    df_previous = df

    for index, row in df.iterrows():
        if(index > 0):
            variation = df.loc[index,'DCOILBRENTEU'] - df.loc[index-1, 'DCOILBRENTEU']
            df.loc[index, 'Variation'] = variation
        else:
            df.loc[index, 'Variation'] = 0

    hist = df.hist(column='Variation', bins=20)


previousVariation()

print(df[:])

plt.show()

