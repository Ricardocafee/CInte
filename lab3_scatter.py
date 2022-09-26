from tkinter.ttk import Separator
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime

df = pd.read_csv('DCOILBRENTEUv2.csv')
df_2 = pd.read_csv('DCOILWTICOv2.csv')

df['DATE'] = pd.to_datetime(df['DATE'])
df_2['DATE'] = pd.to_datetime(df_2['DATE'])


ax = df.plot(x='DATE', y='DCOILBRENTEU')

df_2.plot(x='DATE', y='DCOILWTICO',ax= ax)

ax2 = df.plot.scatter(x='DATE', y='DCOILBRENTEU')
#df_2.plot.scatter(x='DATE', y='DCOILWTICO',ax= ax2)

plt.show()