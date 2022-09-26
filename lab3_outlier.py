from tkinter.ttk import Separator
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime

fields=['Time (UTC)', 'Open', 'High', 'Low', 'Close']

df = pd.read_csv('EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv', sep=';',decimal=',', usecols=fields)


df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'])


stock_data = df.set_index('Time (UTC)')


average = df['Close'].mean()
std = df['Close'].std()

print("Mean - ", average)
print("Standard deviation - ", std)


k = 0.4
ls = [] #Array of outliers


df.plot(x='Time (UTC)', y = 'Close', title='w/ outliers')


def removeLine(ls):
    df_remove = df.drop(labels = ls, axis = 0)

    df_remove.plot(x='Time (UTC)', y = 'Close', title='Outliers removed')


def previousValue(ls):
    df_previous = df
    
    for index in ls:
        df_previous.loc[index, 'Close'] = df_previous.loc[index-1, 'Close']

    df_previous.plot(x='Time (UTC)', y = 'Close', title='Based on previous value')

def interpolation(ls):
    df_interpol = df

    for index in ls:
        df_interpol.loc[index, 'Close'] = (df_interpol.loc[index-1, 'Close'] + df_interpol.loc[index+1, 'Close'])/2

    df_interpol.plot(x='Time (UTC)', y = 'Close', title='Interpolation')



for index, row in df.iterrows():
    if(row['Close'] > average + k*std or row['Close'] < average - k*std):
        print("Outlier: Index -> ", index)
        print("Value -> ", row['Close'])
        ls.append(index)
        
        
removeLine(ls)
previousValue(ls)
interpolation(ls)

plt.show()



