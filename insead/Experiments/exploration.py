#%%
from load_dotenv import load_dotenv
import os
import pandas as pd
import seaborn as sns

load_dotenv()
#%%
df = pd.read_csv(os.environ['mac_insead_path'])
print('Unfiltered row of dataframe is {}'.format(len(df)))
#%%
df.columns
#%%
df.ch1kw.describe()
#%%
# filtering
df = df[(df['effsys']>0.45) & (df['effsys']<0.65)]
df = df[(df['ct1kw']> 2) & (df['ct1kw']<14)]
df = df[(df['ct2akw']> 2) & (df['ct2akw']<14)]
df = df[(df['loadsys']>200) & (df['loadsys']<450)]
df = df[(df['cwrhdr']>31) & (df['cwrhdr']<33.5)]
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# average (3)chiller power input
df['chavgkw'] = (df['ch1kw']+df['ch2kw']+df['ch3kw'])/3
df = df[(df['chavgkw']>20)& (df['chavgkw']<80)]

# cooling tower total
df['ct_tot_kw'] = df['ct1kw'] + df['ct2akw']

# lift: [h_cwrt-h_chwst]
df['lift'] = df['cwrhdr'] - df['chwshdr']
df = df[(df['lift']>21) & (df['lift']<24)]

# weekend
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['weekend'] = df['dayofweek'].apply(lambda x:1 if x<=4 else 2)

# time column
def time_dec(x):
    if x<=12:
        t = 1
    elif x<=17:
        t = 2
    else:
        t = 3
    
    return t

df['time'] = pd.to_datetime(df['timestamp']).dt.hour
df['time'] = df['time'].apply(time_dec)

print('Filtered row of dataframe is {}'.format(len(df)))

# %%
sns.displot(df, x='effsys')
sns.displot(df, x='ct_tot_kw')
sns.displot(df, x='chavgkw')
sns.displot(df, x='loadsys')
sns.displot(df, x='lift')
sns.displot(df, x='weekend')
sns.displot(df, x='time')
sns.displot(df, x='cwrhdr')
# %%
