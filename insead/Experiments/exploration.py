#%%
from load_dotenv import load_dotenv
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import r_regression


load_dotenv()
#%%
df = pd.read_csv(os.environ['mac_insead_path'])
print('Unfiltered row of dataframe is {}'.format(len(df)))

#%%
# filtering
df = df[(df['effsys']>0.45) & (df['effsys']<0.65)]
df = df[(df['ct1kw']> 2) & (df['ct1kw']<14)]
df = df[(df['ct2akw']> 2) & (df['ct2akw']<14)]
df = df[(df['loadsys']>150) & (df['loadsys']<450)]
df = df[(df['cwrhdr']>29) & (df['cwrhdr']<33.5)]
df = df[(df['cwshdr']>27) & (df['cwshdr']<30)]
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

df = df[['effsys', 'ct_tot_kw', 'chavgkw', 'loadsys', 'lift', 'weekend', 'time', 'cwrhdr', 'cwshdr']]
df = df.drop_duplicates().reset_index()

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
sns.displot(df, x='cwshdr')
# %%
# pearsons correlations
sys_feats = ['effsys', 'ct_tot_kw', 'loadsys', 'lift', 'weekend', 'time', 'cwrhdr', 'cwshdr']
sys_df  = df[sys_feats]

X = sys_df.drop(columns=['effsys'])
y = sys_df['effsys']
corr_values = r_regression(X,y)

table_df = pd.DataFrame(
    data=corr_values.reshape(1, -1),
    columns=X.columns)
table_df
# %%
# spearmans correlation
res_corr = []
for feat in sys_feats:
    res = stats.spearmanr(sys_df[feat], y)
    res_corr.append(res.statistic)

res_corr = np.array(res_corr)
table_df = pd.DataFrame(
    data=res_corr.reshape(1, -1),
    columns=sys_feats
)
table_df
