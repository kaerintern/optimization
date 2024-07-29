#%%
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

import warnings

load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)
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
df = df.drop_duplicates().dropna().reset_index()

print('Filtered row of dataframe is {}'.format(len(df)))

# columns manipulation
df['effsys'] = np.int32(os.environ['ch_sysef_const']) ** df['effsys']
df['cwshdr'] = df['cwshdr'] ** np.float32(os.environ['h_cwst_const'])
df['cwrhdr'] = df['cwrhdr'] ** np.float32(os.environ['h_cwrt_const'])
df['ct_tot_kw'] = df['ct_tot_kw'] ** np.float32(os.environ['ct_tot_const'])
df['lift'] = df['lift'] ** np.int32(os.environ['lift_const'])
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

#%%
X = df.drop(columns=['index', 'effsys'])
y = df[['effsys']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
y.transpose()

# Training
max_depth = [100, 250, 500]
min_samples_splits = [10, 25, 50]
no_of_trees = [100, 200, 500]
min_samples_leafs = [10, 25, 50]


parameter_grid = {
    'regressor__max_depth': max_depth,
    'regressor__min_samples_split': min_samples_splits,
    'regressor__min_samples_leaf': min_samples_leafs,
    'regressor__n_estimators': no_of_trees,
    'regressor__random_state': [10],

}

# pipeline
pipeline = Pipeline([
    ('regressor', RandomForestRegressor(
        warm_start=True,
        bootstrap=True,
        n_jobs=-1,
        max_features='sqrt'
    ))
])

grid_search = GridSearchCV(
    estimator= pipeline,
    param_grid=parameter_grid,
    cv=5
)

grid_search.fit(X_train, y_train)

## save model
filename = 'RF_insead_no_wb.pkl'
pickle.dump(grid_search, open(filename, 'wb'))

# %%
