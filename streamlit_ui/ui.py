import pickle

from bokeh.plotting import figure
import numpy as np
import pandas as pd
import streamlit as st

cooling_load = st.number_input("Cooling Load", min_value=300, max_value=450)
lift = st.slider("Lift", min_value=22.00, max_value=26.00, step=0.01)

model = pickle.load(open('/Users/admin/Desktop/optimization/parklane/random_forest_param.sav', 'rb'))


p = figure(
    title='line',
    x_axis_label='ct_tot_kw',
    y_axis_label='ch_sysef',
)
p.plot_height=400
p.plot_width=1000

ch_sysef = []
ct_tot_kw = []
lifts = []
if lift not in lifts:
    for i in range(70, 161):
        ct_tot_kw.append(i/10)
        ch_sysef.append(np.round(model.predict([[lift, cooling_load, (i/10)]]),3))

    p.line(ct_tot_kw, ch_sysef, legend_label=str(lift), line_width=2)

st.bokeh_chart(p, use_container_width=True)
