import pickle

from bokeh.plotting import figure
import numpy as np
import pandas as pd
import streamlit as st

cooling_load = st.slider("Cooling Load", min_value=300, max_value=450)
lift = st.slider("Lift", min_value=22.0, max_value=26.0, step=0.1)

model = pickle.load(open('/Users/admin/Desktop/optimization/parklane/RF_first.sav', 'rb'))

p = figure(
    title='line',
    x_axis_label='ct_tot_kw',
    y_axis_label='ch_sysef',
)
p.plot_height=400
p.plot_width=1000

ct_tot_kw = []

if 'ch_sysef_per_lift' not in st.session_state:
    st.session_state['ch_sysef_per_lift'] = {}

if 'lifts' not in st.session_state:
    st.session_state['lifts'] = []

if 'counter' not in st.session_state:
    st.session_state['counter'] = 0
    
color = ["red", "red", "blue", "red", "blue", "green"]
create = st.button("Create")
reset = st.button("Reset")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    if create and st.session_state.counter <9:
        try:
            if lift not in st.session_state.lifts:
                temp = []
                for i in range(40, 161):
                    ct_tot_kw.append(i/10)
                    temp.append(np.round(model.predict([[lift, cooling_load, (i/10)]]),3))
                st.session_state.ch_sysef_per_lift[np.round(lift,1)] = temp
                st.session_state.lifts.append(np.round(lift,1))
        except IndexError:
            st.write('System Error')
            st.session_state.counter = 0
with c6:
    if reset and cooling_load:
        st.session_state['ch_sysef_per_lift'] = {}
        st.session_state['lifts'] = []
        st.session_state['counter'] = 0

def plot(p, ct_tot_kw, sysef, lift, color, line_width=2):
    p.line(ct_tot_kw, sysef, legend_label=str(lift),color=color, line_width=line_width)

if create:
    for sysef in st.session_state.ch_sysef_per_lift.values():
        try:
            plot(p=p, ct_tot_kw=ct_tot_kw, sysef=sysef, color=color[st.session_state.counter], lift=lift)
            st.session_state.counter+=1
        except IndexError:
            st.session_state['ch_sysef_per_lift'] = {}
            st.session_state['lifts'] = []
            st.session_state['counter'] = -2

st.bokeh_chart(p, use_container_width=True)
# giving comment of the plot
color_comment = ['red', 'blue', 'green']

comment = ''
for i in range(len(st.session_state.lifts)):
    comment += "The {} colored plot has a lift of {} \n".format(color_comment[i], st.session_state.lifts[i])
st.write(comment)