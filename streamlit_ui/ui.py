import pickle

from bokeh.plotting import figure
import numpy as np
import streamlit as st


color = ["red", "red", "blue", "red", "blue", "green"]
model = pickle.load(open('/Users/admin/Desktop/optimization/parklane/RF_first.sav', 'rb'))

cooling_load = st.slider("Cooling Load", min_value=300, max_value=450)
lift = st.slider("Lift", min_value=22.0, max_value=26.0, step=0.1)
create = st.button("Create")
reset_button = st.button("Reset")

p = figure(
    title='line',
    x_axis_label='ct_tot_kw',
    y_axis_label='ch_sysef',
)
p.plot_height=400
p.plot_width=1000

def plot(p, ct_tot_kw, sysef, lift, color, line_width=2):
    p.line(ct_tot_kw, sysef, legend_label=str(lift),color=color, line_width=line_width)

def reset():
    st.session_state['ch_sysef_per_lift'] = {}
    st.session_state['counter'] = 0
    st.session_state['lifts'] = []

if 'ch_sysef_per_lift' not in st.session_state:
    st.session_state['ch_sysef_per_lift'] = {}

if 'counter' not in st.session_state:
    st.session_state['counter'] = 0

if 'lifts' not in st.session_state:
    st.session_state['lifts'] = []
    
ct_tot_kw = []
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    if create and st.session_state.counter <9:
        try:
            # check if the input lift has already been calculated
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
    elif create and st.session_state.counter >=10:
        st.write('Error')
        reset()
with c6:
    if reset_button and cooling_load:
        reset()

if create:
    for sysef in st.session_state.ch_sysef_per_lift.values():
        try:
            plot(p=p, ct_tot_kw=ct_tot_kw, sysef=sysef, color=color[st.session_state.counter], lift=lift)
            st.session_state.counter+=1
        except IndexError:
            st.write('Error')
            reset()

st.bokeh_chart(p, use_container_width=True)

# Comment After
color_comment = ['red', 'blue', 'green']
comment = ''
for i in range(len(st.session_state.lifts)):
    comment += "The {} colored plot has a lift of {} ".format(color_comment[i], st.session_state.lifts[i]) + "  \n"
st.write(comment)