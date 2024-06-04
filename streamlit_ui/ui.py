import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from bokeh.plotting import figure
import numpy as np
import streamlit as st


graph_color = ["red", "red", "blue", "red", "blue", "green"]
model = pickle.load(open('/Users/admin/Desktop/optimization/parklane/RF_first.sav', 'rb'))
second_model = pickle.load(open('/Users/admin/Desktop/optimization/parklane/RF_first_both_ch.sav', 'rb'))

cooling_load = st.slider("Cooling Load", min_value=300, max_value=450)
lift = st.slider("Lift", min_value=22.0, max_value=26.0, step=0.1)

p1 = figure(
    title='Chiller System Efficiency vs Cooling Tower Power Input varied by Lift',
    x_axis_label='Cooling Tower Power Input',
    y_axis_label='Chiller System Efficiency',
)
p1.plot_height=400
p1.plot_width=1000

p2 = figure(
    title='Chiller System Efficiency vs Cooling Tower Power Input varied by Chiller Configuration',
    x_axis_label='Cooling Tower Power Input',
    y_axis_label='Chiller System Efficiency'
)
p2.plot_height=400
p2.plot_width=1000

def plot_1(p, ct_tot_kw, sysef, lift, color, line_width=2):
    p.line(ct_tot_kw, sysef, legend_label=str(lift),color=color, line_width=line_width)

def plot_2(p, ct_tot_kw, sysef, color, line_width=2):
    p.line(ct_tot_kw, sysef, color=color, line_width=line_width)

def reset():
    st.session_state['ch_sysef_per_lift'] = {}
    st.session_state['counter_1'] = 0
    st.session_state['lifts'] = []

if 'ch_sysef_per_ch_typ' not in st.session_state:
    st.session_state['ch_sysef_per_ch_typ'] = {}

if 'ch_sysef_per_lift' not in st.session_state:
    st.session_state['ch_sysef_per_lift'] = {}

if 'counter_1' not in st.session_state:
    st.session_state['counter_1'] = 0

if 'lifts' not in st.session_state:
    st.session_state['lifts'] = []
    
ct_tot_kw = []
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    create = st.button("Create")
    if create and st.session_state.counter_1 <9:
        try:
            # check if the input lift has already been calculated
            if lift not in st.session_state.lifts:
                temp = []

                for i in range(20, 301):
                    ct_tot_kw.append(i/10)
                    temp.append(np.round(model.predict([[lift, cooling_load, (i/10)]]),3))

                st.session_state.ch_sysef_per_lift[np.round(lift,1)] = temp
                st.session_state.lifts.append(np.round(lift,1))
        except IndexError:
            st.write('System Error')
            st.session_state.counter_1 = 0
    elif create and st.session_state.counter_1 >=10:
        st.write('Error')
        reset()
with c3:
    reset_button = st.button("Reset")
    if reset_button:
        reset()

if create:
    for sysef in st.session_state.ch_sysef_per_lift.values():
        try:
            plot_1(p=p1, ct_tot_kw=ct_tot_kw, sysef=sysef, color=graph_color[st.session_state.counter_1], lift=lift)
            st.session_state.counter_1+=1
        except IndexError:
            st.write('Error')
            reset()

st.bokeh_chart(p1, use_container_width=True)

# Comment After
color_comment = ['red', 'blue', 'green']
comment = ''
for i in range(len(st.session_state.lifts)):
    comment += "The {} colored plot has a lift of {} ".format(color_comment[i], st.session_state.lifts[i]) + "  \n"
st.write(comment)

# Second Graph
ct_tot_kw_2 = []

create_2 = st.button("Create", key='2')
if create_2:
    for ch_typ in [0, 1]:
        temp = []

        for i in range(20, 301):
            ct_tot_kw_2.append(i/10)
            temp.append(np.round(second_model.predict([[lift, cooling_load, (i/10), ch_typ]]),3))
        st.session_state.ch_sysef_per_ch_typ[ch_typ] = temp

    for ch_typ, sysef in st.session_state.ch_sysef_per_ch_typ.items():
         
         plot_2(p=p2, ct_tot_kw=ct_tot_kw_2, sysef=sysef, color=color_comment[ch_typ])
    
st.bokeh_chart(p2, use_container_width=True)

# Comment After
comment_2 = " Lift: {}  \n".format(lift)
for ch_typ in st.session_state.ch_sysef_per_ch_typ.keys():
    comment_2 += "The {} colored plot has {} chiller activated  \n".format(
        color_comment[ch_typ], ch_typ+1)
st.write(comment_2)