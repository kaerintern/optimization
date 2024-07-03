import hmac
import os
import pickle
import warnings
from bokeh.plotting import figure
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)

def check_password():

    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

model_path = 'RF_general.sav'

weather = st.multiselect(
    "Select weather category",
    ['Summer', 'Rainy', 'General']
)
print(weather)
# weather selection
model_path = 'RF_summer.sav' if 'Summer' in weather else 'RF_general.sav'
model_path = 'RF_rain.sav' if 'Rain' in weather else 'RF_general.sav'

relative_path = os.path.join(os.path.dirname(__file__), '..', 'parklane', 'model', 'third_iteration', 'RF_h_cwst_ct_approach.sav')

with open(relative_path, 'rb') as file:
    print(relative_path)
    model = pickle.load(file)

graph_color = ["red", "red", "blue", "red", "blue", "green"]


cooling_load = st.number_input("Cooling Load", min_value=300, max_value=450)
lift = st.number_input("Lift", min_value=18.0, max_value=30.0, step=0.1)
h_cwst = st.number_input("CWST", min_value=28.0, max_value=32.0, step=0.1)
ct_approach = st.number_input("CT Approach", min_value=0.1, max_value=5.0, step=0.1)



p1 = figure(
    title='Chiller System Efficiency vs Cooling Tower Power Input On 1 Chiller Configuration',
    x_axis_label='Cooling Tower Power Input',
    y_axis_label='Chiller System Efficiency',
)
p1.plot_height=400
p1.plot_width=1000

p2 = figure(
    title='Chiller System Efficiency vs Cooling Tower Power Input On 2 Chiller Configuration',
    x_axis_label='Cooling Tower Power Input',
    y_axis_label='Chiller System Efficiency'
)
p2.plot_height=400
p2.plot_width=1000

def plot(p, ct_tot_kw, sysef, lift, color, line_width=2):
    p.line(ct_tot_kw, sysef, legend_label=str(lift),color=color, line_width=line_width)


def reset():
    st.session_state['ch_sysef_1_ch'] = {}
    st.session_state['ch_sysef_2_ch'] = {}
    st.session_state['counter_1'] = 0
    st.session_state['counter_2'] = 0
    st.session_state['lifts'] = []

if 'ch_sysef_1_ch' not in st.session_state:
    st.session_state['ch_sysef_1_ch'] = {}

if 'ch_sysef_2_ch' not in st.session_state:
    st.session_state['ch_sysef_2_ch'] = {}

if 'counter_1' not in st.session_state:
    st.session_state['counter_1'] = 0

if 'counter_2' not in st.session_state:
    st.session_state['counter_2'] = 0

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
                
                # 1 chiller
                temp_1 = []
                ch_run = 0
                for i in range(40, 201):
                    ct_tot_kw.append(i/10)
                    # cooling tower manipulation
                    sysef = model.predict([[
                        lift ** 3,
                        cooling_load,
                        (i/10) ** 1.5,
                        ch_run, 
                        h_cwst ** 2,
                        ct_approach ** 1.5
                    ]])
                    # sysef manipulation
                    sysef = np.log10(sysef) / 3
                    temp_1.append(np.round(sysef, 3))
            
                st.session_state.ch_sysef_1_ch[np.round(lift, 1)] = temp_1

                # 2 chiller
                temp_2 = []
                ch_run = 1
                for i in range(40, 201):
                    sysef = model.predict([[
                        lift ** 3,
                        cooling_load,
                        (i/10) ** 1.5,
                        ch_run, 
                        h_cwst ** 2,
                        ct_approach ** 1.5
                    ]])
                    # sysef manipulation
                    sysef = np.log10(sysef) / 3
                    temp_2.append(np.round(sysef, 3))

                st.session_state.ch_sysef_2_ch[np.round(lift, 1)] = temp_2

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
    for sysef in st.session_state.ch_sysef_1_ch.values():
        try:
            plot(p=p1, ct_tot_kw=ct_tot_kw, sysef=sysef, color=graph_color[st.session_state.counter_1], lift=lift)
            st.session_state.counter_1+=1
        except IndexError:
            st.write('Error')
            reset()

    for sysef in st.session_state.ch_sysef_2_ch.values():
        try:
            plot(p=p2, ct_tot_kw=ct_tot_kw, sysef=sysef, color=graph_color[st.session_state.counter_2], lift=lift)
            st.session_state.counter_2+=1
        except IndexError:
            st.write('Error')
            reset()

st.bokeh_chart(p1, use_container_width=True)
st.bokeh_chart(p2, use_container_width=True)

# Comment After
color_comment = ['red', 'blue', 'green']
comment = ''
for i in range(len(st.session_state.lifts)):
    comment += "The {} colored plot has a lift of {} ".format(color_comment[i], st.session_state.lifts[i]) + "  \n"
st.write(comment)
