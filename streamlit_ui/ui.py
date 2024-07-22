from bokeh.plotting import figure
from dotenv import load_dotenv, dotenv_values
import hmac
import numpy as np
import os
import pandas as pd
import pickle
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

@st.cache_resource
def load_model():
    relative_path = os.path.join(os.path.dirname(__file__), '..', 'parklane', 'model', os.environ['model_name'])

    with open(relative_path, 'rb') as file:
        model = pickle.load(file)
    
    return model

graph_color = ["red", "red", "blue", "red", "blue", "green"]

# Password Logic
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

# Functions Logic
def plot(p, ct_tot_kw, sysef, lift, color, line_width=2):
    '''
    Plot graph
    
    Args:
        p = plot
        ct_tot_kw = total cooling tower
        sysef = system efficiency
        lift = lift
        color = colot
    Returns:
        plotted line'''
    p.line(ct_tot_kw, sysef, legend_label=str(lift),color=color, line_width=line_width)


def reset():
    '''
    Reset system, making stored variables empty
    '''
    st.session_state['ch_sysef_1_ch'] = {}
    st.session_state['ch_sysef_2_ch'] = {}
    st.session_state['counter_1'] = 0
    st.session_state['counter_2'] = 0
    st.session_state['lifts'] = []

# UI Logic
col1, col2 = st.columns(2)

with col1:
    cooling_load = st.number_input("Cooling Load", min_value=250, max_value=450)
    lift = st.number_input("Lift", min_value=18.0, max_value=30.0, step=0.1)
    h_cwrt = st.number_input("CWRT", min_value=25.0, max_value=35.0, step=0.1)
    day = st.selectbox('weekday', ('weekday', 'weekend'))
    create = st.button("Create")

with col2:
    h_cwst = st.number_input("CWST", min_value=26.0, max_value=32.0, step=0.1)
    wb = st.number_input("WB", min_value=24.0, max_value=28.0, step=0.1)
    time = st.selectbox('Time of Day', ('morning', 'afternoon', 'evening'))
    ct_approach = st.number_input("CT Approach", min_value=0.1, max_value=4.0, step=0.1)
    reset_button = st.button("Reset")

p1 = figure(
    title='System Efficiency vs Cooling Tower Power Input On 1 Chiller Configuration',
    x_axis_label='Cooling Tower Power Input',
    y_axis_label='System Efficiency',
)
p1.plot_height=400
p1.plot_width=1000

p2 = figure(
    title=' System Efficiency vs Cooling Tower Power Input On 2 Chiller Configuration',
    x_axis_label='Cooling Tower Power Input',
    y_axis_label='System Efficiency'
)
p2.plot_height=400
p2.plot_width=1000



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

if reset_button:
    reset()


ct_tot_kw = []

# Calculation Logic
if create and st.session_state.counter_1 <9:
    try:
        model = load_model()
        # check if the input lift has already been calculated
        if lift not in st.session_state.lifts:
            time = eval(os.environ['time_dict'].replace('"', ''))[time]
            weekday = eval(os.environ['weekday_dict'].replace('"', ''))[day]
            # variable manipulation
            vars = pd.DataFrame({
                'ch_run': 0, # dummy
                'h_cwst': h_cwst ** 0.9,
                'h_cwrt': h_cwrt ** 0.9,
                'ct_tot_kw': 1, # dummy will be change again
                'lift': lift ** 3,
                'sys_cl': cooling_load,
                'wea_ct_wb': wb ** 0.7,
                'ct_approach': ct_approach,
                'time': time,
                'weekend': weekday
            }, index=[0])

            # 1 chiller
            temp_1 = []
            for ct in range(40, 201):
                ct_tot_kw.append(ct/10)
                # ch run configuration
                vars['ch_run'] = 0
                # cooling tower manipulation
                vars['ct_tot_kw'] = (ct/10)**1.5
                # sysef manipulation
                sysef = np.log10(model.predict(vars))/3
                temp_1.append(np.round(sysef, 3))
        
            st.session_state.ch_sysef_1_ch[np.round(lift, 1)] = temp_1

            # 2 chiller
            temp_2 = []
            for ct in range(40, 201):
                # ch run configuration
                vars['ch_run'] = 1
                # cooling tower manipulation
                vars['ct_tot_kw'] = (ct/10)**1.5
                # sysef manipulation
                sysef = np.log10(model.predict(vars)) / 3
                temp_2.append(np.round(sysef, 3))

            st.session_state.ch_sysef_2_ch[np.round(lift, 1)] = temp_2

            st.session_state.lifts.append(np.round(lift,1))
    except IndexError:
        st.write('System Error')
        st.session_state.counter_1 = 0

elif create and st.session_state.counter_1 >=10:
    st.write('Error')
    reset()

# Plotting Logic
if create:
    for sysef in st.session_state.ch_sysef_1_ch.values():
        try:
            plot(p=p1, ct_tot_kw=ct_tot_kw, sysef=sysef, color=graph_color[st.session_state.counter_1], lift=lift)
            st.session_state.counter_1+=1
        except:
            pass

    for sysef in st.session_state.ch_sysef_2_ch.values():
        try:
            plot(p=p2, ct_tot_kw=ct_tot_kw, sysef=sysef, color=graph_color[st.session_state.counter_2], lift=lift)
            st.session_state.counter_2+=1
        except:
            pass

st.bokeh_chart(p1, use_container_width=True)
st.bokeh_chart(p2, use_container_width=True)

# Comment After
color_comment = ['red', 'blue', 'green']
comment = ''
for i in range(len(st.session_state.lifts)):
    comment += "The {} colored plot has a lift of {} ".format(color_comment[i], st.session_state.lifts[i]) + "  \n"
st.write(comment)
