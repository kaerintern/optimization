from bokeh.plotting import figure
from dotenv import load_dotenv
import hmac
import numpy as np
import os
import pandas as pd
import pickle
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

st.title("Insead")

@st.cache_resource
def load_model():
    relative_path = os.path.join(os.path.dirname(__file__), '..', 'insead', 'model', os.environ['insead_model_path'])

    with open(relative_path, 'rb') as file:
        model = pickle.load(file)
    
    return model

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
    st.session_state['ch_sysef'] = {}
    st.session_state['counter'] = 0
    st.session_state['lifts'] = []

# UI Logic
col1, col2 = st.columns(2)

with col1:
    cooling_load = st.number_input("Cooling Load", min_value=250, max_value=600)
    lift = st.number_input("Lift", min_value=18.0, max_value=30.0, step=0.1)
    h_cwrt = st.number_input("CWRT", min_value=25.0, max_value=35.0, step=0.1)
    day = st.selectbox('weekday', ('weekday', 'weekend'))
    create = st.button("Create")

with col2:
    h_cwst = st.number_input("CWST", min_value=26.0, max_value=32.0, step=0.1)    
    time = st.selectbox('Time of Day', ('morning', 'afternoon', 'evening'))
    reset_button = st.button("Reset")

p1 = figure(
    title='System Efficiency vs Cooling Tower Power Input On 1 Chiller Configuration',
    x_axis_label='Cooling Tower Power Input',
    y_axis_label='System Efficiency',
)
p1.plot_height=400
p1.plot_width=1000


if 'ch_sysef' not in st.session_state:
    st.session_state['ch_sysef'] = {} 

if 'counter' not in st.session_state:
    st.session_state['counter'] = 0

if 'lifts' not in st.session_state:
    st.session_state['lifts'] = []

if reset_button:
    reset()


ct_tot_kw = []

# Calculation Logic
if create and st.session_state.counter <9:
    try:
        model = load_model()
        # check if the input lift has already been calculated
        if lift not in st.session_state.lifts:
            time = eval(os.environ['time_dict'].replace('"', ''))[time]
            weekday = eval(os.environ['weekday_dict'].replace('"', ''))[day]
            # variable manipulation
            vars = pd.DataFrame({
                'ct_tot_kw': 1, # dummy input, will be change below
                'loadsys': cooling_load,
                'lift': lift ** np.int32(os.environ['lift_const']),
                'weekend': weekday,
                'time': time,
                'cwrhdr': h_cwst ** np.float32(os.environ['h_cwst_const']),
                'cwshdr': h_cwrt ** np.float32(os.environ['h_cwrt_const'])
            }, index=[0])

            temp_1 = []
            for ct in range(40, 201):
                ct_tot_kw.append(ct/10)
                # cooling tower manipulation
                vars['ct_tot_kw'] = (ct/10) ** np.float32(os.environ['ct_tot_const'])
                # sysef manipulation
                sysef = np.log10(model.predict(vars))/3
                temp_1.append(np.round(sysef, 3))
        
            st.session_state.ch_sysef[np.round(lift, 1)] = temp_1

            st.session_state.lifts.append(np.round(lift,1))
    except IndexError:
        st.write('System Error')
        st.session_state.counter = 0

elif create and st.session_state.counter >=10:
    st.write('Error')
    reset()

# Plotting Logic
if create:
    for sysef in st.session_state.ch_sysef.values():
        try:
            plot(p=p1, ct_tot_kw=ct_tot_kw, sysef=sysef, color=eval(os.environ['graph_color'].replace('"', ''))[st.session_state.counter], lift=lift)
            st.session_state.counter+=1
        except:
            pass

st.bokeh_chart(p1, use_container_width=True)

# Comment After
color_comment = ['red', 'blue', 'green']
comment = ''
for i in range(len(st.session_state.lifts)):
    comment += "The {} colored plot has a lift of {} ".format(color_comment[i], st.session_state.lifts[i]) + "  \n"
st.write(comment)
