## Optimization
Exploring methods and various features to improve overall eficiency of System Efficiency, starting by looking at Chiller Effficiency on its own.

## Setup

1. Do a pip install
- pip install -r requirements.txt 

2. Train/Load models
- Train: run each of the site's 'model_training' script
- load: if you have a model already then, you only need to adjust .env model path

note: place the model in directory 'site/model/model_file/'

3. Adjust config files
- Password: make a new file in folder '.streamlit' titled 'secrets.toml'
Note: this is according to streamlit's password documentation

- Adjust any other necessary paths in .env for model, dataset, or other relevant paths

4. Run Streamlit
- Each of the site has its own streamlit script then each of the site's script is entered to 'menu_ui.py' script so run *menu_ui.py* instead of the other.

To run:

```
    cd optimization/streamlit_ui
    streamlit run menu_ui.py
```

### First Iteration (No longer used)
Uses [lift, ct_tot_kw, cooling_load] as its features.

### Second Iteration (No longer used)
uses [lift, ct_tot_kw, cooling_load, ch_1_kwe, ch_2_kwe] as its features.

### Third Iteration (No longer used)
Uses [lift, ct_tot_kw, cooling_load, chiller_configuration, cwst, ct_approach] as its features
accompanied by feature manipulation in both features and target variables.

### Fourth Iteration (Canceled)
Uses [lift, ct_tot_kw, cooling_load, chiller_configuration, cwst, ct_approach] as its features
accompanied by feature manipulation in both features and target variables.

Further broken down into various weather categories:
- hot: DB>30C
- general: 28C<DB<30C
- rainy: DB<28C & RH>85%
- cool: DB<28C

### Fifth Iteration (In use)
Uses [lift, ct_tot_kw, cooling_load, chiller_configuration, cwst, ct_approach, cwrt, time, weekday] as its features
accompanied by feature manipulation in both features and target variables.
