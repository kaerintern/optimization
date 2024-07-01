## Optimization
Exploring methods and various features to improve overall eficiency of System Efficiency, starting by looking at Chiller Effficiency on its own.

### First Iteration (No longer used)
Uses [lift, ct_tot_kw, cooling_load] as its features.

### Second Iteration (No longer used)
uses [lift, ct_tot_kw, cooling_load, ch_1_kwe, ch_2_kwe] as its features.

### Third Iteration (Completed)
Uses [lift, ct_tot_kw, cooling_load, chiller_configuration, cwst, ct_approach] as its features
accompanied by feature manipulation in both features and target variables.

Model: RF_h_cwst_ct_approach.sav

### Fourth Iteration (In Progress)
Uses [lift, ct_tot_kw, cooling_load, chiller_configuration, cwst, ct_approach] as its features
accompanied by feature manipulation in both features and target variables.

Further broken down into various weather categories:
- hot: DB>30C
- general: 28C<DB<30C
- rainy: DB<28C & RH>85%
- cool: DB<28C

##3 Streamlit UI 
UI is currently developed using streamlit package.

To run:
```
    cd optimization/streamlit_ui
    then run
    % streamlit run ui.py
```