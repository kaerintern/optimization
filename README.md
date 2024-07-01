## Optimization
Exploring methods and various features to improve overall eficiency of System Efficiency, starting by looking at Chiller Effficiency on its own.

### First Iteration
Uses [lift, ct_tot_kw, cooling_load] as its features.

### Second Iteration
uses [lift, ct_tot_kw, cooling_load, ch_1_kwe, ch_2_kwe] as its features.

### Third Iteration
Uses [lift, ct_tot_kw, cooling_load, chiller_configuration, cwst, ct_approach] as its features
accompanied by feature manipulation in both features and target variables.


## Streamlit UI
UI is currently developed using streamlit package.

To run:
```
    cd optimization/streamlit_ui
    then run
    % streamlit run ui.py
```