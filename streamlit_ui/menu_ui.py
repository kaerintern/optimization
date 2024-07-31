import streamlit as st

pg = st.navigation([st.Page("insead.py"), st.Page("parklane.py")])
pg.run()