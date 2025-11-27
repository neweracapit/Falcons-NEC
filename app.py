import streamlit as st
from streamlit_main import *

st.set_page_config(
    page_title="NewEraCap ML-Enabled",
    page_icon="../Falcons-NEC/misc/favicon_box.ico"    # Local file
)



# Set Background Image
set_bg("../Falcons-NEC/misc/new_era_cap_cover.jpeg",opacity=0.9)

# Define 2 tabs Purchase and Sales

# Sidebar tabs
tab = st.sidebar.selectbox(
    "Choose Dashboard",
    ["Sales", "Purchase"],   # <-- Your tab names
    index=0
)

# Tab content
if tab == "Sales":
    st.header("New Era Cap - Falcons - Sales Dashboard")
    st.write("Forecasting content here...")

elif tab == "Purchase":
    st.header("New Era Cap - Falcons - Purchase Plan Dashboard")
    st.write("Forecasting content here...")