import streamlit as st
from streamlit_main import *

try:
    st.set_page_config(
        page_title="NewEraCap ML-Enabled",
        page_icon="../misc/favicon_box.ico"    # Local file
    )
    url = 'https://github.com/neweracapit/Falcons-NEC/blob/main/misc/new_era_cap_cover.jpeg'
except:
    print("Error Loading Background image")

try:
    set_bg(url=url,opacity=0.9)
except: 
    pass



# Set Background Image


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