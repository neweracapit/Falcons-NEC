import streamlit as st
from streamlit_main import *

try:
    st.set_page_config(
        page_title="NewEraCap ML-Enabled",
        page_icon="https://raw.github.com/neweracapit/Falcons-NEC/blob/main/misc/favicon_box.ico"    # Local file
    )
    
except:
    print("Error Loading Background image")

url = 'https://raw.githubusercontent.com/neweracapit/Falcons-NEC/main/misc/new_era_cap_cover.jpeg'
set_bg_url(url=url,opacity=0.9)



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