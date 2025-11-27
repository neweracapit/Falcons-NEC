import streamlit as st
import base64

import streamlit as st

def set_bg_url(url, opacity=0.3):

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            position: relative;
        }}

        /* Overlay */
        .stApp::before {{
            content: "";
            position: absolute;
            inset: 0;
            background: rgba(0,0,0,{opacity});
            z-index: 0;
        }}

        .stApp > * {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
