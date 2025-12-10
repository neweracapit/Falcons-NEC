# ui.py
import streamlit as st
from PIL import Image
import io

def render_header(title: str, logo_path: str = None):
    cols = st.columns([0.1, 0.9])
    with cols[0]:
        if logo_path:
            try:
                img = Image.open("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSldg6YauhDaNpGfy4siAW25AvgwKkMwnsZGA&s")
                st.image(img, width=100)
            except Exception:
                st.image(logo_path, width=100)

    with cols[1]:
        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                height: 100%;
                padding-left: 10px;
            ">
                <h1 style="margin: 0;">{title}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    # st.markdown("<hr>", unsafe_allow_html=True)

def sidebar_navigation(options):
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Choose dashboard", options)
    return choice

def kpi_row(kpis):
    """
    kpis: list of dicts: [{'label': 'Total Sales', 'value': 12345, 'delta': 4.5}, ...]
    """
    cols = st.columns(len(kpis))
    for c, k in zip(cols, kpis):
        with c:
            label = k.get('label', '')
            value = k.get('value', '')
            delta = k.get('delta', None)
            if delta is None:
                st.metric(label, value)
            else:
                st.metric(label, value, delta)

def download_button(df, filename="data.csv", label="Download CSV"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label, data=csv, file_name=filename, mime='text/csv')
