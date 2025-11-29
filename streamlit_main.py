import streamlit as st
import base64
from dotenv import load_dotenv
load_dotenv()
import pandas as pd

import os
from azure.storage.blob import BlobServiceClient


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

# Authenticate and Get data from Azure Glob
def authenticate_azure():
    try:
        connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    except:
        connect_str = st.secrets["AZURE_STORAGE_CONNECTION_STRING"]

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_purchase = blob_service_client.get_container_client('predictions-purchase')
    container_sales = blob_service_client.get_container_client('predictions-sales')
    return container_sales, container_purchase

def load_data():    

    # Get azure clients
    container_sales, container_purchase = authenticate_azure()

    local_path = './predicted_data/'
    os.makedirs(local_path, exist_ok=True)

    # Download and return Sales
    blob_list = list(container_sales.list_blobs())
    latest_sales_blob = max(blob_list, key=lambda b: b.last_modified)
    print("Latest Sales file:", latest_sales_blob.name)

    local_file_name = f'latest_predictions_sales.csv'
    download_path_sales = local_path + local_file_name

    with open(file=download_path_sales, mode="wb") as download_file:
        download_file.write(container_sales.download_blob(latest_sales_blob.name).readall())
            

    # Download and return Purchase
    blob_list = list(container_purchase.list_blobs())
    latest_purchase_blob = max(blob_list, key=lambda b: b.last_modified)
    print("Latest file:", latest_purchase_blob.name)

    local_file_name = f'latest_predictions_purchase.csv'
    download_path_purchase = local_path + local_file_name

    with open(file=download_path_purchase, mode="wb") as download_file:
        download_file.write(container_purchase.download_blob(latest_purchase_blob.name).readall())
        
    return download_path_sales, download_path_purchase

# Load Data
@st.cache_data
def load_predictions(type) -> pd.DataFrame:
    """Load Salespredictions and metrics"""

    sales_path, purchase_path = load_data()

    if type == 'Sales':
        predictions = pd.read_csv(sales_path)
        predictions['month'] = pd.to_datetime(predictions['month'])
    elif type == 'Purchase':
        predictions = pd.read_csv(purchase_path)
        predictions['month'] = pd.to_datetime(predictions['month'])
    
    return predictions
