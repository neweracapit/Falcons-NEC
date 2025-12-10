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
    container_purchase_historical = blob_service_client.get_container_client('purchase-historical')
    container_sales_historical = blob_service_client.get_container_client('sales-historical')
    return container_sales, container_purchase, container_purchase_historical, container_sales_historical

def load_data():    

    # Get azure clients
    container_sales, container_purchase, _, _ = authenticate_azure()

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
        predictions['MONTH_START'] = pd.to_datetime(predictions['MONTH_START'])
    elif type == 'Purchase':
        predictions = pd.read_csv(purchase_path)
        predictions['month'] = pd.to_datetime(predictions['PO_CREATED_DATE'])
        #predictions = predictions.drop(columns=['PO_CREATED_DATE'])
    
    return predictions

def download_historical_data() -> pd.DataFrame:
    """Load Sales historical data"""
    # Setup for Historical Download
    local_path_historical = './historical_data/'
    os.makedirs(local_path_historical, exist_ok=True)

    _, _, container_purchase_historical, container_sales_historical = authenticate_azure()

    # Download and return Sales Historical
    # List blobs in Purchase Historical
    blob_list = list(container_sales_historical.list_blobs())
    latest_sales_historical_blob = max(blob_list, key=lambda b: b.last_modified)
    print("Latest Sales Historical file:", latest_sales_historical_blob.name)

    # Define Downloaded file name
    local_file_name_sales = f'sales_historical.csv'
    download_path_sales_historical = local_path_historical + local_file_name_sales

    # Download the latest purchase historical blob
    with open(file=download_path_sales_historical, mode="wb") as download_file:
        download_file.write(container_sales_historical.download_blob(latest_sales_historical_blob.name).readall())

    # Download and return Purchase Historical
    # List blobs in Purchase Historical
    blob_list = list(container_purchase_historical.list_blobs())
    latest_purchase_historical_blob = max(blob_list, key=lambda b: b.last_modified)
    print("Latest Purchase Historical file:", latest_purchase_historical_blob.name)

    # Define Downloaded file name
    local_file_name = f'purchase_historical.csv'
    download_path_purchase_historical = local_path_historical + local_file_name

    # Download the latest purchase historical blob
    with open(file=download_path_purchase_historical, mode="wb") as download_file:
        download_file.write(container_purchase_historical.download_blob(latest_purchase_historical_blob.name).readall())
    
    return download_path_purchase_historical, download_path_sales_historical

def load_historical_data(type):
    """Load Sales historical data"""
    historical_purchase_path, historical_sales_path = download_historical_data()

    if type == 'Sales':
        historical_data = pd.read_csv(historical_sales_path)
        pass
    elif type == 'Purchase':
        historical_data = pd.read_csv(historical_purchase_path)
        #historical_data['month'] = pd.to_datetime(historical_data['PO_CREATED_DATE'])
        #historical_data = historical_data.drop(columns=['PO_CREATED_DATE'])
    
    return historical_data

# --------------- OpenAI ---------------
from openai import OpenAI
import base64
from PIL import Image

def get_openai_client():
    OPEN_AI_API_KEY = st.secrets['OPENAI_API_KEY']
    client = OpenAI()

    return client

def generate_image_description(image_url: str):
    client = get_openai_client()

    vision_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this forecast graph in detail."},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]
    )

    forecast_description = vision_response.choices[0].message.content
    return forecast_description 

def generate_insights_w_image(historical_json, image_url):
    client = get_openai_client()
    
    forecast_description = generate_image_description(image_url)
    analysis_prompt = f"""
    You are a supply-chain forecasting analyst.

    Here is the forecast description generated from the image:
    {forecast_description}

    Here is 5 years of historical data (JSON):
    {historical_json}

    TASK:
    1. Compare the forecasted trend to historical trends.
    2. Check if seasonality in the forecast matches history.
    3. Identify spikes or dips that break historical patterns.
    4. Detect overprediction or underprediction.
    5. Identify high-risk months or years.
    6. Provide 5–8 actionable business recommendations.

    Format your answer as:

    ### Trend Comparison
    ...

    ### Seasonality Comparison
    ...

    ### Anomalies
    ...

    ### Risks
    ...

    ### Recommendations
    ...

    Stop after recommendations. 
    """

    # Use GPT-5-nano for cheap reasoning
    analysis_response = client.responses.create(
        model="gpt-5-nano",
        input=analysis_prompt
    )

    # Extract text
    analysis_text = analysis_response.output_text

    # Extract Concise
    # Use GPT-5-nano for cheap reasoning
    analysis_response = client.responses.create(
        model="gpt-5-nano",
        input=f"You are a supply-chain forecasting analyst. Create concise version of the following analysis, maintain the same sections:\n{analysis_text}. Stop after recommendations."
        )

    # Extract text
    analysis_text_concise = analysis_response.output_text

    print("\n--- FINAL FORECAST vs HISTORY ANALYSIS ---\n")
    
    return analysis_text, analysis_text_concise

def generate_insights_wo_image(historical_json, forecast_json):
    client = get_openai_client()
    
    analysis_prompt = f"""
    You are a supply-chain forecasting analyst.

    Here is future forecasts from 2025 to 2028:
    {forecast_json}

    Here is 5 years of historical data (JSON):
    {historical_json}

    TASK:
    1. Compare the forecasted trend to historical trends.
    2. Identify spikes or dips that break historical patterns.
    3. Provide 5–8 actionable business recommendations.

    Format your answer as:

    ### Trend Comparison
    ...

    ### Anomalies
    ...

    ### Recommendations
    ...

    Stop after recommendations and do not ask anything else. 
    """

    # Use GPT-5-nano for cheap reasoning
    analysis_response = client.responses.create(
        model="gpt-5-nano",
        input=analysis_prompt
    )

    # Extract text
    analysis_text = analysis_response.output_text
    analysis_text = beautify_output(analysis_text)

    return analysis_text

def generate_concise_insights(analysis_text):
        # Extract Concise
    client = get_openai_client()
    # Use GPT-5-nano for cheap reasoning
    analysis_response = client.responses.create(
        model="gpt-5-nano",
        input=f"You are a supply-chain forecasting analyst. Create concise version of the following analysis, maintain the same sections:\n{analysis_text}. Stop after recommendations."
        )

        # Extract text
    analysis_text_concise = analysis_response.output_text
    analysis_text_concise = beautify_output(analysis_text_concise)

def beautify_output(llm_text):
    import re
    sections = re.split(r"### ", llm_text)
    parsed = {}

    for s in sections:
        if not s.strip():
            continue
        title, content = s.split("\n", 1)
        parsed[title.strip()] = content.strip()

    return parsed




