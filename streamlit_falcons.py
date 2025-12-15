import streamlit as st
import base64
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
import time

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
    container_openai = blob_service_client.get_container_client('openai-data')
    return container_sales, container_purchase, container_purchase_historical, container_sales_historical, container_openai

def load_data():    

    # Get azure clients
    container_sales, container_purchase, _, _, _ = authenticate_azure()

    local_path = './predicted_data_falcons/'
    os.makedirs(local_path, exist_ok=True)

    # Download and return Sales
    local_file_name = f'latest_falcons_predictions_sales.csv'
    download_path_sales = local_path + local_file_name

    with open(file=download_path_sales, mode="wb") as download_file:
        download_file.write(container_sales.download_blob('sales_forecast_2025_2028_20251207_192758.csv').readall())
            

    # Download and return Purchase
    blob_list = list(container_purchase.list_blobs())
    latest_purchase_blob = max(blob_list, key=lambda b: b.last_modified)
    print("Latest file:", latest_purchase_blob.name)

    local_file_name = f'latest_falcons_predictions_purchase.csv'
    download_path_purchase = local_path + local_file_name

    with open(file=download_path_purchase, mode="wb") as download_file:
        download_file.write(container_purchase.download_blob('forecast_global_2025_to_2028_v4_20251207_192758.csv').readall())
        
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

    _, _, container_purchase_historical, container_sales_historical, _ = authenticate_azure()

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

def load_openai_data():
    _, _, _, _, container_openai = authenticate_azure()

    local_path = './predicted_data/'
    os.makedirs(local_path, exist_ok=True)

    # Download and return Sales
    blob_list = list(container_openai.list_blobs())
    latest_openai_blob = max(blob_list, key=lambda b: b.last_modified)
    print("Latest Sales file:", latest_openai_blob.name)

    local_file_name = f'latest_openai_dataset.csv'
    download_path_openai = local_path + local_file_name

    with open(file=download_path_openai, mode="wb") as download_file:
        download_file.write(container_openai.download_blob(latest_openai_blob.name).readall())

    openai_df = pd.read_csv(download_path_openai)
    openai_df['PO_CREATED_DATE'] = pd.to_datetime(openai_df['PO_CREATED_DATE'], errors='coerce')
    openai_df['YEAR'] = openai_df['PO_CREATED_DATE'].dt.year
    openai_df['QUARTER'] = openai_df['PO_CREATED_DATE'].dt.quarter
    openai_df['MONTH'] = openai_df['PO_CREATED_DATE'].dt.month

    return openai_df


def get_openai_client():
    OPEN_AI_API_KEY = st.secrets['OPENAI_API_KEY']
    client = OpenAI()

    return client


import pandas as pd
import numpy as np
import ast

def compute_group_overview(filtered_df):

    pos_driver_list = []
    pos_value_list = []
    neg_driver_list = []
    neg_value_list = []
    magnitude_list = []

    for shap in filtered_df["shap_summary"]:

        # 🔥 FIX 1: Convert whole shap object from str → dict
        if isinstance(shap, str):
            try:
                shap = ast.literal_eval(shap)
            except:
                continue  # skip bad rows

        # 🔥 FIX 2: Extract and convert their subfields safely
        top_pos = shap.get("top_positive", {})
        top_neg = shap.get("top_negative", {})

        if isinstance(top_pos, str):
            try:
                top_pos = ast.literal_eval(top_pos)
            except:
                top_pos = {}

        if isinstance(top_neg, str):
            try:
                top_neg = ast.literal_eval(top_neg)
            except:
                top_neg = {}

        # --- Extract top positive drivers ---
        for feat, val in top_pos.items():
            pos_driver_list.append(feat)
            pos_value_list.append(val)

        # --- Extract top negative drivers ---
        for feat, val in top_neg.items():
            neg_driver_list.append(feat)
            neg_value_list.append(val)

        # Magnitude
        magnitude_list.append(shap.get("shap_magnitude", 0))

    # Handle empty lists gracefully
    if not pos_driver_list or not neg_driver_list:
        return {
            "dominant_positive_driver": None,
            "avg_positive_influence": 0,
            "dominant_negative_driver": None,
            "avg_negative_influence": 0,
            "avg_shap_magnitude": float(np.mean(magnitude_list)) if magnitude_list else 0
        }

    # --- Summary Computation ---
    dominant_pos = pd.Series(pos_driver_list).value_counts().idxmax()
    dominant_neg = pd.Series(neg_driver_list).value_counts().idxmax()

    avg_pos_value = float(np.mean([v for f, v in zip(pos_driver_list, pos_value_list) if f == dominant_pos]))
    avg_neg_value = float(np.mean([v for f, v in zip(neg_driver_list, neg_value_list) if f == dominant_neg]))

    avg_magnitude = float(np.mean(magnitude_list))

    return {
        "dominant_positive_driver": dominant_pos,
        "avg_positive_influence": avg_pos_value,
        "dominant_negative_driver": dominant_neg,
        "avg_negative_influence": avg_neg_value,
        "avg_shap_magnitude": avg_magnitude
    }

def build_historical_summary(df: pd.DataFrame,):
    target_col = "PURCHASE_COUNT"
    categorical_features = [
        "SPORT",
        "SEASON",
        "GENDER",
        "SILHOUETTE"
    ]

    """
    Builds an LLM-friendly driver summary from historical data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Historical dataset
    categorical_features : list
        List of categorical column names (e.g. ['SPORT','SEASON','GENDER'])
    target_col : str
        Target variable (default = PURCHASE_COUNT)

    Returns
    -------
    dict
        Compact driver summary for LLM consumption
    """

    summary_grp = {}
    summary_main = {}
    total_demand = df[target_col].sum()

    
    for feature in categorical_features:
        
        grp = (
            df.groupby(feature)[target_col]
            .sum()
            .reset_index()
        )
        grp = grp[grp[feature] != 'Others']
        
        feature_values = grp[feature].unique()

        for grp_val in feature_values:
            grp_df = grp[grp[feature] == grp_val]

            avg_val = grp_df[target_col].mean()
            std_val = grp_df[target_col].std()
            share_pct = (grp_df[target_col].sum() / total_demand) * 100

            # Relative strength (based on share of demand)
            if share_pct > 60:
                strength = "high"
            elif share_pct > 25:
                strength = "medium"
            else:
                strength = "low"
            
            # Variation (coefficient of variation)
            cv = std_val / avg_val if avg_val > 0 else 0
            if cv > 1:
                variation = "high"
            elif cv > 0.4:
                variation = "medium"
            else:
                variation = "low"

            
            summary_grp[grp_val] = {
                "avg_purchase": round(float(avg_val), 2),
                "share_of_demand_pct": round(float(share_pct), 1),
                "relative_strength": strength,
                "variation": variation
            }
        summary_main[feature] = summary_grp

    return summary_main


def build_insight_prompt(summary_dict,type, level="row"):
    if type == 'Forecast':
        return f"""
                You are an AI assistant helping to explain machine learning demand forecasting outputs to Non-Technical Supply Chain Team.
                The model is LightGBM, and SHAP values show the contribution of each feature to the forecast.

                ### CONTEXT
                - SHAP positive values increase the forecast.
                - SHAP negative values decrease the forecast.
                - SHAP magnitude indicates the overall influence and volatility of the prediction.

                ### TASK
                Provide a clear, simple, business-friendly explanation of the forecast drivers.
                And avoid stating technical term, instead what that term means.

                ### INPUT (SHAP SUMMARY)
                {summary_dict}

                ### INSTRUCTIONS
                1. Summarize the most important positive and negative drivers.
                2. Give a simple explanation of how these features relate to increased or decreased demand.
                3. For group-level summaries, explain the general behavior of the segment or year.
                4. Avoid technical jargon.
                5. Keep it concise but meaningful.

                ### OUTPUT
                An explanation for Non-Techincal Supply Chain and Business Stake Holders.
                """
    elif type == 'Historical':
        return f"""
        You are a supply-chain demand forecasting analyst.

        You are given a compact historical driver summary derived from purchase data.
        Each driver includes:
        - avg_purchase: average demand contribution
        - share_of_demand_pct: share of total demand
        - relative_strength: high / medium / low
        - variation: high / medium / low

        HISTORICAL DRIVER SUMMARY:
        {summary_dict}

        TASK:
        Generate a clear, concise, business-friendly explanation of demand drivers.

        INSTRUCTIONS:
        1. Identify the strongest and weakest demand drivers.
        2. Explain how each key driver influences demand in plain language.
        3. Highlight drivers with high variation as potential planning risks.
        4. Avoid technical or statistical jargon.
        5. Do NOT repeat the raw numbers unless necessary.
        6. Focus on interpretation and business impact, not data processing.
        7. Keep the response to 5–8 sentences.

        OUTPUT FORMAT:
        Write a short narrative paragraph suitable for supply-chain planners and business stakeholders.
        """


def generate_llm_review(summary_dict, type, level="row", model="gpt-4.1-mini"):

    client = get_openai_client()
    prompt = build_insight_prompt(summary_dict, type, level)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at communicating with non-tech Supply Chain and Business Stake holder."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    # Updated: use .content instead of ["content"]
    return response.choices[0].message.content


def typewriter(text, speed=0.01):
    placeholder = st.empty()
    typed = ""
    for char in text:
        typed += char
        placeholder.markdown(typed)
        time.sleep(speed)




        