# libs/data_loader.py
from io import BytesIO
from urllib.parse import urlparse, unquote
from azure.storage.blob import BlobClient
from azure.identity import DefaultAzureCredential
import pandas as pd
import streamlit as st

def looks_like_sas(url: str) -> bool:
    return any(k in url for k in ("sig=", "sv=", "se=", "sp="))

def parse_blob_url_no_sas(blob_url: str):
    parsed = urlparse(blob_url)
    path = parsed.path.lstrip("/")
    parts = path.split("/")
    if len(parts) < 2:
        raise ValueError(f"Unexpected blob URL format: {blob_url}")
    container = parts[0]
    blob_name = "/".join(parts[1:])
    account = parsed.netloc.split(".")[0]
    account_url = f"https://{account}.blob.core.windows.net"
    return account_url, container, unquote(blob_name)

@st.cache_data(ttl=300)
def load_csv(blob_url: str):
    """
    Load CSV from a blob URL. Accepts SAS URLs or storage URLs (uses DefaultAzureCredential).
    Returns pandas.DataFrame.
    """
    if not blob_url:
        raise ValueError("No blob_url provided to load_csv.")
    if looks_like_sas(blob_url):
        blob = BlobClient.from_blob_url(blob_url)
        raw = blob.download_blob().readall()
    else:
        account_url, container, blob_name = parse_blob_url_no_sas(blob_url)
        credential = DefaultAzureCredential()
        blob = BlobClient(account_url=account_url, container_name=container, blob_name=blob_name, credential=credential)
        raw = blob.download_blob().readall()
    df = pd.read_csv(BytesIO(raw))
    return df
