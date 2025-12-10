# libs/colmap.py
def pick_column(df, *names):
    for name in names:
        for c in df.columns:
            if c.lower() == name.lower():
                return c
    return None

def build_map(df):
    """
    Returns a dict with logical keys mapped to actual df column names (or None).
    """
    return {
        "year": pick_column(df,"FORECAST_YEAR", "YEAR", "year"),
        "date": pick_column(df, "DATE", "date"),
        "region": pick_column(df, "REGION", "region"),
        "sales_org": pick_column(df, "SALESORG", "sales_org", "sales org", "country"),
        "silhouette": pick_column(df, "SILHOUETTE", "silhouette"),
        "fabric": pick_column(df, "FABRIC_TYPE", "fabric_type", "product_division"),
        "sport": pick_column(df, "SPORT", "sport"),
        "country": pick_column(df, "COUNTRY", "country"),
        "gender": pick_column(df, "GENDER", "gender"),
        "prediction": pick_column(df, "FORECAST_P50", "PREDICTION", "predicted", "pred", "sales_count"),
        "actual": pick_column(df, "FORECAST_P50", "ACTUAL", "actual", "sales", "y")
    }
