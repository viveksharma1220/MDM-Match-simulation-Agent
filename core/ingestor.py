import pandas as pd
import io

def process_file(uploaded_file):
    """Detects extension and returns a DataFrame from an uploaded file object."""
    ext = uploaded_file.name.split('.')[-1].lower()
    
    if ext == 'csv':
        return pd.read_csv(uploaded_file)
    elif ext in ['xls', 'xlsx']:
        return pd.read_excel(uploaded_file)
    elif ext == 'json':
        return pd.read_json(uploaded_file)
    elif ext == 'parquet':
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Format .{ext} not supported by the Ingestion Agent.")