import pandas as pd
import io
import requests
import boto3
from sqlalchemy import create_engine

def fetch_from_url(url):
    r = requests.get(url)
    return pd.read_csv(io.BytesIO(r.content))

def fetch_from_db(uri, query):
    engine = create_engine(uri)
    return pd.read_sql(query, engine)

def fetch_from_s3(bucket, key, ak, sk, region):
    s3 = boto3.client('s3', aws_access_key_id=ak, aws_secret_access_key=sk, region_name=region)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))