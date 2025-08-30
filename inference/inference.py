import os
import joblib
import pandas as pd
import boto3
from sagemaker.s3 import S3Downloader


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)

    s3_uri = os.environ["SCALER_S3"]
    print(s3_uri)
    local_dir = "/tmp"
    S3Downloader.download(s3_uri, local_dir)
    scaler_local_path = os.path.join(local_dir, "scaler.pkl")
    scaler = joblib.load(scaler_local_path)

    return {"model": model, "scaler": scaler}

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = pd.DataFrame(request_body["instances"])
        return data
    elif request_content_type == "text/csv":
        return pd.read_csv(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_and_scaler):

    model = model_and_scaler["model"]
    scaler = model_and_scaler["scaler"]

    # Binary features
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    input_data[binary_cols] = input_data[binary_cols].apply(lambda col: col.str.lower())
    input_data[binary_cols] = input_data[binary_cols].replace({'yes': 1, 'no': 0}).astype(int)

    input_data = pd.get_dummies(input_data, columns=['furnishingstatus'])
    input_data = input_data.astype({col: int for col in input_data.select_dtypes('bool').columns})
    input_data["area"] =scaler.transform(input_data["area"])

    return model.predict(input_data)

def output_fn(prediction, content_type):
    if content_type == "application/json":
        return {"predictions": prediction.tolist()}
    elif content_type == "text/csv":
        return ",".join(map(str, prediction))
    else:
        raise ValueError(f"Unsupported response type: {content_type}")