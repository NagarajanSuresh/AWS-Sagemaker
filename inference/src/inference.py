import os
import joblib
import pandas as pd

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = pd.DataFrame(request_body["instances"])
        return data
    elif request_content_type == "text/csv":
        return pd.read_csv(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    if content_type == "application/json":
        return {"predictions": prediction.tolist()}
    elif content_type == "text/csv":
        return ",".join(map(str, prediction))
    else:
        raise ValueError(f"Unsupported response type: {content_type}")