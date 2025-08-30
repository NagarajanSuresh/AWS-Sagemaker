# server.py
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

import inference  
import os

app = Flask(__name__)
MODEL_PATH = "/opt/ml/model"
model = inference.model_fn(MODEL_PATH)

app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

@app.route("/ping", methods=["GET"])
def ping():
    """
    Healthcheck function.
    """
    return "pong"

@app.route("/invocations", methods=["POST"])
def predict():
    content_type = request.headers.get("Content-Type")
    data = inference.input_fn(request.data, content_type)
    preds = inference.predict_fn(data, model)
    return jsonify(inference.output_fn(preds, "application/json"))
   