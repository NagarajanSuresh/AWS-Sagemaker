import boto3
import json

# Initialize the SageMaker runtime client
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

# Your payload wrapped in "instances"
payload = {
    "area": 1200,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 4,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "no",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "parking": 2,
    "prefarea": "yes",
    "furnishingstatus": "furnished"
}


# Call the endpoint
response = runtime.invoke_endpoint(
    EndpointName="ns2312-housing-lr-endpoint",   # Replace with your endpoint name
    ContentType="application/json",
    Body=json.dumps(payload)
)

# Read and decode the prediction result
result = response["Body"].read().decode("utf-8")
print("Prediction:", result)