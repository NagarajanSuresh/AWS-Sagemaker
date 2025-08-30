FROM python:3.9-slim

# Install build tools and other dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /opt/ml/code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# required for SageMaker
RUN pip install sagemaker-training 
COPY train.py .
ENV SAGEMAKER_PROGRAM=train.py