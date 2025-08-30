FROM python:3.9-slim
WORKDIR /opt/ml/code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY preprocess.py .
ENTRYPOINT ["python3", "preprocess.py"]