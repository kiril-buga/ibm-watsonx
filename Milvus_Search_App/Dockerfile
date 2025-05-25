FROM python:3.11-slim

RUN pip install --no-cache-dir flask pymilvus torch transformers

WORKDIR /app
COPY . /app


CMD ["python", "app.py"]
