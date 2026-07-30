FROM python:3.11-slim

RUN pip install --no-cache-dir transcribe-cpp==0.1.3

WORKDIR /bench
ENTRYPOINT ["python", "/bench/run_model.py"]
