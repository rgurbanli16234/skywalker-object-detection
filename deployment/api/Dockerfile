FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and ONNX
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY deployment/api/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source
COPY deployment/api /app/api

# Copy release models
# Assumes context is repository root
COPY release /app/release

ENV MODEL_PATH=/app/release/onnx/best.onnx
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
