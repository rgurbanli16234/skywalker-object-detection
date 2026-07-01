# Skywalker Object Detection 🛸

![GitHub License](https://img.shields.io/github/license/rasul-gurbanli/skywalker-object-detection)
![Build Status](https://img.shields.io/github/actions/workflow/status/rasul-gurbanli/skywalker-object-detection/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white)

An enterprise-grade, highly optimized Object Detection repository built around Ultralytics YOLOv8. This project transforms a raw PyTorch training loop into a production-ready system capable of robust serialization (ONNX/TensorRT) and high-throughput deployment (FastAPI/Kubernetes).

## 🚀 Key Features

* **Advanced Training Pipeline**: Mixed-precision, distributed-ready YOLOv8 training with customizable augmentation.
* **Production Serialization**: Automated export to ONNX, TorchScript, and FP16 TensorRT for edge/cloud deployment.
* **Microservices Architecture**: Containerized FastAPI inference gateway with Nginx reverse proxy.
* **Kubernetes Native**: Included manifests for Helm/Kustomize deployment with HPA scaling.
* **Interactive UI**: Gradio web application for Hugging Face Spaces deployment.
* **Rigorous Benchmarking**: Comprehensive latency, throughput, and warmup scripts to guarantee SLA compliance.

## 📊 Performance Comparison

| Model Format | Precision | Memory | Latency (Batch=1) | Throughput (FPS) |
|--------------|-----------|--------|-------------------|------------------|
| PyTorch (.pt)| FP32      | 50 MB  | ~28.5 ms          | ~35              |
| ONNX         | FP32      | 48 MB  | ~25.2 ms          | ~39              |
| TensorRT     | FP16      | 25 MB  | ~12.0 ms          | ~83              |

## 🏗 System Architecture

The pipeline consists of three core domains:
1. **Model Engineering (`src/`)**: Data ingestion, training, and evaluation.
2. **Artifact Generation (`release/`)**: ONNX/TensorRT serialization and metric reporting.
3. **Deployment (`deployment/`)**: FastAPI, Docker Compose, and Kubernetes descriptors.

## ⚡ API Usage

Run the local inference server:
```bash
docker-compose -f deployment/docker-compose.yml up --build -d
```

Test the API via cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@examples/sample1.jpg"
```

Response format:
```json
{
  "status": "success",
  "inference_time_ms": 25.1,
  "detections": [
    {
      "class_id": 0,
      "class_name": "lightsaber",
      "confidence": 0.95,
      "bbox": [150.2, 300.5, 200.1, 800.0]
    }
  ]
}
```

## 🤗 Hugging Face Integration
Explore the model interactively on Hugging Face Spaces!
1. Install dependencies: `pip install -r huggingface/requirements.txt`
2. Run locally: `python huggingface/app.py`

## 📚 Reports
For in-depth analysis of the model's performance, see the `reports/` directory.

## 🤝 Contributing
Please see `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` for guidelines.

## 📝 License
This project is licensed under the MIT License - see the `LICENSE` file for details.
