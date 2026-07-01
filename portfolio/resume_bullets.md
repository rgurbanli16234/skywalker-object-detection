# Resume Bullet Points

**Machine Learning Engineer - Skywalker Object Detection Project**
* Architected an end-to-end computer vision pipeline utilizing **YOLOv8** and **PyTorch**, fine-tuning a custom 44,000-image dataset to achieve high-precision bounding box regression and classification.
* Optimized model inference latency by **57%** (from 28ms to 12ms) by serializing raw PyTorch graphs into **ONNX** and **NVIDIA TensorRT (FP16)** formats for production edge deployment.
* Developed a high-throughput, asynchronous inference API using **FastAPI**, serving predictions with real-time hardware metrics tracking via **psutil**.
* Designed a scalable microservices architecture utilizing **Docker**, orchestrating the API and an **Nginx** reverse proxy via **Docker Compose** and **Kubernetes** manifests (Deployments, Services, Ingress).
* Established rigorous MLOps practices, including comprehensive evaluation benchmarking scripts, deterministic environment replication (`requirements.txt`), and automated CI/CD pipelines via **GitHub Actions**.
* Deployed an interactive user-facing web application utilizing **Gradio** on **Hugging Face Spaces**, demonstrating real-time threshold adjustments and visual bounding box overlays.
