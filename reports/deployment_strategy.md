# Deployment Strategy

## 1. Operational Overview
The Skywalker Object Detection model will be deployed using a microservices architecture, ensuring high availability, horizontal scalability, and strict isolation between the inference engine and the client-facing gateways.

## 2. Component Architecture

### A. Inference Engine (FastAPI)
- **Framework**: FastAPI (Python) for asynchronous, high-throughput request handling.
- **Payload**: Accepts base64 encoded images or multipart form data.
- **Response**: JSON payloads containing bounding boxes, class IDs, and confidence scores.
- **Endpoints**: `/predict` (inference), `/health` (liveness probe), `/metrics` (Prometheus metrics).

### B. Containerization (Docker)
- The entire FastAPI application and YOLOv8 ONNX runtime are packaged inside a minimal `python:3.10-slim` Docker container.
- Ensures environment parity across dev, staging, and production.

### C. Reverse Proxy (Nginx)
- Acts as the API gateway.
- Handles SSL termination, rate limiting, and payload size restrictions (preventing malicious multi-megabyte image uploads).

### D. Orchestration (Kubernetes)
- **Deployment**: Manages replica sets of the FastAPI containers.
- **Service**: Internal load balancing (ClusterIP).
- **Ingress**: Exposes the Nginx gateway to external traffic.
- **Autoscaling (HPA)**: Scales pods horizontally based on CPU utilization or custom inference queue length metrics.

## 3. Hugging Face Spaces (Interactive Demo)
Alongside the enterprise Kubernetes deployment, a highly visual, interactive web application built with **Gradio** will be hosted on Hugging Face Spaces. This provides an immediate, zero-setup portfolio showcase allowing users to upload images, adjust NMS thresholds interactively, and download visual results.
