🚀 I'm excited to share my latest machine learning project: **Skywalker Object Detection**!

Over the past few weeks, I built an end-to-end, production-grade computer vision pipeline using Ultralytics YOLOv8. My goal wasn't just to train a model in a Jupyter Notebook, but to architect a system that adheres to strict enterprise deployment standards.

Key engineering highlights:
🔹 **Deep Learning**: Fine-tuned a YOLOv8m network on a custom 44k image dataset with FP16 Automatic Mixed Precision.
🔹 **Performance Optimization**: Serialized PyTorch weights to ONNX and TensorRT, reducing inference latency to ~12ms (80+ FPS throughput).
🔹 **Microservices Architecture**: Containerized the inference engine using FastAPI and Docker, load-balanced behind an Nginx reverse proxy.
🔹 **Kubernetes Ready**: Engineered K8s manifests (Deployments, Services, Ingress) for scalable cloud deployment.
🔹 **CI/CD**: Implemented GitHub Actions for automated linting, dry-run testing, and Docker image builds.

I also deployed an interactive frontend via Hugging Face Spaces using Gradio! You can test it out live.

Check out the full open-source repository and architectural reports here: [Link to GitHub]
Try the live demo here: [Link to Hugging Face Space]

A massive thank you to the open-source community for the incredible tools (PyTorch, ONNXRuntime, FastAPI) that made this possible. I'm actively looking for Machine Learning Engineering roles—if your team is tackling tough computer vision or MLOps challenges, let's connect!

#MachineLearning #ComputerVision #YOLOv8 #PyTorch #MLOps #FastAPI #Kubernetes #Docker #AI
