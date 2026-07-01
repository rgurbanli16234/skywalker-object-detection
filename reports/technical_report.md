# Technical Report: YOLOv8 Object Detection Architecture

## 1. System Architecture
The Skywalker Detection pipeline utilizes **Ultralytics YOLOv8 Medium (YOLOv8m)**, a state-of-the-art single-stage object detector balancing high spatial accuracy with sub-30ms latency bounds. 

### Core Components:
- **Backbone**: Modified CSPDarknet53 with advanced spatial pyramid pooling (SPPF).
- **Neck**: Path Aggregation Network (PANet) for multi-scale feature fusion.
- **Head**: Anchor-free decoupled head for independent classification and bounding box regression.

## 2. Training Infrastructure
- **Framework**: PyTorch 2.x
- **Hardware**: NVIDIA CUDA Acceleration with Automatic Mixed Precision (AMP - FP16).
- **Optimizer**: AdamW / SGD (configurable via `train_config.yaml`).
- **Learning Rate Scheduler**: Cosine Annealing (lr0 to lrf).
- **Augmentation Pipeline**: 
  - Mosaic augmentation (combines 4 images for contextual variety).
  - HSV color space shifting (Hue, Saturation, Value).
  - Random affine transformations (translation, scaling).

## 3. Deployment Artifacts
The training pipeline emits the following deployable representations:
- **best.pt**: Raw PyTorch checkpoint retaining full gradient graphs and optimizer states.
- **best.onnx**: Open Neural Network Exchange format for cross-platform CPU/GPU inference via ONNXRuntime.
- **best.engine**: NVIDIA TensorRT serialized engine for maximum throughput on specific CUDA compute capabilities.

## 4. Software Engineering Standards
- **Modularization**: Segregated source code (`src/train.py`, `src/evaluate.py`, `src/infer.py`) following Single Responsibility Principle.
- **Typed Signatures**: Comprehensive PEP-484 type hints throughout the Python codebase.
- **Error Handling**: Graceful exception propagation and centralized logging (`utils.py`).
