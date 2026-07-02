
# Skywalker Object Detection Pipeline

## 🚀 Final Key Detection Metrics (Prominent Results)
### Core Object Detection Metrics
| Metric         | Value       |
|----------------|-------------|
| Precision      | 0.74093     |
| Recall         | 0.91048     |
| F1-score       | 0.817       |
| mAP50          | 0.73835     |
| mAP50-95       | 0.58531     |

### Inference Speed & Latency
| Metric               | Value        |
|----------------------|--------------|
| GPU Inference FPS    | 30.6         |
| Avg. Latency (ms)    | 32.683       |
| Wall Clock FPS       | 27.03        |

---

## Overview
This is a high-performance object detection pipeline for the custom "Skywalker" dataset. Built on YOLOv8, it is specifically optimized for low VRAM GPUs (4GB or less), making it accessible for a wide range of hardware configurations. The pipeline includes everything from training and evaluation to inference and model export.

## Features
- 🎯 Training with advanced hyperparameter tuning
- 📊 Comprehensive evaluation and benchmarking
- 🚀 Fast inference with export to ONNX and TorchScript
- 📈 Visualization of training curves and PR curves
- 📦 Production-ready Docker deployment

## Dataset
The Skywalker dataset is a single-class object detection dataset with:
- **~49,947 total images**
- **~44,340 labeled objects**
- Train/Val split: ~38k training / ~11k validation

## Model Used
### YOLOv8n (Nano)
The YOLOv8n model was chosen as it offers the best balance of:
- ✅ Low VRAM usage (fits easily in 4GB VRAM)
- ✅ Fast inference speed
- ✅ Good accuracy for single-class detection

## Final Performance
| Metric         | Value    |
|----------------|----------|
| Precision      | 0.74093  |
| Recall         | 0.91048  |
| F1-score       | 0.817    |
| mAP50          | 0.73835  |
| mAP50-95       | 0.58531  |

## Training Curves
- **Loss Curves**: [Loss Curves](outputs/plots/results.png)
- **PR Curve**: [PR Curve](outputs/plots/BoxPR_curve.png)
- **Confusion Matrix**: [Confusion Matrix](outputs/plots/confusion_matrix.png)

## Benchmark Results
| Metric           | Value     |
|------------------|-----------|
| GPU FPS          | 30.6      |
| Avg. Latency (ms)| 32.683    |
| Wall Clock FPS   | 27.03     |

## Project Structure
```
skywalker-object-detection/
├── configs/
│   └── train_config.yaml     # Training configuration
├── src/
│   ├── train.py             # Training pipeline
│   ├── evaluate.py          # Evaluation pipeline
│   ├── benchmark.py         # Inference benchmarking
│   ├── infer.py             # Inference pipeline
│   └── utils.py             # Helper utilities
├── outputs/
│   ├── weights/             # Trained models
│   │   └── best.pt          # Best performing model
│   ├── metrics/             # Evaluation metrics
│   ├── plots/               # Training curves and visualizations
│   ├── benchmark/           # Benchmark results
│   ├── reports/             # Technical reports
│   └── predictions/         # Sample detection outputs
├── dataset/                 # Raw dataset (ignored in git)
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/rgurbanli16234/skywalker-object-detection.git
   cd skywalker-object-detection
   ```
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training
To train the model using default configuration:
```bash
python3 -m src.train
```

## Evaluation
To evaluate the trained model:
```bash
python3 -m src.evaluate --weights outputs/weights/best.pt
```

## Inference
To run inference on an image:
```bash
python3 -m src.infer --weights outputs/weights/best.pt --source path/to/image.jpg
```

## Export
To export the model to ONNX:
```bash
python3 -c "from ultralytics import YOLO; model = YOLO('outputs/weights/best.pt'); model.export(format='onnx', imgsz=320)"
```

## Results
Sample detection outputs are available in `outputs/predictions/`

## Future Work
- [ ] Increase image size for better accuracy
- [ ] Implement data augmentation for better generalization
- [ ] Add class weights for balanced training
- [ ] Add more model export formats (TensorRT, CoreML)
- [ ] Implement object tracking

## Author
**Resul Qurbanli**

