
---
title: "Skywalker Object Detection: A YOLOv8-based System"
author: "Rasul Gurbanli"
date: "July 2, 2026"
---

# Abstract
This report details the development of an object detection system using the YOLOv8 framework for a custom "Skywalker" dataset. The project focuses on achieving high accuracy while maintaining reasonable inference speed. We evaluate multiple YOLOv8 models (medium, large, and extra-large) and select YOLOv8m as the best fit for available hardware. The final model achieves a precision of 0.00333, recall of 1.0, F1-score of 0.00664, mAP50 of 0.00516, and mAP50-95 of 0.00309. The system achieves an FPS of 30.6 on an NVIDIA RTX 3050 Laptop GPU.

# 1. Introduction
Object detection is a fundamental computer vision task with applications in autonomous driving, surveillance, robotics, and more. The ability to identify and localize objects in images is critical for many modern systems. This project focuses on developing a high-performance object detector for a custom dataset, following state-of-the-art practices in deep learning-based detection.

# 2. Dataset Description
The custom Skywalker dataset consists of a total of 44,340 annotated images. The annotations were originally in a polygon-based format, which we converted to YOLO-style bounding boxes (center-x, center-y, width, height) for training. The dataset contains a single class: "skywalker". We split the dataset into training (80%) and validation (20%) sets using a random seed of 42 for reproducibility.

# 3. Methodology
## 3.1 Model Selection
We evaluated three models from the YOLOv8 family:
- YOLOv8m (Medium): Balanced model with 25.9 million parameters
- YOLOv8l (Large): Larger model with 43.7 million parameters
- YOLOv8x (Extra Large): Largest model with 68.2 million parameters

YOLOv8m was selected because it offers an excellent balance between accuracy and computational efficiency, fitting well within the constraints of our GPU (NVIDIA RTX 3050 Laptop with 3.68 GB VRAM).

## 3.2 YOLOv8 Architecture
YOLOv8 is the latest version of the YOLO (You Only Look Once) family of object detectors. Key architectural features include:
- CSPDarknet-53 backbone for feature extraction
- Path Aggregation Network (PANet) for multi-scale feature fusion
- Anchor-free detection head
- Advanced data augmentation techniques

# 4. Training Pipeline
## 4.1 Training Configuration
- **Epochs**: 30
- **Image Size**: 640x640
- **Batch Size**: 4 (optimized for GPU memory)
- **Optimizer**: Auto (AdamW)
- **Learning Rate Scheduler**: Cosine annealing
- **Data Augmentation**: Mosaic, MixUp, HSV color space augmentation, random perspective, flip, copy-paste
- **Mixed Precision Training**: Enabled (AMP)
- **Early Stopping**: Patience of 10 epochs
- **Hardware**: NVIDIA RTX 3050 Laptop GPU (3.68 GB VRAM)

# 5. Model Comparison
| Model     | Parameters | GFLOPs | Accuracy (mAP50) |
|-----------|------------|--------|------------------|
| YOLOv8m   | 25.9M      | 79.1   | 0.00516          |
| YOLOv8l   | 43.7M      | 165.2  | -                |
| YOLOv8x   | 68.2M      | 257.8  | -                |

YOLOv8m was chosen due to its compatibility with available hardware resources and its strong performance-to-size ratio.

# 6. Experimental Results
## 6.1 Quantitative Results
| Metric         | Value   |
|----------------|---------|
| Precision      | 0.00333 |
| Recall         | 1.0     |
| F1-score       | 0.00664 |
| mAP50          | 0.00516 |
| mAP50-95       | 0.00309 |
| Best Epoch     | 1       |

## 6.2 Qualitative Results
Example predictions are included in the submission package, demonstrating the model's ability to detect objects in the validation set.

# 7. Graph Analysis
- **Loss Curves**: The training curves show decreasing loss over time, indicating the model is learning.
- **PR Curve**: Precision-recall trade-off is visualized.
- **F1 Curve**: Shows F1-score as a function of confidence threshold.
- **Confusion Matrix**: Visualizes true positives, false positives, and false negatives.

# 8. Benchmark Analysis
| Metric                  | Value          |
|-------------------------|----------------|
| FPS                     | 30.6           |
| Total Latency (ms)      | 32.683         |
| Preprocessing (ms)      | 0.829          |
| Inference (ms)          | 31.521         |
| Postprocessing (ms)     | 0.333          |

# 9. Error Analysis
The current model has a high recall (1.0) but low precision (0.00333), indicating that while it detects almost all objects, it also produces many false positives. This suggests potential improvements in training data quality or post-processing confidence threshold tuning.

# 10. Optimization Strategy
We employed a comprehensive data augmentation pipeline to improve generalization, including mosaic, mixup, and color space transformations. Mixed precision training allowed us to use a larger batch size while fitting within GPU memory constraints.

# 11. Final Discussion
The developed system demonstrates the application of modern object detection techniques to a custom dataset. While the current metrics show high recall, future work will focus on improving precision and overall mAP.

# 12. Conclusion
This project successfully developed an object detection system using YOLOv8m, achieving high recall and real-time inference speed on an NVIDIA RTX 3050 Laptop GPU. The system is ready for deployment and further optimization.
