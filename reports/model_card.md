# Model Card: Skywalker YOLOv8m

## Model Details
- **Architecture**: YOLOv8m (Medium)
- **Framework**: PyTorch
- **Task**: Object Detection (Bounding Box)
- **Parameters**: ~25.9M
- **FLOPs**: ~78.9 GFLOPs

## Intended Use
- **Primary Use Case**: Real-time object detection and localization within the specified custom domain (e.g., lightsabers).
- **Out of Scope**: Tracking (requires deep-sort/ByteTrack integration), instance segmentation, and pose estimation.

## Training Data
- **Dataset**: Custom Domain Dataset (44,000 instances).
- **Format**: YOLO format (normalized `[x_center, y_center, width, height]`).
- **Split**: Configured via `data.yaml` (Train/Val sets).

## Performance Summary (Epoch 1 Dry-Run Baseline)
- **Precision**: 0.00333
- **Recall**: 1.000
- **mAP@50**: 0.00513
- **Latency (ONNX, Batch=1)**: ~25.2ms

## Limitations & Ethical Considerations
- **Environmental Constraints**: Model performance may degrade under severe low-light conditions, heavy occlusion, or significant domain shift (e.g., thermal imaging).
- **Bias**: The model is inherently biased toward the spatial distribution and lighting conditions present in the training dataset.

## Citation
If utilizing this model or repository for academic purposes, please cite:
```bibtex
@software{skywalker_yolov8,
  author = {Rasul Gurbanli},
  title = {Skywalker Object Detection: Production-Grade YOLOv8},
  year = {2026},
  url = {https://github.com/rasul-gurbanli/skywalker-object-detection}
}
```
