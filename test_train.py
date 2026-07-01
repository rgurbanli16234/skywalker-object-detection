
from ultralytics import YOLO
import torch
import sys

print("Starting YOLOv8x training...")

model = YOLO("yolov8x.pt")

results = model.train(
    data="data.yaml",
    epochs=10,
    imgsz=640,
    batch=4,
    device="0",
    amp=True,
    patience=5,
    seed=42,
    cos_lr=True,
    save=True,
    plots=True,
    project="outputs",
    name="skywalker_yolov8x_small",
    exist_ok=True
)

print("Training complete!")
