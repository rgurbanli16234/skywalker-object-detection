
from ultralytics import YOLO
import torch

def main():
    print("Initializing YOLOv8m training...")
    
    # Load pretrained model
    model = YOLO("yolov8m.pt")
    
    # Train the model with optimized settings for high accuracy
    results = model.train(
        data="data.yaml",
        epochs=100,              # Train for 100 epochs to get high accuracy
        imgsz=640,               # Image size
        batch=4,                 # Batch size for RTX 3050 Laptop (3.68GB VRAM)
        device=0,                # GPU 0
        amp=True,                # Mixed precision
        patience=20,             # Early stopping patience
        seed=42,                 # Reproducibility
        cos_lr=True,             # Cosine learning rate
        lr0=0.001,               # Lower initial learning rate for stability
        lrf=0.0001,              # Lower final learning rate
        mosaic=1.0,              # Heavy mosaic
        mixup=0.15,              # Mixup
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        copy_paste=0.3,
        label_smoothing=0.1,
        cache="ram",
        multi_scale=True,
        project="outputs",
        name="skywalker_yolov8m_high_acc",
        exist_ok=True,
        plots=True
    )
    
    print("Training complete!")
    
    # Evaluate on validation set
    print("\nEvaluating model...")
    metrics = model.val(data="data.yaml", device=0, plots=True)
    print(f"\nFinal Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    # Export best model
    print("\nExporting best.pt to submission folder...")
    import shutil
    shutil.copy("outputs/skywalker_yolov8m_high_acc/weights/best.pt", "submission/best.pt")
    print("Done!")

if __name__ == "__main__":
    main()
