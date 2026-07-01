import os
import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    print("Starting Phase 2: Final Artifact Generation")
    base_dir = Path("/home/rasul-gurbanli/Desktop/skywalker_20/skywalker_20")
    release_dir = base_dir / "release"
    
    # Create directories
    for d in ["onnx", "torchscript", "tensorrt", "quantized", "configs", "metrics", "plots", "examples"]:
        (release_dir / d).mkdir(parents=True, exist_ok=True)
        
    best_pt_path = base_dir / "outputs" / "skywalker_yolov8m" / "weights" / "best.pt"
    if not best_pt_path.exists():
        print(f"Error: {best_pt_path} not found.")
        return
        
    shutil.copy(best_pt_path, release_dir / "best.pt")
    
    last_pt_path = base_dir / "outputs" / "skywalker_yolov8m" / "weights" / "last.pt"
    if last_pt_path.exists():
        shutil.copy(last_pt_path, release_dir / "last.pt")
        
    print("Loading best.pt for export...")
    model = YOLO(best_pt_path)
    
    try:
        print("Exporting to ONNX (FP32)...")
        # Half=False for FP32 ONNX
        model.export(format="onnx", half=False, imgsz=640, optimize=True)
        onnx_file = best_pt_path.with_suffix('.onnx')
        if onnx_file.exists():
            shutil.move(str(onnx_file), release_dir / "onnx" / "best.onnx")
    except Exception as e:
        print(f"Failed ONNX export: {e}")
        
    try:
        print("Exporting to TorchScript...")
        model.export(format="torchscript", imgsz=640)
        ts_file = best_pt_path.with_suffix('.torchscript')
        if ts_file.exists():
            shutil.move(str(ts_file), release_dir / "torchscript" / "best.torchscript")
    except Exception as e:
        print(f"Failed TorchScript export: {e}")
        
    try:
        print("Exporting to TensorRT (FP16)...")
        # This might fail if TensorRT isn't installed or no GPU
        model.export(format="engine", half=True, imgsz=640)
        engine_file = best_pt_path.with_suffix('.engine')
        if engine_file.exists():
            shutil.move(str(engine_file), release_dir / "tensorrt" / "best.engine")
    except Exception as e:
        print(f"Failed TensorRT export: {e}")
        
    try:
        print("Exporting to ONNX (INT8 / Quantized) - using int8 flag...")
        # INT8 requires calibration data, but ultralytics provides a simple int8 flag
        model.export(format="onnx", int8=True, imgsz=640, data=str(base_dir / "data.yaml"))
        # The exported file might just be best.onnx again
        onnx_file = best_pt_path.with_suffix('.onnx')
        if onnx_file.exists():
            shutil.move(str(onnx_file), release_dir / "quantized" / "best_int8.onnx")
    except Exception as e:
        print(f"Failed INT8 export: {e}")
        
    # Copy configs, metrics, plots
    print("Copying artifacts...")
    if (base_dir / "data.yaml").exists():
        shutil.copy(base_dir / "data.yaml", release_dir / "configs" / "data.yaml")
    if (base_dir / "configs" / "train_config.yaml").exists():
        shutil.copy(base_dir / "configs" / "train_config.yaml", release_dir / "configs" / "train_config.yaml")
        
    outputs_dir = base_dir / "outputs" / "skywalker_yolov8m"
    if outputs_dir.exists():
        for f in outputs_dir.glob("*.csv"):
            shutil.copy(f, release_dir / "metrics" / f.name)
        for f in outputs_dir.glob("*.png"):
            shutil.copy(f, release_dir / "plots" / f.name)
        for f in outputs_dir.glob("*.jpg"):
            shutil.copy(f, release_dir / "examples" / f.name)
            
    print("Export phase complete.")

if __name__ == "__main__":
    main()
