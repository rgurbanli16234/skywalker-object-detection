from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO
from pydantic import BaseModel
import time
import io
from PIL import Image
import numpy as np
import os
import psutil

app = FastAPI(title="Skywalker Object Detection API",
              description="Enterprise API for real-time bounding box inference using YOLOv8",
              version="1.0.0")

# Global model instance
MODEL_PATH = os.getenv("MODEL_PATH", "/app/release/onnx/best.onnx")
model = None

# Metrics tracking
startup_time = time.time()
total_requests = 0
total_inference_time = 0.0

@app.on_event("startup")
async def load_model():
    global model
    try:
        # Load the ONNX exported model or PyTorch depending on env
        model = YOLO(MODEL_PATH, task='detect')
        # Warmup
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        model.predict(dummy_img, verbose=False)
        print(f"Model {MODEL_PATH} loaded and warmed up successfully.")
    except Exception as e:
        print(f"Failed to load model from {MODEL_PATH}: {e}")
        # Not exiting so /health can report failure, in production you might want to crash

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_path": MODEL_PATH}

@app.get("/metrics")
async def metrics():
    uptime = time.time() - startup_time
    avg_inference = (total_inference_time / total_requests) if total_requests > 0 else 0
    return {
        "uptime_seconds": round(uptime, 2),
        "total_requests": total_requests,
        "average_inference_time_ms": round(avg_inference * 1000, 2),
        "memory_usage_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf_threshold: float = 0.25, iou_threshold: float = 0.45):
    global total_requests, total_inference_time
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    start_time = time.perf_counter()
    results = model.predict(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
    inference_time = time.perf_counter() - start_time
    
    total_requests += 1
    total_inference_time += inference_time

    # Process results
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id] if model.names else str(cls_id)
            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })

    return JSONResponse(content={
        "status": "success",
        "inference_time_ms": round(inference_time * 1000, 2),
        "detections": detections,
        "image_size": image.size
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
