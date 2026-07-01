import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

model_path = "best.pt" # Should be copied to this dir during deployment
model = YOLO(model_path)

def predict(image, conf_threshold, iou_threshold):
    if image is None:
        return None, "Please upload an image."
        
    results = model.predict(source=image, conf=conf_threshold, iou=iou_threshold)
    
    # Plot results
    res_img = results[0].plot()
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    
    # Format metrics
    metrics = f"Detected {len(results[0].boxes)} objects.\n"
    for idx, box in enumerate(results[0].boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        metrics += f"- {cls_name} (Conf: {conf:.2f})\n"
        
    return res_img_rgb, metrics

with gr.Blocks(title="Skywalker Object Detection") as demo:
    gr.Markdown("# Skywalker Object Detection (YOLOv8m)")
    gr.Markdown("Upload an image to detect custom objects. Adjust confidence and IoU thresholds dynamically.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            conf_slider = gr.Slider(minimum=0.01, maximum=1.0, value=0.25, step=0.01, label="Confidence Threshold")
            iou_slider = gr.Slider(minimum=0.01, maximum=1.0, value=0.45, step=0.01, label="IoU Threshold")
            submit_btn = gr.Button("Detect Objects")
            
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Detection Visualization")
            metrics_text = gr.Textbox(label="Detection Metrics", lines=10)
            
    submit_btn.click(predict, inputs=[input_image, conf_slider, iou_slider], outputs=[output_image, metrics_text])
    
    gr.Examples(
        examples=[
            ["examples/sample1.jpg", 0.25, 0.45],
            ["examples/sample2.jpg", 0.25, 0.45],
        ],
        inputs=[input_image, conf_slider, iou_slider],
    )

if __name__ == "__main__":
    demo.launch()
