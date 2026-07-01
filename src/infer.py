import os
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Union, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Import local utilities
from utils import (
    setup_logger,
    verify_environment,
    DatasetError
)

# Initialize logger
logger = setup_logger("inference_pipeline")

# =====================================================================
# Bounding Box Renderer (Custom Aesthetic Visualizer)
# =====================================================================

class AestheticVisualizer:
    """
    Renders high-quality bounding boxes and translucent glassmorphism-style labels 
    on images for a professional visual appearance.
    """
    
    # Custom color palette (BGR format)
    # Coral/Neon Orange (239, 131, 84) and Teal (150, 206, 180)
    COLOR_PRIMARY = (84, 131, 239)   # BGR for beautiful Coral/Neon Orange
    COLOR_TEXT = (255, 255, 255)     # BGR for White
    
    @staticmethod
    def draw_prediction(
        img: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        label: str, 
        confidence: float
    ) -> np.ndarray:
        """
        Draws a customized aesthetic bounding box and label tag on the image.
        
        Args:
            img (np.ndarray): Target image to draw on.
            bbox (Tuple[int, int, int, int]): Coordinates (xmin, ymin, xmax, ymax).
            label (str): Class label text.
            confidence (float): Confidence score between 0.0 and 1.0.
            
        Returns:
            np.ndarray: Image with drawn predictions.
        """
        xmin, ymin, xmax, ymax = bbox
        h, w = img.shape[:2]
        
        # Clip bounding box coordinates to image boundaries
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        
        # 1. Draw elegant double-layered bounding box (outer primary color, inner black shadow)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), AestheticVisualizer.COLOR_PRIMARY, 2, lineType=cv2.LINE_AA)
        
        # 2. Prepare text label string
        text_str = f"{label.upper()} | {confidence:.2f}"
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text_str, font_face, font_scale, thickness)
        
        # Determine text background position
        margin = 6
        bg_xmin = xmin - 1
        bg_ymin = ymin - text_h - baseline - margin - 2
        bg_xmax = xmin + text_w + margin
        bg_ymax = ymin + 1
        
        # If label falls off the top, shift it inside the bounding box
        if bg_ymin < 0:
            bg_ymin = ymin + 1
            bg_ymax = ymin + text_h + baseline + margin + 2
            
        # Draw translucent glassmorphism label background
        overlay = img.copy()
        # Draw dark gray translucent tag background
        cv2.rectangle(overlay, (bg_xmin, bg_ymin), (bg_xmax, bg_ymax), (30, 30, 30), cv2.FILLED)
        # Apply translucency blend (alpha = 0.75, beta = 0.25)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        
        # Draw left border accent on label tag
        cv2.rectangle(img, (bg_xmin, bg_ymin), (bg_xmin + 3, bg_ymax), AestheticVisualizer.COLOR_PRIMARY, cv2.FILLED)
        
        # Write text label
        text_y_offset = baseline + margin // 2
        if bg_ymin == ymin + 1:
            text_y = bg_ymin + text_h + text_y_offset
        else:
            text_y = bg_ymax - text_y_offset - 1
            
        cv2.putText(
            img, 
            text_str, 
            (xmin + margin, int(text_y)), 
            font_face, 
            font_scale, 
            AestheticVisualizer.COLOR_TEXT, 
            thickness, 
            lineType=cv2.LINE_AA
        )
        
        return img

# =====================================================================
# Batch Inference Engine
# =====================================================================

class YOLOv8Inferencer:
    """
    Executes modular batch inference on folders of images and saves results.
    """
    
    def __init__(self, weights_path: str, output_dir: str, conf_threshold: float = 0.25):
        """
        Initializes the inferencer.
        
        Args:
            weights_path (str): Path to trained model weights (.pt).
            output_dir (str): Location where predicted images and metadata will be saved.
            conf_threshold (float): Confidence threshold for detections.
        """
        self.weights_path = Path(weights_path)
        self.output_dir = Path(output_dir)
        self.conf_threshold = conf_threshold
        
    def validate_inputs(self, input_source: str) -> List[Path]:
        """
        Validates model checkpoint and compiles list of target images.
        
        Args:
            input_source (str): Path to image file or directory containing images.
            
        Returns:
            List[Path]: List of validated image paths.
        """
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model checkpoint weights file not found: {self.weights_path.absolute()}")
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        source_path = Path(input_source)
        if not source_path.exists():
            raise FileNotFoundError(f"Input source path not found: {source_path.absolute()}")
            
        if source_path.is_file():
            # Single image prediction
            valid_exts = [".jpg", ".jpeg", ".png"]
            if source_path.suffix.lower() in valid_exts:
                return [source_path]
            else:
                raise ValueError(f"Unsupported file format: {source_path.suffix}")
        
        # Directory prediction
        image_paths = []
        for ext in ["*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]", "*.[pP][nN][gG]"]:
            image_paths.extend(source_path.glob(ext))
            
        if not image_paths:
            raise DatasetError(f"No valid images found in folder: {source_path.absolute()}")
            
        return image_paths

    def run_inference(self, image_paths: List[Path], batch_size: int = 16, device: str = "0") -> Dict[str, Any]:
        """
        Runs batch predictions and exports visualized images and detections database.
        
        Args:
            image_paths (List[Path]): List of images to run predictions on.
            batch_size (int): Size of image batches loaded simultaneously.
            device (str): Device to run inference on (e.g. '0' or 'cpu').
            
        Returns:
            Dict[str, Any]: Saved detection outputs metadata.
        """
        logger.info(f"Loading model weights for inference: {self.weights_path}")
        model = YOLO(self.weights_path)
        
        # Override device fallback if CUDA isn't actually available
        if device != "cpu" and not torch.cuda.is_available():
            logger.warning("CUDA specified but GPU is not available. Falling back to CPU for inference.")
            device = "cpu"
            
        num_images = len(image_paths)
        logger.info(f"Running inference on {num_images} images in batches of {batch_size} on device '{device}'...")
        
        all_records = {}
        
        # Batch image list generator
        for i in range(0, num_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_str_paths = [str(p) for p in batch_paths]
            
            # Predict batch
            results = model.predict(
                source=batch_str_paths,
                conf=self.conf_threshold,
                device=device,
                imgsz=640,
                verbose=False
            )
            
            # Process outputs
            for idx, res in enumerate(results):
                original_img_path = batch_paths[idx]
                image_name = original_img_path.name
                
                # Load original image for custom OpenCV drawing (instead of model.plot())
                img = cv2.imread(str(original_img_path))
                if img is None:
                    logger.warning(f"Failed to read image for visualization: {original_img_path}")
                    continue
                    
                detections_list = []
                
                # Check for bounding boxes
                if res.boxes is not None:
                    # Extract classes, confs, and box coordinates
                    boxes = res.boxes.xyxy.cpu().numpy()      # xmin, ymin, xmax, ymax
                    confs = res.boxes.conf.cpu().numpy()      # confidence scores
                    classes = res.boxes.cls.cpu().numpy()     # class float indices
                    names = res.names                         # class dictionary mapping
                    
                    for box, conf, cls_idx in zip(boxes, confs, classes):
                        cls_name = names.get(int(cls_idx), "unknown")
                        bbox_coords = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                        
                        # Add detection details to database record
                        detections_list.append({
                            "class": cls_name,
                            "confidence": float(conf),
                            "bbox": bbox_coords
                        })
                        
                        # Custom draw
                        img = AestheticVisualizer.draw_prediction(
                            img=img,
                            bbox=tuple(bbox_coords),
                            label=cls_name,
                            confidence=conf
                        )
                        
                # Save predicted image
                save_path = self.output_dir / f"pred_{image_name}"
                cv2.imwrite(str(save_path), img)
                
                # Store records
                all_records[image_name] = {
                    "num_detections": len(detections_list),
                    "detections": detections_list,
                    "visualized_image_path": str(save_path.absolute())
                }
                
            completed = min(i + batch_size, num_images)
            logger.info(f"Processed batch predictions: {completed}/{num_images} images")
            
        # Export programmatic predictions database to JSON
        json_path = self.output_dir / "detection_results.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_records, f, indent=4)
            logger.info(f"Exported prediction metadata database to: {json_path.absolute()}")
        except Exception as e:
            logger.error(f"Failed to save JSON prediction results: {e}")
            
        return all_records

# =====================================================================
# Main CLI Entry Point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skywalker YOLOv8 Batch Inference Utility")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="outputs/skywalker_yolov8m/weights/best.pt", 
        help="Path to trained YOLOv8 model weights (.pt)"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        required=True, 
        help="Path to single image file or directory containing test images"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs/predictions", 
        help="Folder to save visualized predictions and JSON outputs"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25, 
        help="Confidence threshold for filtering detections"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16, 
        help="Batch size for loading and predicting images"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="0", 
        help="Hardware device to run inference on (e.g. '0' or 'cpu')"
    )
    
    args = parser.parse_args()
    
    # Import torch to verify GPU states
    import torch
    
    inferencer = YOLOv8Inferencer(
        weights_path=args.weights,
        output_dir=args.output_dir,
        conf_threshold=args.conf
    )
    
    try:
        image_paths = inferencer.validate_inputs(input_source=args.source)
        inferencer.run_inference(
            image_paths=image_paths,
            batch_size=args.batch_size,
            device=args.device
        )
    except FileNotFoundError as e:
        logger.error(f"Input validation failure: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Parameter value error: {e}")
        sys.exit(1)
    except DatasetError as e:
        logger.error(f"Inference input error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Inference engine execution failed:")
        sys.exit(1)
