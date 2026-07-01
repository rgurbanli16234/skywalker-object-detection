import os
import argparse
import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any

from ultralytics import YOLO
import pandas as pd

# Import local utilities
from utils import (
    setup_logger,
    verify_environment,
    DatasetError
)

# Initialize logger
logger = setup_logger("evaluate_pipeline")

# =====================================================================
# Model Evaluator Class
# =====================================================================

class YOLOv8Evaluator:
    """
    Evaluates trained YOLOv8 models on validation datasets and exports performance metrics.
    """
    
    def __init__(self, weights_path: str, data_config: str, output_dir: str):
        """
        Initializes the evaluator.
        
        Args:
            weights_path (str): Path to the PyTorch model checkpoint (.pt).
            data_config (str): Path to the dataset data.yaml file.
            output_dir (str): Folder where evaluation reports will be saved.
        """
        self.weights_path = Path(weights_path)
        self.data_config = Path(data_config)
        self.output_dir = Path(output_dir)
        
    def validate_inputs(self) -> None:
        """
        Validates model checkpoint, dataset configuration, and output directories.
        
        Raises:
            FileNotFoundError: If weights or dataset configs do not exist.
        """
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model checkpoint weights file not found at: {self.weights_path.absolute()}")
            
        if not self.data_config.exists():
            raise FileNotFoundError(f"Dataset configuration file not found at: {self.data_config.absolute()}")
            
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_evaluation(self, device: str = "0") -> Dict[str, float]:
        """
        Runs validation and extracts quantitative object detection metrics.
        
        Args:
            device (str): Device to run validation on (e.g. '0' for CUDA GPU, or 'cpu').
            
        Returns:
            Dict[str, float]: Extracted metrics (Precision, Recall, mAP50, mAP50-95, F1).
        """
        logger.info(f"Loading YOLOv8 weights from: {self.weights_path}")
        model = YOLO(self.weights_path)
        
        # Override device fallback if CUDA isn't actually available
        if device != "cpu" and not torch.cuda.is_available():
            logger.warning("CUDA specified but GPU is not available. Falling back to CPU for evaluation.")
            device = "cpu"
            
        logger.info(f"Starting evaluation run on device '{device}'...")
        
        # Run validation
        results = model.val(
            data=str(self.data_config),
            device=device,
            plots=True, # Ensure PR, F1, confusion matrices are saved/updated
            verbose=True
        )
        
        # Extract metrics safely from results
        try:
            # results.box contains bounding box metrics
            mp = float(results.box.mp)            # Mean Precision
            mr = float(results.box.mr)            # Mean Recall
            map50 = float(results.box.map50)      # mAP50
            map50_95 = float(results.box.map)     # mAP50-95
            
            # Mathematical F1-score computation
            f1_score = 0.0
            if (mp + mr) > 0:
                f1_score = 2.0 * (mp * mr) / (mp + mr)
                
            metrics = {
                "mAP50": round(map50, 5),
                "mAP50-95": round(map50_95, 5),
                "precision": round(mp, 5),
                "recall": round(mr, 5),
                "f1_score": round(f1_score, 5)
            }
            
            logger.info("Evaluation metrics successfully extracted:")
            logger.info(f"  - Precision: {metrics['precision']:.4f}")
            logger.info(f"  - Recall:    {metrics['recall']:.4f}")
            logger.info(f"  - F1-Score:  {metrics['f1_score']:.4f}")
            logger.info(f"  - mAP50:     {metrics['mAP50']:.4f}")
            logger.info(f"  - mAP50-95:  {metrics['mAP50-95']:.4f}")
            
            return metrics
            
        except AttributeError as e:
            logger.error(f"Failed to access model metrics box: {e}")
            raise RuntimeError("The model validation output was empty or malformed.")

    def export_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Exports the metrics dict to JSON and CSV formats.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metrics.
        """
        json_path = self.output_dir / "evaluation_metrics.json"
        csv_path = self.output_dir / "evaluation_metrics.csv"
        
        # Save JSON
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Exported evaluation metrics to JSON: {json_path.absolute()}")
        except Exception as e:
            logger.error(f"Failed to save JSON metrics report: {e}")
            
        # Save CSV (using pandas for clean, standardized formatting)
        try:
            df = pd.DataFrame([metrics])
            df.to_csv(csv_path, index=False)
            logger.info(f"Exported evaluation metrics to CSV: {csv_path.absolute()}")
        except Exception as e:
            logger.error(f"Failed to save CSV metrics report: {e}")

# =====================================================================
# Main CLI Entry Point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skywalker YOLOv8 Evaluation Utility")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="outputs/skywalker_yolov8m/weights/best.pt", 
        help="Path to trained YOLOv8 weights file"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data.yaml", 
        help="Path to dataset configuration data.yaml file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs", 
        help="Directory to save evaluation reports"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="0", 
        help="Hardware device to run validation on (e.g. '0' or 'cpu')"
    )
    
    args = parser.parse_args()
    
    # Import torch here to verify GPU state in evaluator
    import torch
    
    evaluator = YOLOv8Evaluator(
        weights_path=args.weights,
        data_config=args.data,
        output_dir=args.output_dir
    )
    
    try:
        evaluator.validate_inputs()
        metrics = evaluator.run_evaluation(device=args.device)
        evaluator.export_metrics(metrics)
    except FileNotFoundError as e:
        logger.error(f"File not found during verification: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Evaluation execution encountered an unhandled exception:")
        sys.exit(1)
