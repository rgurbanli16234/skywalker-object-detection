import os
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
import random

from ultralytics import YOLO

# Import local utilities
from utils import (
    setup_logger,
    verify_environment,
    load_yaml_config,
    prepare_output_directories,
    DatasetError,
    ConfigError
)

# Initialize logger
logger = setup_logger("train_pipeline")

# =====================================================================
# Reproducibility Setup
# =====================================================================

def set_seeds(seed: int = 42) -> None:
    """
    Sets random seeds for Python, NumPy, PyTorch, and CUDA to ensure reproducible training runs.
    
    Args:
        seed (int): Seed number to set.
    """
    logger.info(f"Setting seed: {seed} for reproducibility.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure PyTorch backends for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =====================================================================
# Training Engine
# =====================================================================

class YOLOv8Trainer:
    """
    Modular training orchestrator for YOLOv8 Object Detection on custom datasets.
    """
    
    def __init__(self, config_path: str):
        """
        Initializes the trainer with hyperparameters from a configuration file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.config = self._load_and_validate_config()
        
    def _load_and_validate_config(self) -> Dict[str, Any]:
        """
        Loads the YAML config and ensures crucial parameters are present.
        
        Returns:
            Dict[str, Any]: Verified configuration dictionary.
        """
        config = load_yaml_config(self.config_path)
        required_keys = ["model", "data", "epochs", "imgsz", "batch", "device"]
        for key in required_keys:
            if key not in config:
                raise ConfigError(f"Missing required parameter '{key}' in training config: {self.config_path}")
        return config

    def run_training(self, dry_run: bool = False, epochs_override: int = None, batch_override: int = None, data_override: str = None) -> Path:
        """
        Executes the YOLOv8 training loop.
        
        Args:
            dry_run (bool): If True, runs training for 1 epoch with batch size 2 for pipeline validation.
            epochs_override (int): Optional epoch override.
            batch_override (int): Optional batch size override.
            data_override (str): Optional data config file override.
            
        Returns:
            Path: Directory path where the run outputs were saved.
            
        Raises:
            RuntimeError: If training fails due to environment or hardware errors.
        """
        # 1. Prepare outputs
        prepare_output_directories()
        
        # 2. Check environment & dataset validity
        try:
            verify_environment()
        except DatasetError as e:
            logger.error(f"Dataset verification failed: {e}")
            sys.exit(1)
            
        # 3. Configure determinism
        if self.config.get("deterministic", True):
            seed = self.config.get("seed", 42)
            set_seeds(seed)
            
        # 4. Resolve parameters (checking overrides)
        model_name = self.config["model"]
        epochs = epochs_override if epochs_override is not None else self.config["epochs"]
        batch = batch_override if batch_override is not None else self.config["batch"]
        data_yaml = data_override if data_override is not None else self.config["data"]
        imgsz = self.config["imgsz"]
        amp = self.config.get("amp", True)
        device = self.config["device"]
        
        # Resolve fraction and val for dry-run
        fraction = self.config.get("fraction", 1.0)
        val = self.config.get("val", True)
        if dry_run:
            logger.info("Dry-run requested. Overriding hyperparameters for pipeline test: 1 epoch, batch size 2, fraction 1.0, val True (on dummy).")
            epochs = 1
            batch = 2
            fraction = 1.0
            val = True
            
        # Resolve project to absolute path to avoid runs/detect prefixing
        project_path = self.config.get("project", "outputs")
        project_abs_path = str(Path(project_path).resolve())
            
        # Override device fallback if CUDA isn't actually available
        if device != "cpu" and not torch.cuda.is_available():
            logger.warning("CUDA specified but GPU is not available. Falling back to CPU for training.")
            device = "cpu"
            
        logger.info("Initializing YOLOv8 training with properties:")
        logger.info(f"  - Pretrained model: {model_name}")
        logger.info(f"  - Dataset config:   {data_yaml}")
        logger.info(f"  - Total epochs:     {epochs}")
        logger.info(f"  - Batch size:       {batch}")
        logger.info(f"  - Image resolution: {imgsz}")
        logger.info(f"  - Mixed Precision:  {amp}")
        logger.info(f"  - Hardware device:  {device}")
        logger.info(f"  - Dataset fraction: {fraction}")
        logger.info(f"  - Run validation:   {val}")
        logger.info(f"  - Output project:   {project_abs_path}")
        
        # 5. Load model weights
        try:
            logger.info(f"Loading pretrained weights for: {model_name}")
            model = YOLO(model_name)
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Model initialization failure: {e}")
            
        # 6. Execute model training
        try:
            logger.info("Starting model training...")
            # We train utilizing parameters from train_config.yaml
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                amp=amp,
                fraction=fraction,
                val=val,
                deterministic=self.config.get("deterministic", True),
                plots=self.config.get("plots", True),
                save=self.config.get("save", True),
                project=project_abs_path,
                name=self.config.get("name", "skywalker_yolov8m"),
                exist_ok=self.config.get("exist_ok", True),
                patience=self.config.get("patience", 15),
                seed=self.config.get("seed", 42),
                verbose=self.config.get("verbose", True)
            )
            
            run_save_dir = Path(results.save_dir)
            logger.info(f"Training completed successfully. Weights and curves saved to: {run_save_dir.absolute()}")
            return run_save_dir
            
        except torch.cuda.OutOfMemoryError as e:
            logger.critical("CUDA Out of Memory (OOM) error detected during training.")
            logger.critical(
                "Suggested fixes: \n"
                "  1. Reduce your batch size (e.g. '--batch 8' or '--batch 4')\n"
                "  2. Reduce image resolution (e.g. '--imgsz 416' or '--imgsz 320')\n"
                "  3. Free other processes occupying the GPU VRAM."
            )
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred during training loop: {e}")
            raise e

# =====================================================================
# Main CLI Entry Point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skywalker YOLOv8 Object Detection Training Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/train_config.yaml", 
        help="Path to YAML training config file"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        help="Override path to dataset configuration data.yaml file"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="If set, runs 1 training epoch with batch size 2 to verify scripts work end-to-end"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Override total epochs for training"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        help="Override batch size for training"
    )
    
    args = parser.parse_args()
    
    try:
        trainer = YOLOv8Trainer(config_path=args.config)
        trainer.run_training(
            dry_run=args.dry_run,
            epochs_override=args.epochs,
            batch_override=args.batch,
            data_override=args.data
        )
    except Exception as e:
        logger.exception("Pipeline execution failed:")
        sys.exit(1)
