
import os
import gc
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path to import from src/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import random

from ultralytics import YOLO

# Import local utilities
from src.utils import (
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
# Reproducibility & VRAM Optimization Setup
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
    
    # Configure PyTorch backends for speed (not deterministic for low VRAM)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def log_vram_stats():
    """
    Logs VRAM usage statistics.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3
        logger.info(f"VRAM Stats: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Free: {free:.2f} GB")


def cleanup():
    """
    Cleans up memory (VRAM and RAM).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =====================================================================
# Training Engine - Optimized for Low VRAM
# =====================================================================

class YOLOv8Trainer:
    """
    Modular training orchestrator for YOLOv8 Object Detection on custom datasets, optimized for low VRAM.
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

    def run_training(self, dry_run: bool = False, epochs_override: int = None, batch_override: int = None, 
                     data_override: str = None, resume: bool = False) -> Path:
        """
        Executes the YOLOv8 training loop with low VRAM optimizations.
        
        Args:
            dry_run (bool): If True, runs training for 1 epoch with batch size 2 for pipeline validation.
            epochs_override (int): Optional epoch override.
            batch_override (int): Optional batch size override.
            data_override (str): Optional data config file override.
            resume (bool): If True, resume training from last.pt
            
        Returns:
            Path: Directory path where the run outputs were saved.
            
        Raises:
            RuntimeError: If training fails due to environment or hardware errors.
        """
        # 1. Prepare outputs
        prepare_output_directories()
        
        # 2. Clean up memory before starting
        cleanup()
        log_vram_stats()
        
        # 3. Check environment & dataset validity
        try:
            verify_environment()
        except DatasetError as e:
            logger.error(f"Dataset verification failed: {e}")
            sys.exit(1)
            
        # 4. Configure determinism
        if self.config.get("deterministic", False):
            seed = self.config.get("seed", 42)
            set_seeds(seed)
            
        # 5. Resolve parameters (checking overrides)
        model_name = self.config["model"]
        epochs = epochs_override if epochs_override is not None else self.config["epochs"]
        batch = batch_override if batch_override is not None else self.config["batch"]
        data_yaml = data_override if data_override is not None else self.config["data"]
        imgsz = self.config["imgsz"]
        amp = self.config.get("amp", True)
        device = self.config["device"]
        workers = self.config.get("workers", 1)
        
        # Resolve fraction and val for dry-run
        fraction = self.config.get("fraction", 1.0)
        val = self.config.get("val", True)
        if dry_run:
            logger.info("Dry-run requested. Overriding hyperparameters for pipeline test.")
            epochs = 1
            batch = 2
            fraction = 1.0
            val = True
            
        # Resolve project to absolute path
        project_path = self.config.get("project", "outputs")
        project_abs_path = str(Path(project_path).resolve())
        run_name = self.config.get("name", "skywalker_yolov8n_low_vram")
            
        # Check if last.pt exists for resume
        last_checkpoint = Path(project_abs_path) / run_name / "weights" / "last.pt"
        if resume and last_checkpoint.exists():
            logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
            model_name = str(last_checkpoint)
            resume_flag = True
        else:
            resume_flag = False
            
        # Override device fallback if CUDA isn't available
        if device != "cpu" and not torch.cuda.is_available():
            logger.warning("CUDA specified but GPU is not available. Falling back to CPU for training.")
            device = "cpu"
            
        logger.info("Initializing YOLOv8 training (LOW VRAM MODE) with properties:")
        logger.info(f"  - Pretrained model: {model_name}")
        logger.info(f"  - Dataset config:   {data_yaml}")
        logger.info(f"  - Total epochs:     {epochs}")
        logger.info(f"  - Batch size:       {batch}")
        logger.info(f"  - Image resolution: {imgsz}")
        logger.info(f"  - Mixed Precision:  {amp}")
        logger.info(f"  - Hardware device:  {device}")
        logger.info(f"  - Data workers:     {workers}")
        logger.info(f"  - Resume training:  {resume_flag}")
        
        # 5. Load model weights with cleanup first
        try:
            cleanup()
            log_vram_stats()
            logger.info(f"Loading pretrained weights for: {model_name}")
            model = YOLO(model_name)
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Model initialization failure: {e}")
            
        # 6. Execute model training with OOM fallback to CPU
        try:
            logger.info("Starting model training...")
            log_vram_stats()
            
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                amp=amp,
                fraction=fraction,
                val=val,
                deterministic=self.config.get("deterministic", False),
                plots=self.config.get("plots", True),
                save=self.config.get("save", True),
                project=project_abs_path,
                name=run_name,
                exist_ok=True,
                patience=self.config.get("patience", 20),
                seed=self.config.get("seed", 42),
                verbose=self.config.get("verbose", True),
                lr0=self.config.get("lr0", 0.001),
                lrf=self.config.get("lrf", 0.0001),
                cos_lr=self.config.get("cos_lr", True),
                mosaic=self.config.get("mosaic", 0.0),
                mixup=self.config.get("mixup", 0.0),
                hsv_h=self.config.get("hsv_h", 0.015),
                hsv_s=self.config.get("hsv_s", 0.5),
                hsv_v=self.config.get("hsv_v", 0.3),
                degrees=self.config.get("degrees", 0.0),
                translate=self.config.get("translate", 0.05),
                scale=self.config.get("scale", 0.2),
                shear=self.config.get("shear", 0.0),
                perspective=self.config.get("perspective", 0.0),
                flipud=self.config.get("flipud", 0.0),
                fliplr=self.config.get("fliplr", 0.5),
                copy_paste=self.config.get("copy_paste", 0.0),
                label_smoothing=self.config.get("label_smoothing", 0.0),
                cache=self.config.get("cache", False),
                multi_scale=self.config.get("multi_scale", False),
                workers=workers,
                resume=resume_flag
            )
            
            run_save_dir = Path(results.save_dir)
            logger.info(f"Training completed successfully. Weights and curves saved to: {run_save_dir.absolute()}")
            log_vram_stats()
            return run_save_dir
            
        except torch.cuda.OutOfMemoryError as e:
            logger.critical("CUDA Out of Memory (OOM) error detected!")
            
            # Fallback to CPU training
            logger.info("Falling back to CPU training!")
            cleanup()
            
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device="cpu",
                amp=False,  # CPU doesn't need AMP
                fraction=fraction,
                val=val,
                deterministic=self.config.get("deterministic", False),
                plots=self.config.get("plots", True),
                save=self.config.get("save", True),
                project=project_abs_path,
                name=run_name,
                exist_ok=True,
                patience=self.config.get("patience", 20),
                seed=self.config.get("seed", 42),
                verbose=self.config.get("verbose", True),
                lr0=self.config.get("lr0", 0.001),
                lrf=self.config.get("lrf", 0.0001),
                cos_lr=self.config.get("cos_lr", True),
                mosaic=self.config.get("mosaic", 0.0),
                mixup=self.config.get("mixup", 0.0),
                hsv_h=self.config.get("hsv_h", 0.015),
                hsv_s=self.config.get("hsv_s", 0.5),
                hsv_v=self.config.get("hsv_v", 0.3),
                degrees=self.config.get("degrees", 0.0),
                translate=self.config.get("translate", 0.05),
                scale=self.config.get("scale", 0.2),
                shear=self.config.get("shear", 0.0),
                perspective=self.config.get("perspective", 0.0),
                flipud=self.config.get("flipud", 0.0),
                fliplr=self.config.get("fliplr", 0.5),
                copy_paste=self.config.get("copy_paste", 0.0),
                label_smoothing=self.config.get("label_smoothing", 0.0),
                cache=self.config.get("cache", False),
                multi_scale=self.config.get("multi_scale", False),
                workers=workers,
                resume=resume_flag
            )
            
            run_save_dir = Path(results.save_dir)
            logger.info(f"CPU training completed successfully. Weights saved to: {run_save_dir.absolute()}")
            return run_save_dir
            
        except Exception as e:
            logger.error(f"An unexpected error occurred during training loop: {e}")
            raise e

# =====================================================================
# Main CLI Entry Point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skywalker YOLOv8 Object Detection Training Pipeline - Low VRAM Optimized")
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last.pt checkpoint if exists"
    )
    
    args = parser.parse_args()
    
    try:
        trainer = YOLOv8Trainer(config_path=args.config)
        trainer.run_training(
            dry_run=args.dry_run,
            epochs_override=args.epochs,
            batch_override=args.batch,
            data_override=args.data,
            resume=args.resume
        )
    except Exception as e:
        logger.exception("Pipeline execution failed:")
        sys.exit(1)
