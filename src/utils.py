import os
import sys
import logging
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Union

# =====================================================================
# Custom Exceptions for Robust Error Handling
# =====================================================================

class DatasetError(Exception):
    """Exception raised for errors in the dataset structure or files."""
    pass

class ConfigError(Exception):
    """Exception raised for errors in configuration files."""
    pass

class HardwareError(Exception):
    """Exception raised when hardware constraints are violated."""
    pass

# =====================================================================
# Logging System Setup
# =====================================================================

def setup_logger(name: str = "skywalker_pipeline") -> logging.Logger:
    """
    Sets up a dual-destination logger that records messages to both the console 
    and a persistent log file in the project logs directory.

    Args:
        name (str): The name of the logger instance.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Create logs directory
    log_dir = Path("/home/rasul-gurbanli/Desktop/skywalker_20/skywalker_20/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"
    
    # Custom format for enterprise logging
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler (keeps logs on disk)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    # Console handler (prints to screen)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logger()

# =====================================================================
# Environment Verification & Checks
# =====================================================================

def verify_environment(dataset_root: Union[str, Path] = "/home/rasul-gurbanli/Desktop/skywalker_20/skywalker_20/dataset") -> Dict[str, Any]:
    """
    Verifies that the hardware (GPU/CUDA) and dataset directories exist 
    and are properly configured.

    Args:
        dataset_root (str or Path): Path to the dataset root folder.

    Returns:
        Dict[str, Any]: Summary of verified environment details.
    
    Raises:
        DatasetError: If dataset folders do not exist or are empty.
    """
    logger.info("Initializing environment verification check...")
    env_info = {}
    
    # 1. Hardware/GPU checks
    cuda_available = torch.cuda.is_available()
    env_info["cuda_available"] = cuda_available
    
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        device_properties = torch.cuda.get_device_properties(0)
        vram_gb = device_properties.total_memory / (1024 ** 3)
        env_info["device_name"] = device_name
        env_info["vram_gb"] = round(vram_gb, 2)
        env_info["cuda_version"] = torch.version.cuda
        
        logger.info(f"CUDA GPU detected: {device_name}")
        logger.info(f"Total GPU VRAM: {vram_gb:.2f} GB")
        logger.info(f"CUDA Toolkit Version: {torch.version.cuda}")
        
        if vram_gb < 6.0:
            logger.warning(
                f"GPU VRAM is relatively low ({vram_gb:.2f} GB). "
                "Consider setting a smaller batch size if out-of-memory errors occur during training."
            )
    else:
        logger.warning("CUDA is not available. Execution will fall back to CPU, which is NOT recommended for training.")
        env_info["device_name"] = "CPU"
        env_info["vram_gb"] = 0.0
        env_info["cuda_version"] = "N/A"
        
    # 2. Dataset checks
    dataset_path = Path(dataset_root)
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    
    if not dataset_path.exists():
        msg = f"Dataset root directory not found at: {dataset_path.absolute()}"
        logger.error(msg)
        raise DatasetError(msg)
        
    if not images_dir.exists() or not images_dir.is_dir():
        msg = f"Images subdirectory not found at: {images_dir.absolute()}"
        logger.error(msg)
        raise DatasetError(msg)
        
    if not labels_dir.exists() or not labels_dir.is_dir():
        msg = f"Labels subdirectory not found at: {labels_dir.absolute()}"
        logger.error(msg)
        raise DatasetError(msg)
        
    # Count files to verify there's data
    img_files = list(images_dir.glob("*.[jJ][pP][gG]")) + list(images_dir.glob("*.[jJ][pP][eE][gG]")) + list(images_dir.glob("*.[pP][nN][gG]"))
    lbl_files = list(labels_dir.glob("*.txt"))
    
    env_info["num_images"] = len(img_files)
    env_info["num_labels"] = len(lbl_files)
    
    logger.info(f"Verified dataset directory: {dataset_path.absolute()}")
    logger.info(f"Found {len(img_files)} images and {len(lbl_files)} label files.")
    
    if len(img_files) == 0:
        msg = f"No images found in: {images_dir.absolute()}"
        logger.error(msg)
        raise DatasetError(msg)
        
    if len(lbl_files) == 0:
        msg = f"No label text files found in: {labels_dir.absolute()}"
        logger.error(msg)
        raise DatasetError(msg)
        
    logger.info("Environment verification check completed successfully.")
    return env_info

# =====================================================================
# Configuration File Utilities
# =====================================================================

def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely loads configurations from a YAML file.

    Args:
        config_path (str or Path): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed configuration values as a dictionary.

    Raises:
        ConfigError: If file loading fails or structure is malformed.
    """
    path = Path(config_path)
    if not path.exists():
        msg = f"Configuration file not found at: {path.absolute()}"
        logger.error(msg)
        raise ConfigError(msg)
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            msg = f"Malformed YAML file structure at {path.absolute()}. Must represent a key-value dictionary."
            logger.error(msg)
            raise ConfigError(msg)
        logger.info(f"Successfully loaded configuration file: {path.name}")
        return config
    except yaml.YAMLError as e:
        msg = f"Failed to parse YAML file at {path.absolute()}: {e}"
        logger.error(msg)
        raise ConfigError(msg)
    except Exception as e:
        msg = f"Error reading file {path.absolute()}: {e}"
        logger.error(msg)
        raise ConfigError(msg)

def prepare_output_directories(project_root: Union[str, Path] = "/home/rasul-gurbanli/Desktop/skywalker_20/skywalker_20") -> None:
    """
    Ensures that standard output paths (`outputs/`, `logs/`) exist before run.

    Args:
        project_root (str or Path): The root folder of the project.
    """
    root = Path(project_root)
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    logger.info("Output and logging directory structure prepared.")
