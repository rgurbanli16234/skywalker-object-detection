import os
import argparse
import time
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

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
logger = setup_logger("benchmark_pipeline")

# =====================================================================
# Latency and FPS Benchmarker
# =====================================================================

class YOLOv8Benchmarker:
    """
    Benchmarks YOLOv8 inference speed, frame rate (FPS), and latency components (preprocess, inference, postprocess).
    """
    
    def __init__(self, weights_path: str, dataset_images_dir: str, output_dir: str):
        """
        Initializes the benchmarker.
        
        Args:
            weights_path (str): Path to trained YOLOv8 model weights (.pt).
            dataset_images_dir (str): Path to directory containing benchmark images.
            output_dir (str): Folder where benchmark reports will be saved.
        """
        self.weights_path = Path(weights_path)
        self.images_dir = Path(dataset_images_dir)
        self.output_dir = Path(output_dir)
        
    def validate_inputs(self) -> List[Path]:
        """
        Validates model weight files and collects test images.
        
        Returns:
            List[Path]: List of absolute paths to images found in the directory.
            
        Raises:
            FileNotFoundError: If weights or images directory are missing.
            DatasetError: If no images are found.
        """
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {self.weights_path.absolute()}")
            
        if not self.images_dir.exists() or not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found at: {self.images_dir.absolute()}")
            
        # Support common image extensions
        extensions = ["*.[jJ][pP][gG]", "*.[jJ][pP][eE][gG]", "*.[pP][nN][gG]"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(self.images_dir.glob(ext))
            
        if not image_paths:
            raise DatasetError(f"No valid images found in: {self.images_dir.absolute()}")
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return image_paths

    def run_benchmark(self, image_paths: List[Path], device: str = "0", max_samples: int = None) -> Dict[str, Any]:
        """
        Runs the benchmark, including GPU warmup, model inference, speed collection, and metric computation.
        
        Args:
            image_paths (List[Path]): List of images to run inference on.
            device (str): Hardware device to run validation on ('0' or 'cpu').
            max_samples (int): Optional cap on number of images to process.
            
        Returns:
            Dict[str, Any]: Compiled benchmark results.
        """
        logger.info(f"Loading YOLOv8 model weights from: {self.weights_path}")
        model = YOLO(self.weights_path)
        
        # Override device fallback if CUDA isn't actually available
        if device != "cpu" and not torch.cuda.is_available():
            logger.warning("CUDA specified but GPU is not available. Falling back to CPU for benchmarking.")
            device = "cpu"
            
        # Sample capping for fast testing
        if max_samples is not None and max_samples < len(image_paths):
            logger.info(f"Capping benchmarking sample size to: {max_samples} images")
            image_paths = image_paths[:max_samples]
            
        num_images = len(image_paths)
        
        # 1. Hardware Warmup (critical for accurate GPU benchmarking)
        if device != "cpu":
            logger.info("Starting GPU hardware warmup (10 iterations)...")
            warmup_img = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(10):
                _ = model.predict(warmup_img, device=device, verbose=False)
            logger.info("Warmup complete.")
        else:
            logger.info("CPU device selected. Skipping GPU warmup phase.")
            
        # 2. Benchmark Loop
        logger.info(f"Benchmarking inference speed over {num_images} images on device '{device}'...")
        
        preprocess_times = []
        inference_times = []
        postprocess_times = []
        total_latencies = []
        wall_clock_times = []
        
        # Disable logging for individual predictions to ensure clean performance tracking
        for idx, img_path in enumerate(image_paths):
            start_wall = time.perf_counter()
            
            # Predict using model
            results = model.predict(
                source=str(img_path),
                device=device,
                imgsz=640,
                verbose=False
            )
            
            end_wall = time.perf_counter()
            
            if not results:
                continue
                
            res = results[0]
            # Speed dict contains timings in milliseconds
            pre = res.speed.get("preprocess", 0.0)
            inf = res.speed.get("inference", 0.0)
            post = res.speed.get("postprocess", 0.0)
            
            preprocess_times.append(pre)
            inference_times.append(inf)
            postprocess_times.append(post)
            total_latencies.append(pre + inf + post)
            wall_clock_times.append((end_wall - start_wall) * 1000.0) # convert to ms
            
            if (idx + 1) % 100 == 0 or (idx + 1) == num_images:
                logger.info(f"Processed {idx + 1}/{num_images} images...")

        # 3. Compute Metrics
        avg_preprocess = np.mean(preprocess_times)
        avg_inference = np.mean(inference_times)
        avg_postprocess = np.mean(postprocess_times)
        avg_total_latency = np.mean(total_latencies)
        avg_wall_clock = np.mean(wall_clock_times)
        
        # FPS = 1000 ms / average latency
        fps = 1000.0 / avg_total_latency if avg_total_latency > 0 else 0.0
        wall_clock_fps = 1000.0 / avg_wall_clock if avg_wall_clock > 0 else 0.0
        
        # Percentiles for latency consistency (jitter)
        p50_latency = np.percentile(total_latencies, 50)
        p95_latency = np.percentile(total_latencies, 95)
        p99_latency = np.percentile(total_latencies, 99)
        
        benchmark_results = {
            "device": device,
            "num_images_benchmarked": num_images,
            "latency_ms": {
                "preprocess": round(avg_preprocess, 3),
                "inference": round(avg_inference, 3),
                "postprocess": round(avg_postprocess, 3),
                "total_average": round(avg_total_latency, 3),
                "p50_median": round(p50_latency, 3),
                "p95_percentile": round(p95_latency, 3),
                "p99_percentile": round(p99_latency, 3)
            },
            "fps": round(fps, 2),
            "wall_clock_ms_per_image": round(avg_wall_clock, 3),
            "wall_clock_fps": round(wall_clock_fps, 2)
        }
        
        self._print_results_table(benchmark_results)
        return benchmark_results

    def _print_results_table(self, res: Dict[str, Any]) -> None:
        """
        Prints a neat ASCII table summarizing benchmark performance metrics.
        """
        print("\n" + "="*50)
        print("          YOLOv8 INFERENCE BENCHMARK REPORT          ")
        print("="*50)
        print(f" Device:                 {res['device']}")
        print(f" Images Evaluated:       {res['num_images_benchmarked']}")
        print("-"*50)
        print(f" Preprocess Latency:     {res['latency_ms']['preprocess']:.2f} ms")
        print(f" Model Inference:        {res['latency_ms']['inference']:.2f} ms")
        print(f" Postprocess Latency:    {res['latency_ms']['postprocess']:.2f} ms")
        print(f" Total Model Latency:    {res['latency_ms']['total_average']:.2f} ms")
        print(f" Wall-clock time/Image:  {res['wall_clock_ms_per_image']:.2f} ms")
        print("-"*50)
        print(f" Model Pipeline FPS:     {res['fps']:.2f} frames/sec")
        print(f" System End-to-End FPS:  {res['wall_clock_fps']:.2f} frames/sec")
        print("-"*50)
        print(f" Latency Jitter (p50):   {res['latency_ms']['p50_median']:.2f} ms")
        print(f" Latency Jitter (p95):   {res['latency_ms']['p95_percentile']:.2f} ms")
        print(f" Latency Jitter (p99):   {res['latency_ms']['p99_percentile']:.2f} ms")
        print("="*50 + "\n")

    def export_results(self, results: Dict[str, Any]) -> None:
        """
        Saves the results dictionary as a JSON file.
        
        Args:
            results (Dict[str, Any]): Dictionary of benchmark results.
        """
        json_path = self.output_dir / "benchmark_results.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Successfully saved benchmark report to JSON: {json_path.absolute()}")
        except Exception as e:
            logger.error(f"Failed to export benchmark JSON: {e}")

# =====================================================================
# Main CLI Entry Point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skywalker YOLOv8 Benchmarking Script")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="outputs/skywalker_yolov8m/weights/best.pt", 
        help="Path to trained YOLOv8 model weights (.pt)"
    )
    parser.add_argument(
        "--images-dir", 
        type=str, 
        default="dataset/images", 
        help="Directory containing validation dataset images"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs", 
        help="Folder to save benchmark reports"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="0", 
        help="Device to benchmark on (e.g. '0' or 'cpu')"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        help="Cap the number of images benchmarked for quick runs"
    )
    
    args = parser.parse_args()
    
    benchmarker = YOLOv8Benchmarker(
        weights_path=args.weights,
        dataset_images_dir=args.images_dir,
        output_dir=args.output_dir
    )
    
    try:
        image_paths = benchmarker.validate_inputs()
        results = benchmarker.run_benchmark(
            image_paths=image_paths,
            device=args.device,
            max_samples=args.max_samples
        )
        benchmarker.export_results(results)
    except FileNotFoundError as e:
        logger.error(f"Input validation failure: {e}")
        sys.exit(1)
    except DatasetError as e:
        logger.error(f"Dataset error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception("Benchmarker execution failed:")
        sys.exit(1)
