
from pathlib import Path
import numpy as np

def convert_polygon_to_bbox(points):
    # points are (x1, y1, x2, y2, x3, y3, ...)
    x_coords = []
    y_coords = []
    for i in range(0, len(points), 2):
        x_coords.append(float(points[i]))
        y_coords.append(float(points[i+1]))
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Convert to YOLO bbox format: x_center, y_center, width, height
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_center, y_center, width, height]

def process_label_file(label_path):
    with open(label_path, 'r') as f:
        lines = f.read().splitlines()
    
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        
        parts = line.strip().split()
        if len(parts) < 5:
            new_lines.append(line)
            continue
        
        class_id = int(parts[0])
        polygon_points = parts[1:]
        
        bbox = convert_polygon_to_bbox(polygon_points)
        
        new_line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
        new_lines.append(new_line)
    
    with open(label_path, 'w') as f:
        for line in new_lines:
            f.write(line + '\n')

def main():
    dataset_dir = Path("dataset")
    
    for split in ["train", "val"]:
        labels_dir = dataset_dir / "labels" / split
        print(f"Processing {split} labels in {labels_dir}")
        
        label_files = list(labels_dir.glob("*.txt"))
        
        for idx, label_file in enumerate(label_files):
            process_label_file(label_file)
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(label_files)} files")
    
    print("Label conversion complete!")

if __name__ == "__main__":
    main()

