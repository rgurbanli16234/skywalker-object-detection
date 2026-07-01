import os
import random
import shutil
from pathlib import Path

def main():
    print("Starting Dataset Split...")
    dataset_dir = Path("/home/rasul-gurbanli/Desktop/skywalker_20/skywalker_20/dataset")
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    # Target directories
    train_img_dir = images_dir / "train"
    val_img_dir = images_dir / "val"
    train_lbl_dir = labels_dir / "train"
    val_lbl_dir = labels_dir / "val"
    
    # Create target directories
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Get all image files (excluding train/val directories themselves)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} image files.")
    
    # Shuffle and split (80% train, 20% val)
    random.seed(42)
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Staging split: {len(train_files)} training, {len(val_files)} validation.")
    
    # Move function with label check
    def move_pair(img_file, target_img_dir, target_lbl_dir):
        # Move image
        shutil.move(str(img_file), str(target_img_dir / img_file.name))
        
        # Check corresponding label file
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.move(str(label_file), str(target_lbl_dir / label_file.name))
            
    # Process moves
    for idx, f in enumerate(train_files):
        if idx % 5000 == 0:
            print(f"Moving train pairs: {idx}/{len(train_files)}")
        move_pair(f, train_img_dir, train_lbl_dir)
        
    for idx, f in enumerate(val_files):
        if idx % 1000 == 0:
            print(f"Moving val pairs: {idx}/{len(val_files)}")
        move_pair(f, val_img_dir, val_lbl_dir)
        
    print("Dataset split complete!")

if __name__ == "__main__":
    main()
