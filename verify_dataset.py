
import os
from pathlib import Path

def verify_dataset():
    dataset_path = Path("dataset")
    images_train = dataset_path / "images" / "train"
    labels_train = dataset_path / "labels" / "train"
    images_val = dataset_path / "images" / "val"
    labels_val = dataset_path / "labels" / "val"
    
    for img_dir, lbl_dir in [(images_train, labels_train), (images_val, labels_val)]:
        print(f"\nChecking {img_dir.name}:")
        
        img_files = {f.stem for f in img_dir.glob("*.jpg")}
        lbl_files = {f.stem for f in lbl_dir.glob("*.txt")}
        
        print(f"  Total images: {len(img_files)}")
        print(f"  Total labels: {len(lbl_files)}")
        print(f"  Matching pairs: {len(img_files & lbl_files)}")
        print(f"  Images without labels: {len(img_files - lbl_files)}")
        print(f"  Labels without images: {len(lbl_files - img_files)}")
        
        # Check a few label files
        if len(list(lbl_dir.glob("*.txt"))) > 0:
            first_label = list(lbl_dir.glob("*.txt"))[0]
            print(f"\n  Example label ({first_label.name}):")
            with open(first_label, "r") as f:
                print(f.read())

if __name__ == "__main__":
    verify_dataset()
