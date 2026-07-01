# Error & Failure Case Analysis

## 1. Convergence & Loss Analysis
Based on the `results.csv` and graphical loss curves (`results.png`):
- **Box Loss**: Shows healthy initial descent.
- **Class Loss**: Relatively high, indicating that the network needs more extensive training (epochs) to effectively separate the custom classes.
- **DFL Loss**: Dropping steadily, indicating bounding box boundaries are becoming tighter around the targets.

## 2. Confusion Matrix Insights
Analyzing `confusion_matrix.png` and `confusion_matrix_normalized.png`:
- **False Positives (Background detected as Object)**: Currently prevalent due to the limited 1-epoch dry-run. The network is hallucinating objects in high-texture background areas.
- **False Negatives (Missed Detections)**: Recall is artificially high (1.0) in the dummy test, but in real-world scenarios, occlusion and blur will introduce false negatives.

## 3. Precision-Recall Tradeoffs
The `BoxPR_curve.png` dictates the thresholding strategy:
- A high confidence threshold (e.g., 0.6) will maximize **Precision** (minimizing false alarms) but hurt **Recall** (missing faint objects).
- A lower threshold (e.g., 0.25) maximizes **Recall**, critical for safety or security applications where missing an object is worse than a false alarm.

## 4. Mitigation Strategies
To resolve false detections in full-scale training:
1. **Background Images**: Include images with absolutely no targets (`empty` images) to suppress background false positives.
2. **Hard Negative Mining**: Re-train the model specifically on crops where it previously made high-confidence mistakes.
3. **Data Augmentation**: Increase Mosaic and MixUp probabilities if overfitting to specific backgrounds is observed.
