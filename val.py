## importなどなど
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from ultralytics import YOLO

## 評価
total_confusion_matrix = np.sum(fold_metrics, axis=0)

accuracy = np.trace(total_confusion_matrix) / np.sum(total_confusion_matrix)
precision = np.diag(total_confusion_matrix) / np.sum(total_confusion_matrix, axis=0)
recall = np.diag(total_confusion_matrix) / np.sum(total_confusion_matrix, axis=1)
f1 = 2 * (precision * recall) / (precision + recall)

class_names = ["graupel", "snowflakes", "background"]

plt.figure(figsize=(8, 6))
sns.heatmap(
    total_confusion_matrix,
    annot=True,
    fmt=".0f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Aggregated Confusion Matrix (Accuracy: {accuracy:.4f})")
plt.tight_layout()

output_path = Path("results")
output_path.mkdir(exist_ok=True)
plt.savefig(output_path / "aggregated_confusion_matrix.png", dpi=300)
plt.close()

print(f"Confusion matrix saved to {output_path / 'aggregated_confusion_matrix.png'}")
print(f"Total confusion matrix:\n{total_confusion_matrix}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
