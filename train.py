## importなどなど
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from ultralytics import YOLO


## 学習データ作成
def save_paths_to_txt(file_path, image_paths):
    with open(file_path, "w") as f:
        for p in image_paths:
            f.write(f"{p}\n")
    return


def prepare_fold_data(fold_data):
    """データ準備のみを行う（並列化可能）"""
    fold_idx, train_images, val_images, test_images = fold_data

    fold_dir = Path(f"coco_dataset/fold_{fold_idx}")
    fold_dir.mkdir(exist_ok=True, parents=True)

    train_txt = fold_dir / "train.txt"
    val_txt = fold_dir / "val.txt"
    test_txt = fold_dir / "test.txt"

    save_paths_to_txt(train_txt, train_images)
    save_paths_to_txt(val_txt, val_images)
    save_paths_to_txt(test_txt, test_images)

    return fold_idx, fold_dir

## 学習
def train_fold(fold_idx, fold_dir):
    """1つのFoldの学習と評価を実行（GPU使用のため逐次実行）"""
    model = YOLO("yolo11n.yaml").load("yolo11n.pt")
    model.train(
        data="/content/snow-crystal-detector/coco.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        workers=2,
        # --- 精度向上のためのパラメータ ---
        optimizer='AdamW',
        cos_lr=True,          # 学習率を滑らかに下げて後半の微調整を丁寧に行う
        label_smoothing=0.1,  # クラス間の境界を曖昧にして汎化性能を向上

        # --- 検出力不足(Background対策) ---
        overlap_mask=True,    # 重なりに強くする
        val=True,             # 各エポックで検証を行い、最良の重みを保存

        # --- データ拡張(あられ/雪片のバリエーション対応) ---
        degrees=180.0,        # 回転不変性を学習
        scale=0.5,            # 小さいあられのサイズ変化に対応
        fliplr=0.5,           # 左右反転
        mosaic=1.0,           # 複数の結晶が写る状況を模倣
        plots=True,
        name=f"example_fold_{fold_idx + 1}",
    )

    model.val(split="val")
    results = model.val(split="test")

    return results.confusion_matrix.matrix


base_path = Path("coco_dataset/images/")
extensions = {".png"}
image_paths = [
    str(p) for p in base_path.glob("*")
    if p.suffix.lower() in extensions
]

labels = []
for path_str in image_paths:
    file_name = Path(path_str).name.lower()
    if "graupel" in file_name:
        labels.append("graupel")
    elif "snowflakes" in file_name:
        labels.append("snowflakes")
    else:
        labels.append("unknown")

np_image_paths = np.array(image_paths)
np_labels = np.array(labels)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_data_list = []
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np_image_paths, np_labels)):
    train_images = [np_image_paths[i] for i in train_idx]
    train_labels = np_labels[train_idx]
    test_images = [np_image_paths[i] for i in test_idx]

    train_images, val_images, _, _ = train_test_split(
        train_images, train_labels, test_size=0.25, random_state=42
    )

    fold_data_list.append((fold_idx, train_images, val_images, test_images))

with ThreadPoolExecutor(max_workers=5) as executor:
    prepared_folds = list(executor.map(prepare_fold_data, fold_data_list))

fold_metrics = []
for fold_idx, fold_dir in prepared_folds:
    print(f"Training Fold {fold_idx + 1}/5...")
    confusion_matrix = train_fold(fold_idx, fold_dir)
    fold_metrics.append(confusion_matrix)
