import os
import shutil
import random
from pathlib import Path
import yaml

# ===== PATHS (TAVI CEĻI) =====

dataset_path = Path("/home/rica/Documents/dataset700/dataset_v2")

images_path = dataset_path / "images_all"
labels_path = dataset_path / "labels_all"

output_path = Path("/home/rica/Documents/dataset700/dataset_5fold")

yaml_file_path = Path("/home/rica/Documents/dataset700/data_v2.yaml")

num_folds = 5

random.seed(42)

# ==============================

# ===== LOAD CLASS NAMES FROM YAML =====

with open(yaml_file_path, "r") as f:
    data_yaml = yaml.safe_load(f)

class_names = data_yaml["names"]

print("Classes:", class_names)

# ===== LOAD IMAGES =====

images = sorted(list(images_path.glob("*.jpg")))

print(f"Total images found: {len(images)}")

if len(images) == 0:
    print("ERROR: No images found!")
    exit()

# ===== SHUFFLE =====

random.shuffle(images)

fold_size = len(images) // num_folds

folds = []

for i in range(num_folds):

    start = i * fold_size

    if i < num_folds - 1:
        end = (i + 1) * fold_size
    else:
        end = len(images)

    folds.append(images[start:end])

print("Folds prepared.")

# ===== CREATE FOLDS =====

for i in range(num_folds):

    fold_name = f"fold{i+1}"

    fold_dir = output_path / fold_name

    train_img_dir = fold_dir / "images/train"
    val_img_dir = fold_dir / "images/val"

    train_lbl_dir = fold_dir / "labels/train"
    val_lbl_dir = fold_dir / "labels/val"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)

    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    val_images = folds[i]

    train_images = []

    for j in range(num_folds):

        if j != i:
            train_images.extend(folds[j])

    print(f"\nCreating {fold_name}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")

    # ===== VALIDATION =====

    for img_path in val_images:

        label_path = labels_path / (img_path.stem + ".txt")

        shutil.copy(img_path, val_img_dir / img_path.name)

        if label_path.exists():
            shutil.copy(label_path, val_lbl_dir / label_path.name)

    # ===== TRAIN =====

    for img_path in train_images:

        label_path = labels_path / (img_path.stem + ".txt")

        shutil.copy(img_path, train_img_dir / img_path.name)

        if label_path.exists():
            shutil.copy(label_path, train_lbl_dir / label_path.name)

    # ===== CREATE YAML =====

    yaml_path = fold_dir / "data.yaml"

    with open(yaml_path, "w") as f:

        f.write(f"path: {fold_dir}\n")

        f.write("train: images/train\n")
        f.write("val: images/val\n")

        f.write(f"nc: {len(class_names)}\n")

        f.write("names:\n")

        for name in class_names:
            f.write(f"  - {name}\n")

print("\nDONE! All 5 folds created successfully.")