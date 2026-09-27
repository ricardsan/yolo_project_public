from pathlib import Path
import random
import shutil

ROOT = Path("/home/rica/Documents/dataset700")
IMAGES = ROOT / "images"
LABELS = ROOT / "labels"

TRAIN_RATIO = 0.8

# Create folders
for subset in ["train", "val"]:
    (IMAGES / subset).mkdir(exist_ok=True)
    (LABELS / subset).mkdir(exist_ok=True)

# Collect all images
all_images = [p for p in IMAGES.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

print(f"Found {len(all_images)} images")

random.shuffle(all_images)
split_idx = int(len(all_images) * TRAIN_RATIO)

train_imgs = all_images[:split_idx]
val_imgs = all_images[split_idx:]

def move_pair(image_path, subset):
    # Move image
    shutil.move(str(image_path), str(IMAGES / subset / image_path.name))

    # Corresponding label
    label_path = LABELS / (image_path.stem + ".txt")
    if label_path.exists():
        shutil.move(str(label_path), str(LABELS / subset / label_path.name))
    else:
        # create empty label for YOLO
        (LABELS / subset / (image_path.stem + ".txt")).write_text("")

for img in train_imgs:
    move_pair(img, "train")

for img in val_imgs:
    move_pair(img, "val")

print("Done!")
