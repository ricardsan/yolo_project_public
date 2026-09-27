from pathlib import Path
import random
import shutil

# IMPORTANT — pointing to dataset_v3
ROOT = Path("/home/rica/Documents/dataset700/dataset_v3")

IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"

TRAIN_RATIO = 0.8

# create folders
for sub in ["train", "val"]:
    (IMG_DIR / sub).mkdir(parents=True, exist_ok=True)
    (LBL_DIR / sub).mkdir(parents=True, exist_ok=True)

# collect images
images = [p for p in IMG_DIR.iterdir() 
          if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

print("Found images:", len(images))

random.shuffle(images)

split_idx = int(len(images) * TRAIN_RATIO)

train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def move_pair(img_path, subset):

    # Move image
    shutil.move(
        str(img_path),
        str(IMG_DIR / subset / img_path.name)
    )

    # matching label
    label_path = LBL_DIR / (img_path.stem + ".txt")
    out_label = LBL_DIR / subset / (img_path.stem + ".txt")

    if label_path.exists():
        shutil.move(str(label_path), str(out_label))
    else:
        out_label.write_text("")

for img in train_imgs:
    move_pair(img, "train")

for img in val_imgs:
    move_pair(img, "val")

print("DONE!")
print("Train:", len(train_imgs))
print("Val:", len(val_imgs))