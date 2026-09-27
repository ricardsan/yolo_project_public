import os
import random
import shutil

image_dir = "dataset_v2/images_all"
label_dir = "dataset_v2/labels_all"
output_dir = "dataset_v2"

train_ratio = 0.8

os.makedirs(f"{output_dir}/images/train", exist_ok=True)
os.makedirs(f"{output_dir}/images/val", exist_ok=True)
os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
os.makedirs(f"{output_dir}/labels/val", exist_ok=True)

images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
random.shuffle(images)

split = int(len(images) * train_ratio)

train = images[:split]
val = images[split:]

def copy(imgs, split_name):
    for img in imgs:
        label = img.replace(".jpg", ".txt")

        shutil.copy(f"{image_dir}/{img}", f"{output_dir}/images/{split_name}/{img}")
        shutil.copy(f"{label_dir}/{label}", f"{output_dir}/labels/{split_name}/{label}")

copy(train, "train")
copy(val, "val")

print("DONE ✅")