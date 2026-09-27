from pathlib import Path
from PIL import Image

# Folder ar bildēm
IMG_DIR = Path("/home/rica/Documents/dataset700/dataset_v3/images")

count = 0

# Meklē .jpeg failus
for img_path in IMG_DIR.rglob("*"):
    if img_path.suffix.lower() == ".jpeg":

        new_path = img_path.with_suffix(".jpg")

        try:
            img = Image.open(img_path)
            img.convert("RGB").save(new_path, "JPEG", quality=95)

            img_path.unlink()  # delete old .jpeg

            count += 1
            print("Converted:", img_path.name)

        except Exception as e:
            print("Error:", img_path.name, e)

print("DONE!")
print("Total converted:", count)