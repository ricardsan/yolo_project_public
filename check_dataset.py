import cv2
import os

print("Script started")

base = "/home/rica/Documents/dataset700"

# Klases
classes = ["glass", "metal", "paper", "plastic"]

image_dirs = [
    os.path.join(base, "images/train"),
    os.path.join(base, "images/val")
]

label_dirs = [
    os.path.join(base, "labels/train"),
    os.path.join(base, "labels/val")
]

output_dir = os.path.join(base, "checked_images")
os.makedirs(output_dir, exist_ok=True)

count = 0

for img_dir, lbl_dir in zip(image_dirs, label_dirs):

    if not os.path.exists(img_dir):
        print("Missing folder:", img_dir)
        continue

    for file in os.listdir(img_dir):

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(img_dir, file)
        label_path = os.path.join(lbl_dir, file.rsplit(".",1)[0] + ".txt")

        img = cv2.imread(img_path)

        if img is None:
            print("Could not read image:", img_path)
            continue

        h, w, _ = img.shape

        if os.path.exists(label_path):

            with open(label_path) as f:
                lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                cls, x, y, bw, bh = map(float, parts)

                cls = int(cls)

                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                label = classes[cls]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

                cv2.putText(img,
                            label,
                            (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)

        cv2.imwrite(os.path.join(output_dir, file), img)

        count += 1

print("Processed:", count)
print("Saved images in:", output_dir)