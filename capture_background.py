import cv2
import os

save_path = "images/train"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 60

print("Press SPACE to capture image")
print("Press ESC to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        filename = f"{save_path}/bg_{count:03d}.jpg"
        cv2.imwrite(filename, frame)
        print("Saved:", filename)
        count += 1

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()