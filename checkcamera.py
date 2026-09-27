import cv2

cap = cv2.VideoCapture(4)   # pamēģini 0, 1 vai 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera test", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()