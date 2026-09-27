import pyrealsense2 as rs
import numpy as np
import cv2
import socket
from ultralytics import YOLO

ROBOT_HOST = '192.168.1.131'
ROBOT_PORT = 5000

model = YOLO("/home/rica/Documents/dataset700/runs/detect/train8/weights/best.pt")

# === kalibrācija: kad būs T_base_cam.npy, atkomentē ===
# T = np.load("T_base_cam.npy")
T = None


def send_message(msg: str) -> str:
    parts = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect((ROBOT_HOST, ROBOT_PORT))
            sock.sendall(msg.encode('utf-8'))
            while True:
                p = sock.recv(4096)
                if not p:
                    break
                parts.append(p)
    except Exception as e:
        print(f"[ERROR] send_message: {e}")
    return b''.join(parts).decode('utf-8', errors='ignore')


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

profile = pipeline.get_active_profile()
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
intrinsics = color_profile.get_intrinsics()

align = rs.align(rs.stream.color)

last_detection = None   # (cls_name, X, Y, Z) kameras koordinātās

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        results = model(color_image, verbose=False)
        annotated = color_image.copy()

        best_conf = 0
        for box in results[0].boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            distance = depth_frame.get_distance(cx, cy)
            if distance == 0:
                continue  # nav derīga depth

            point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], distance)
            X, Y, Z = point

            cls_name = model.names[int(box.cls[0])]

            # paturam objektu ar augstāko confidence
            if conf > best_conf:
                best_conf = conf
                last_detection = (cls_name, X, Y, Z)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(annotated, f"{cls_name} {conf:.2f} Z={Z:.2f}m",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(annotated, "S = send to robot, ESC = exit", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("YOLO RealSense", annotated)
        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key in (ord('s'), ord('S')) and last_detection:
            cls_name, X, Y, Z = last_detection

            if T is not None:
                p = T @ np.array([X, Y, Z, 1.0])
                x_mm, y_mm, z_mm = p[0] * 1000, p[1] * 1000, p[2] * 1000
            else:
                # PAGAIDĀM bez kalibrācijas - kameras koordinātas mm
                x_mm, y_mm, z_mm = X * 1000, Y * 1000, Z * 1000

            msg = f"{cls_name},{x_mm:.1f},{y_mm:.1f},{z_mm:.1f}"
            print(f"Sūtu: {msg}")
            reply = send_message(msg)
            print(f"Robots: {reply}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()