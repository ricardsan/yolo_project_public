from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2

# ====================================
# YOLO modelis
# ====================================

model = YOLO("runs/detect/train_v4/weights/best.pt")
# ====================================
# RealSense
# ====================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

align = rs.align(rs.stream.color)

# Kamera intrinsics
color_profile = profile.get_stream(rs.stream.color)
intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

print("ESC - iziet")

try:

    while True:

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # ==========================
        # YOLO
        # ==========================

        results = model(frame, conf=0.6, verbose=False)

        annotated = frame.copy()

        for box in results[0].boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Bounding box centrs
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Depth
            depth = depth_frame.get_distance(cx, cy)

            if depth <= 0:
                continue

            # 3D koordinātas
            X, Y, Z = rs.rs2_deproject_pixel_to_point(
                intrinsics,
                [cx, cy],
                depth
            )

            # Uzzīmē bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            cv2.circle(
                annotated,
                (cx, cy),
                5,
                (0,0,255),
                -1
            )

            label = model.names[cls]

            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

            cv2.putText(
                annotated,
                f"Z={Z:.3f}m",
                (x1, y2+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,255),
                2
            )

            cv2.putText(
                annotated,
                f"X={X:.3f}",
                (x1, y2+40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,0,0),
                2
            )

            cv2.putText(
                annotated,
                f"Y={Y:.3f}",
                (x1, y2+60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,0,0),
                2
            )

            print("--------------------------------")
            print(label)
            print(f"Confidence : {conf:.2f}")
            print(f"Pixel      : ({cx}, {cy})")
            print(f"Depth      : {depth:.3f} m")
            print(f"X = {X:.3f}")
            print(f"Y = {Y:.3f}")
            print(f"Z = {Z:.3f}")

        cv2.imshow("YOLO 3D Detection", annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:

    pipeline.stop()
    cv2.destroyAllWindows()