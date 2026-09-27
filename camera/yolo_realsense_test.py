import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# load your trained model
model = YOLO(r"C:\Users\jager\Documents\dataset700\runs\detect\train_v4\weights\best.pt")

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

profile = pipeline.get_active_profile()
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
intrinsics = color_profile.get_intrinsics()

align = rs.align(rs.stream.color)

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

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            distance = depth_frame.get_distance(cx, cy)

            point = rs.rs2_deproject_pixel_to_point(
                intrinsics,
                [cx, cy],
                distance
            )

            X = point[0]
            Y = point[1]
            Z = point[2]

            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            print(f"{cls_name}: X={X:.3f} Y={Y:.3f} Z={Z:.3f}")

            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(annotated, (cx,cy), 5, (0,0,255), -1)

            cv2.putText(
                annotated,
                f"{cls_name} Z={Z:.2f}m",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        cv2.imshow("YOLO RealSense", annotated)

        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()