import pyrealsense2 as rs
import numpy as np
import cv2

# ==========================
# RealSense konfigurācija
# ==========================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# Izlīdzina depth uz RGB koordinātu sistēmu
align = rs.align(rs.stream.color)

print("ESC - iziet")
print("S - saglabāt Overlay attēlu")

try:

    while True:

        frames = pipeline.wait_for_frames()

        # Izlīdzina depth pret RGB
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # ==========================
        # Punkts attēla centrā
        # ==========================

        x = color.shape[1] // 2
        y = color.shape[0] // 2

        distance = depth_frame.get_distance(x, y)

        # RGB
        rgb = color.copy()

        cv2.circle(rgb, (x, y), 6, (0, 0, 255), -1)

        cv2.putText(
            rgb,
            f"Distance: {distance:.3f} m",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        # ==========================
        # Depth Colormap
        # ==========================

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.03),
            cv2.COLORMAP_JET
        )

        cv2.circle(depth_colormap, (x, y), 6, (0, 0, 255), -1)

        # ==========================
        # Overlay (Pierādījums)
        # ==========================

        overlay = cv2.addWeighted(
            rgb,
            0.70,
            depth_colormap,
            0.30,
            0
        )

        cv2.putText(
            overlay,
            "RGB + Depth Overlay",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # ==========================
        # Kopējais logs
        # ==========================

        both = np.hstack((rgb, depth_colormap))

        cv2.imshow("RGB", rgb)
        cv2.imshow("Depth", depth_colormap)
        cv2.imshow("Overlay", overlay)
        cv2.imshow("RGB | Depth", both)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):

            cv2.imwrite("rgb.png", rgb)
            cv2.imwrite("depth.png", depth_colormap)
            cv2.imwrite("overlay.png", overlay)

            print("Saglabāti:")
            print("rgb.png")
            print("depth.png")
            print("overlay.png")

        elif key == 27:
            break

finally:

    pipeline.stop()
    cv2.destroyAllWindows()