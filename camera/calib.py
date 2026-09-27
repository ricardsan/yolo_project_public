import pyrealsense2 as rs
import numpy as np
import cv2
print(cv2.__version__)
SQUARE = 0.6
MARKER = 0.43

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Divi board varianti - legacy un jaunais
board_legacy = cv2.aruco.CharucoBoard((6, 5), SQUARE, MARKER, dictionary)
board_legacy.setLegacyPattern(True)

board_new = cv2.aruco.CharucoBoard((6, 5), SQUARE, MARKER, dictionary)

det_legacy = cv2.aruco.CharucoDetector(board_legacy)
det_new = cv2.aruco.CharucoDetector(board_new)

aruco_detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        vis = color.copy()

        # 1) tikai ArUco markeri
        corners, ids, _ = aruco_detector.detectMarkers(gray)
        n_markers = 0 if ids is None else len(ids)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        # 2) ChArUco - legacy
        ch_c1, ch_i1, _, _ = det_legacy.detectBoard(gray)
        n_legacy = 0 if ch_i1 is None else len(ch_i1)

        # 3) ChArUco - jaunais
        ch_c2, ch_i2, _, _ = det_new.detectBoard(gray)
        n_new = 0 if ch_i2 is None else len(ch_i2)

        cv2.putText(vis, f"markers: {n_markers}  legacy: {n_legacy}  new: {n_new}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("debug", vis)
        if cv2.waitKey(1) == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()