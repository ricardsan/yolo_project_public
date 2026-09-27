import pyrealsense2 as rs
import numpy as np
import cv2

SQUARE = 0.060
MARKER = 0.043

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((6, 7), SQUARE, MARKER, dictionary)
board.setLegacyPattern(True)
detector = cv2.aruco.CharucoDetector(board)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

intr = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()
K = np.array([[intr.fx, 0, intr.ppx],
              [0, intr.fy, intr.ppy],
              [0, 0, 1]])
dist = np.array(intr.coeffs)

print("SPACE = saglabat, ESC = iziet")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        ch_corners, ch_ids, mk_corners, mk_ids = detector.detectBoard(gray)

        vis = color.copy()
        pose_ok = False

        if ch_ids is not None and len(ch_ids) >= 6:
            # zimejam sturus pasi (OpenCV 5 saderiba)
            for i in range(len(ch_ids)):
                c = ch_corners[i].ravel()
                cv2.circle(vis, (int(c[0]), int(c[1])), 4, (0, 0, 255), -1)

            obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)

            if obj_pts is not None and len(obj_pts) >= 6:
                obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
                img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)

                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)

                if ok:
                    pose_ok = True
                    cv2.drawFrameAxes(vis, K, dist, rvec, tvec, 0.1)
                    cv2.putText(vis, f"corners: {len(ch_ids)}  POSE OK  (SPACE = save)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "Board not detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("calib", vis)
        key = cv2.waitKey(1)

        if key == 32 and pose_ok:  # SPACE
            R, _ = cv2.Rodrigues(rvec)
            T_cam_board = np.eye(4)
            T_cam_board[:3, :3] = R
            T_cam_board[:3, 3] = tvec.flatten()
            np.save("T_cam_board.npy", T_cam_board)
            print("Saved T_cam_board.npy")
            print(T_cam_board)
            break
        elif key == 27:  # ESC
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()