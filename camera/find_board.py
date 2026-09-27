import pyrealsense2 as rs
import numpy as np
import cv2

# === nomaini pec dela: majas A4 = 0.040, laba A2 = 0.060 ===
SQUARE = 0.040
MARKER = SQUARE * 0.75

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

print("Turi deli kadra... panemsu kadru pec 3 sek")
for _ in range(90):
    frames = pipeline.wait_for_frames()
color = np.asanyarray(frames.get_color_frame().get_data())
pipeline.stop()

gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
cv2.imwrite("frame_used.png", color)
print("Kadrs saglabats: frame_used.png")

dicts = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250,
    "4X4_1000": cv2.aruco.DICT_4X4_1000,
}

best = None
for dname, dval in dicts.items():
    dictionary = cv2.aruco.getPredefinedDictionary(dval)
    ad = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = ad.detectMarkers(gray)
    n_m = 0 if ids is None else len(ids)
    if n_m == 0:
        print(f"[{dname}] markers: 0")
        continue
    print(f"\n[{dname}] markers: {n_m}, ids: {sorted(ids.flatten().tolist())}")

    for cols in range(4, 14):
        for rows in range(4, 14):
            for legacy in (True, False):
                try:
                    board = cv2.aruco.CharucoBoard((cols, rows), SQUARE, MARKER, dictionary)
                    board.setLegacyPattern(legacy)
                    det = cv2.aruco.CharucoDetector(board)
                    ch_c, ch_i, _, _ = det.detectBoard(gray)
                    n = 0 if ch_i is None else len(ch_i)
                    if n > 0:
                        print(f"  cols={cols} rows={rows} legacy={legacy} -> {n} corners")
                        if best is None or n > best[4]:
                            best = (dname, cols, rows, legacy, n)
                except Exception:
                    pass

print()
if best:
    print(f"=== BEST: dict={best[0]}, (cols,rows)=({best[1]},{best[2]}), "
          f"legacy={best[3]}, corners={best[4]} ===")
else:
    print("=== ChArUco sturi nekur neatradas ===")

# ugenere delus salidzinasanai ar izprinteto
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for legacy in (True, False):
    b = cv2.aruco.CharucoBoard((6, 5), SQUARE, MARKER, dictionary)
    b.setLegacyPattern(legacy)
    img = b.generateImage((1200, 1000), marginSize=40)
    cv2.imwrite(f"generated_6x5_legacy_{legacy}.png", img)
print("Saglabati ari: generated_6x5_legacy_True.png / False.png")