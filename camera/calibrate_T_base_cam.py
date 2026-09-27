"""
Camera -> Robot kalibrācija ar ChArUco dēli (viss vienā skriptā).

1. Dēlis guļ darba zonā, NEKUSTINĀT visu procedūras laiku.
2. Kamera nosaka dēļa pozu, ekrānā parāda P0/P1/P2 punktus.
3. Ar robotu (jogging) aizved TCP uz tiem punktiem, ievadi koordinātas (mm).
4. Skripts izrēķina T_base_cam, parāda kļūdu, saglabā .npy.
"""

import pyrealsense2 as rs
import numpy as np
import cv2

# ====================================
# ChArUco dēļa parametri (A2 dēlis, apstiprināti ar find_board.py)
# ====================================

SQUARES_X = 6          # lauciņu skaits pa horizontāli
SQUARES_Y = 7          # lauciņu skaits pa vertikāli
SQUARE_LENGTH = 0.060  # lauciņa izmērs metros (60 mm)
MARKER_LENGTH = 0.043  # markera izmērs metros (43 mm)
ARUCO_DICT = cv2.aruco.DICT_4X4_50
LEGACY_PATTERN = True

OUTPUT_FILE = "T_base_cam.npy"

# ====================================
# RealSense
# ====================================

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

color_profile = profile.get_stream(rs.stream.color)
intr = color_profile.as_video_stream_profile().get_intrinsics()

camera_matrix = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array(intr.coeffs, dtype=np.float64)

# ====================================
# ChArUco dēlis
# ====================================

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)
board.setLegacyPattern(LEGACY_PATTERN)

charuco_detector = cv2.aruco.CharucoDetector(board)

# Atsauces punkti dēļa koordinātās (metros) - iekšējie šaha stūri
cols = SQUARES_X - 1   # 5
rows = SQUARES_Y - 1   # 6

ref_points_board = np.array([
    [1 * SQUARE_LENGTH,    1 * SQUARE_LENGTH,    0.0],  # P0
    [cols * SQUARE_LENGTH, 1 * SQUARE_LENGTH,    0.0],  # P1: +240mm pa X no P0
    [1 * SQUARE_LENGTH,    rows * SQUARE_LENGTH, 0.0],  # P2: +300mm pa Y no P0
], dtype=np.float64)


def transform_points(T, pts):
    """Pielieto 4x4 transformāciju Nx3 punktiem."""
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (T @ pts_h.T).T[:, :3]


def kabsch(A, B):
    """Atrod rigid transformāciju T (4x4), lai T @ A ~= B."""
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ====================================
# 1. SOLIS: dēļa poza no kameras
# ====================================

print("=" * 50)
print("1. SOLIS: dēļa pozas noteikšana")
print("Novieto dēli darba zonā. NEKUSTINI to pēc šī brīža!")
print("SPACE - fiksēt pozu | ESC - iziet")
print("=" * 50)

T_cam_board = None

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            charuco_detector.detectBoard(gray)

        display = frame.copy()
        pose_ok = False
        rvec, tvec = None, None

        if charuco_ids is not None and len(charuco_ids) >= 6:
            # sturu zimesana manuali (OpenCV 5 saderiba)
            for i in range(len(charuco_ids)):
                c = charuco_corners[i].ravel()
                cv2.circle(display, (int(c[0]), int(c[1])), 4, (0, 255, 0), -1)

            obj_pts, img_pts = board.matchImagePoints(
                charuco_corners, charuco_ids)

            if obj_pts is not None and len(obj_pts) >= 6:
                obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
                img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)

                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, camera_matrix, dist_coeffs)
                if ok:
                    pose_ok = True
                    cv2.drawFrameAxes(display, camera_matrix, dist_coeffs,
                                      rvec, tvec, 0.1)

                    # Uzzīmē atsauces punktus P0, P1, P2
                    proj, _ = cv2.projectPoints(
                        ref_points_board, rvec, tvec,
                        camera_matrix, dist_coeffs)
                    for i, p in enumerate(proj.reshape(-1, 2)):
                        pt = (int(p[0]), int(p[1]))
                        cv2.circle(display, pt, 8, (0, 0, 255), -1)
                        cv2.putText(display, f"P{i}",
                                    (pt[0] + 10, pt[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 0, 255), 2)

        status = "POZA OK - spied SPACE" if pose_ok else "Mekle deli..."
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if pose_ok else (0, 0, 255), 2)

        cv2.imshow("Kalibracija", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Partraukts.")
            raise SystemExit

        if key == 32 and pose_ok:
            R, _ = cv2.Rodrigues(rvec)
            T_cam_board = np.eye(4)
            T_cam_board[:3, :3] = R
            T_cam_board[:3, 3] = tvec.flatten()
            print("Dela poza fikseta.")
            print("Dela origin kameras rami (m):", T_cam_board[:3, 3])
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# Atsauces punkti KAMERAS rāmī (metros)
ref_points_cam = transform_points(T_cam_board, ref_points_board)

print()
print("Atsauces punkti kameras rami (mm):")
for i, p in enumerate(ref_points_cam):
    print(f"  P{i}: X={p[0]*1000:.1f}  Y={p[1]*1000:.1f}  Z={p[2]*1000:.1f}")

# ====================================
# 2. SOLIS: robota koordinātas
# ====================================

print()
print("=" * 50)
print("2. SOLIS: robota koordinatas")
print("Ar jogging aizved TCP smaili PRECIZI uz katru punktu")
print("(sarkanie punkti ekrana: P0, P1, P2).")
print("Nolasi robota poziciju (mm, Base frame) un ievadi seit.")
print("=" * 50)

ref_points_robot = np.zeros((3, 3))

for i in range(3):
    while True:
        try:
            raw = input(f"P{i} robota koordinatas 'x,y,z' mm: ")
            x, y, z = [float(v.strip()) for v in raw.split(",")]
            ref_points_robot[i] = [x / 1000.0, y / 1000.0, z / 1000.0]
            break
        except ValueError:
            print("Nepareizs formats. Piemers: 412.5, -87.3, 12.0")

# Geometrijas parbaude pirms aprekina
d01 = np.linalg.norm(ref_points_robot[1] - ref_points_robot[0]) * 1000
d02 = np.linalg.norm(ref_points_robot[2] - ref_points_robot[0]) * 1000
print()
print(f"Parbaude: P0-P1 attalums = {d01:.1f} mm (jabut ~240)")
print(f"          P0-P2 attalums = {d02:.1f} mm (jabut ~300)")
if abs(d01 - 240) > 15 or abs(d02 - 300) > 15:
    print("BRIDINAJUMS: attalumi neatbilst delim - punkti var but sajaukti!")

# ====================================
# 3. SOLIS: T_base_cam aprēķins
# ====================================

T_base_cam = kabsch(ref_points_cam, ref_points_robot)

check = transform_points(T_base_cam, ref_points_cam)
errors = np.linalg.norm(check - ref_points_robot, axis=1) * 1000  # mm

print()
print("=" * 50)
print("REZULTATS")
print("=" * 50)
print("T_base_cam:")
print(T_base_cam)
print()
print("Atbilstibas kluda katra punkta:")
for i, e in enumerate(errors):
    print(f"  P{i}: {e:.2f} mm")
print(f"Videja kluda: {errors.mean():.2f} mm")

if errors.mean() > 5.0:
    print()
    print("BRIDINAJUMS: kluda > 5 mm. Iespejamie iemesli:")
    print("  - TCP nebija precizi uz stura")
    print("  - delis pakustejas starp 1. un 2. soli")
    print("  - nepareizi dela parametri (SQUARE_LENGTH?)")
    print("Iesaku atkartot.")

np.save(OUTPUT_FILE, T_base_cam)
print()
print(f"Saglabats: {OUTPUT_FILE}")
print()
print("Lietosana yolo_robot.py:")
print("  T = np.load('T_base_cam.npy')")
print("  p_cam = np.array([X, Y, Z, 1.0])   # metros!")
print("  p_robot = (T @ p_cam)[:3] * 1000   # -> mm robotam")