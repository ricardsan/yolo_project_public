#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_robot_test.py
Testa klients virtuālajam RobotStudio kontrolierim (ArtCamera.mod).

Divi režīmi:
  1) --mock   : bez kameras, sūta manuāli ievadītas / iepriekš definētas
                koordinātas. Pārbauda tikai socket ķēdi un RAPID loģiku.
  2) (noklusējums) : reāla D435i + YOLO. Detektē objektu, aprēķina 3D
                punktu (9x9 mediānas dziļums), pielieto T_base_cam un
                nosūta "class,x,y,z" robotam.

Lietošana:
  python yolo_robot_test.py --mock
  python yolo_robot_test.py --model C:\\Users\\jager\\Documents\\dataset700\\best.pt
"""

import argparse
import os
import socket
import sys

ROBOT_HOST = "127.0.0.1"   #127.0.0.1 virtuālais kontrolieris; reālajam: 192.168.125.1
ROBOT_PORT = 5002

# Mock punkti trīs RAPID atbilžu ceļu pārbaudei
MOCK_TESTS = [
    ("0,0.523362338,0.177566765,-0.000906113",   "Robot moved, received class 0"),
    ("0,400,0,70",       "DONE     (derīgs punkts zonas vidū)"),
    ("1,700,0,70",       "REJECTED (X=700 > X_MAX=650)"),
    ("2,400,400,70",     "REJECTED (Y=400 > Y_MAX=350)"),
    ("abc,def",          "PARSE_ERROR (nav skaitļu)"),
    ("1,300",            "PARSE_ERROR (par maz lauku)"),
    ("3,300.5,-120.2,72","DONE     (decimāldaļas)"),
    ("3,300.5,-120.2,72","DONE     (decimāldaļas)"),
    ("-1, 1000,2000,3000","DONE (moved to home even with crazy x,y,z target)"),
    ("-3, 1000,2000,3000","DONE (Closed gripper even with crazy x,y,z target)"),
    ("-2, 1000,2000,3000","DONE (Opened gripper even with crazy x,y,z target)")
]


def send_and_wait(sock: socket.socket, msg: str) -> str:
    """Nosūta ziņojumu un gaida atbildi (DONE bloķējas, kamēr robots brauc)."""
    sock.sendall(msg.encode("ascii"))
    reply = sock.recv(1024).decode("ascii").strip()
    return reply


def run_mock(sock: socket.socket) -> None:
    print("=== MOCK režīms: socket + RAPID loģikas tests ===\n")
    for msg, expected in MOCK_TESTS:
        print(f"TX: {msg!r:28s} gaidāms: {expected}")
        reply = send_and_wait(sock, msg)
        print(f"RX: {reply}\n")
    print("Mock tests pabeigts.")


def run_camera(sock: socket.socket, model_path: str) -> None:
    import numpy as np
    import pyrealsense2 as rs
    from ultralytics import YOLO

    # --- Kalibrācija -----------------------------------------------------
    T_PATH = "T_base_cam.npy"
    if os.path.exists(T_PATH):
        T_base_cam = np.load(T_PATH)
        if np.allclose(T_base_cam, np.eye(4)):
            print("[BRĪDINĀJUMS] T_base_cam ir identitātes matrica (aizstājējs).")
            print("             Koordinātas būs KAMERAS sistēmā, nevis robota bāzē.")
            print("             Virtuālajam socket testam tas ir OK, bet punkti,")
            print("             visticamāk, saņems REJECTED. Reālam darbam vajag")
            print("             atkārtoto roku-acs kalibrāciju.\n")
    else:
        T_base_cam = np.eye(4)
        print(f"[BRĪDINĀJUMS] {T_PATH} nav atrasts, izmantoju identitāti.\n")

    # --- YOLO ------------------------------------------------------------
    model = YOLO(model_path)

    # --- RealSense D435i -------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    print("D435i palaista. Enter = detektēt un sūtīt, 'q' + Enter = iziet.\n")

    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "q":
                break

            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                print("Nav kadra, mēģini vēlreiz.")
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            intr = color_frame.profile.as_video_stream_profile().intrinsics

            results = model(color, verbose=False)[0]
            if len(results.boxes) == 0:
                print("YOLO neko neatrada.")
                continue

            # Ņem detekciju ar augstāko ticamību
            best = max(results.boxes, key=lambda b: float(b.conf))
            cls = int(best.cls)
            name = model.names[cls]
            cx, cy = [int(v) for v in best.xywh[0][:2]]

            # 9x9 mediānas dziļums (izturīgs pret atstarojošām virsmām)
            y0, y1 = max(0, cy - 4), min(depth.shape[0], cy + 5)
            x0, x1 = max(0, cx - 4), min(depth.shape[1], cx + 5)
            window = depth[y0:y1, x0:x1].astype(float) * depth_scale
            window = window[window > 0]
            if window.size == 0:
                print("Nav derīga dziļuma šajā logā.")
                continue
            z_m = float(np.median(window))

            # Pikselis -> kameras 3D (metri)
            X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], z_m)

            # Kameras sistēma -> robota bāze (mm)
            p_cam = np.array([X, Y, Z, 1.0])
            p_base = (T_base_cam @ p_cam)[:3] * 1000.0

            msg = f"{cls},{p_base[0]:.1f},{p_base[1]:.1f},{p_base[2]:.1f}"
            print(f"Detektēts: {name} (cls={cls}, conf={float(best.conf):.2f})")
            print(f"TX: {msg}")
            reply = send_and_wait(sock, msg)
            print(f"RX: {reply}\n")
    finally:
        pipeline.stop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="bez kameras: sūta testa ziņojumus")
    ap.add_argument("--host", default=ROBOT_HOST)
    ap.add_argument("--port", type=int, default=ROBOT_PORT)
    ap.add_argument("--model",
                    default=r"C:\Users\jager\Documents\dataset700\best.pt"
                    if os.name == "nt"
                    else "/home/rica/Documents/dataset700/runs/detect/train8/weights/best.pt")
    args = ap.parse_args()

    print(f"Savienojos ar {args.host}:{args.port} ...")
    try:
        sock = socket.create_connection((args.host, args.port), timeout=10)
    except OSError as e:
        print(f"Neizdevās savienoties: {e}")
        print("Pārbaudi, vai RAPID programma (main) ir palaista virtuālajā kontrolierī.")
        sys.exit(1)
    sock.settimeout(120)  # DONE var prasīt laiku, kamēr robots izbrauc ciklu
    print("Savienots.\n")

    try:
        if args.mock:
            run_mock(sock)
        else:
            run_camera(sock, args.model)
    finally:
        sock.close()


if __name__ == "__main__":
    main()
