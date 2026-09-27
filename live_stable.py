#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_stable.py

Stabila reallaika YOLO + Intel RealSense D435i detekcija iepakojuma
skirosanai. Atskiribas no vienkarsa per-frame skripta:

  1. Depth tiek nolasits ka NUMPY masivs vienu reizi kadra
     (nevis 81x depth_frame.get_distance() Python cikla).
  2. Multi-object tracking (ByteTrack) - katram objektam ir ID,
     tapec viena palaista kadra detekcija netiek uzreiz aizmirsta ->
     nav "stuttering" efekta, kad stabilitates buferis atkritas uz nulli.
  3. Klases balsojums pa treka ID - klases nosaukums vairs nelec
     starp klasem katru kadru.
  4. EMA izlidzinasana bounding box un XYZ koordinatem.
  5. Geometriskie filtri: Z diapazons (darba virsma), min/max laukums,
     malu attiecibas parbaude, ROI. Sie filtri izmet skapja/cilveka/
     galda maldigas detekcijas BEZ modela partrenesanas - tas ir
     galvenais ierocis pret "nejausus objektus atzime ka stable".
  6. Interaktivs ROI selektors (taustins 'r') - iezimet darba zonu ar
     peli tiesi uz dzivas ainas, nevis minet koordinates aklu.
  7. Atsevisks inference / depth / kopejais FPS meraparats.

ESC = iziet.  'r' = iezimet jaunu ROI ar peli.  'c' = notirit ROI.
"""

import time
from collections import deque, Counter

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

# =====================================================================
# KONFIGURACIJA
# =====================================================================

MODEL_PATH = r"C:\Users\jager\Documents\dataset700\runs\detect\train_v4\weights\best.pt"

# --- Kameras straume ---
WIDTH, HEIGHT, FPS_CAM = 640, 480, 30

# --- YOLO ---
IMGSZ          = 640      # JABUT TADAM PASAM ka treninos (parbaudi args.yaml!)
CONF_THRESHOLD = 0.55     # augstaks slieksnis = mazak "spoku" detekciju
IOU_THRESHOLD  = 0.5
MAX_DET        = 10
USE_HALF       = True     # FP16 uz CUDA (RTX 3050 -> ~1.5-2x atrak)
TRACKER_CFG    = "bytetrack.yaml"

# --- Geometriskie filtri (galvenais ierocis pret fona kludam) ---
Z_MIN_M        = 0.25     # tuvak par so - ignore
Z_MAX_M        = 1.20     # talak par so - ignore (skapis, siena, cilveks fona)
MIN_BOX_AREA   = 0.004    # dala no kadra laukuma (0.4%)
MAX_BOX_AREA   = 0.45     # dala no kadra laukuma (45%)
MAX_ASPECT     = 5.0      # w/h vai h/w robeza - izmet garas sauras "svitras"

# --- ROI: (x1, y1, x2, y2) kadra dalas [0..1]. None = viss kadrs. ---
# Iezime interaktivi ar 'r' vai uzstadi seit fiksetu darba zonu.
ROI = None

# --- Temporala stabilizacija ---
VOTE_WINDOW    = 9        # kadru skaits klases balsojumam
EMA_ALPHA      = 0.4      # 0..1, mazaks = gludaks, bet lenaks
STABLE_FRAMES  = 10       # cik kadrus treks jabut redzams -> STABLE
Z_TOLERANCE    = 0.015    # 15 mm
XY_TOLERANCE   = 0.020    # 20 mm
MAX_MISSED     = 15       # pec cik kadriem pazudusu treku aizmirst

# --- Depth ---
DEPTH_WINDOW   = 9        # nepara skaitlis
DEPTH_INNER    = 0.6      # meranam tikai iekseja box dala (0.6 = 60%)

# =====================================================================


class Track:
    """Viena izsekota objekta stavoklis laika."""

    __slots__ = ("box", "xyz", "cls_hist", "xyz_hist", "hits", "missed", "conf")

    def __init__(self):
        self.box = None                       # (x1, y1, x2, y2) float, EMA
        self.xyz = None                       # (X, Y, Z) float, EMA
        self.cls_hist = deque(maxlen=VOTE_WINDOW)
        self.xyz_hist = deque(maxlen=STABLE_FRAMES)
        self.hits = 0
        self.missed = 0
        self.conf = 0.0

    def update(self, box, xyz, cls_id, conf):
        box = np.asarray(box, dtype=np.float32)
        xyz = np.asarray(xyz, dtype=np.float32)

        if self.box is None:
            self.box = box
            self.xyz = xyz
        else:
            a = EMA_ALPHA
            self.box = a * box + (1.0 - a) * self.box
            self.xyz = a * xyz + (1.0 - a) * self.xyz

        self.cls_hist.append(cls_id)
        self.xyz_hist.append(tuple(float(v) for v in xyz))
        self.hits += 1
        self.missed = 0
        self.conf = conf

    @property
    def voted_cls(self):
        return Counter(self.cls_hist).most_common(1)[0][0]

    @property
    def vote_ratio(self):
        c = Counter(self.cls_hist)
        return c.most_common(1)[0][1] / len(self.cls_hist)

    def is_stable(self):
        """Stabils = pietiekami ilgi redzams UN 3D pozicija nesvarstas."""
        if len(self.xyz_hist) < STABLE_FRAMES:
            return False
        arr = np.asarray(self.xyz_hist, dtype=np.float32)
        spread = arr.max(axis=0) - arr.min(axis=0)
        return (spread[0] < XY_TOLERANCE and
                spread[1] < XY_TOLERANCE and
                spread[2] < Z_TOLERANCE)

    def median_xyz(self):
        arr = np.asarray(self.xyz_hist, dtype=np.float32)
        return np.median(arr, axis=0)


def median_depth_np(depth_np, depth_scale, x1, y1, x2, y2):
    """
    Mediana dziluma metros no bounding box IEKSEJAS dalas.
    Nulles (nederigi pikseli) tiek ignoretas - kritiski spidigam metalam
    un stiklam. Viss numpy, bez Python cikliem.
    """
    h, w = depth_np.shape

    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * (1.0 - DEPTH_INNER) / 2.0)
    my = int(bh * (1.0 - DEPTH_INNER) / 2.0)

    ix1 = max(0, x1 + mx)
    iy1 = max(0, y1 + my)
    ix2 = min(w, x2 - mx)
    iy2 = min(h, y2 - my)

    if ix2 <= ix1 or iy2 <= iy1:
        # Fallback: mazs logs ap centru
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half = DEPTH_WINDOW // 2
        ix1, iy1 = max(0, cx - half), max(0, cy - half)
        ix2, iy2 = min(w, cx + half + 1), min(h, cy + half + 1)

    patch = depth_np[iy1:iy2, ix1:ix2]
    valid = patch[patch > 0]
    if valid.size < 10:
        return 0.0
    return float(np.median(valid)) * depth_scale


def passes_geometry(x1, y1, x2, y2, frame_w, frame_h, roi):
    """Geometriska pieklajibas parbaude pirms depth nolasisanas."""
    bw, bh = x2 - x1, y2 - y1
    if bw <= 2 or bh <= 2:
        return False

    area = (bw * bh) / float(frame_w * frame_h)
    if area < MIN_BOX_AREA or area > MAX_BOX_AREA:
        return False

    aspect = max(bw / bh, bh / bw)
    if aspect > MAX_ASPECT:
        return False

    if roi is not None:
        cx = (x1 + x2) / 2.0 / frame_w
        cy = (y1 + y2) / 2.0 / frame_h
        rx1, ry1, rx2, ry2 = roi
        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
            return False

    return True


class RoiSelector:
    """Peles vilkto taisnstura ROI iezimesana virsu dzivajam skatam."""

    def __init__(self, window_name):
        self.window_name = window_name
        self.active = False
        self.dragging = False
        self.start = None
        self.end = None
        cv2.setMouseCallback(window_name, self._on_mouse)

    def begin(self):
        self.active = True
        self.dragging = False
        self.start = None
        self.end = None

    def _on_mouse(self, event, x, y, flags, param):
        if not self.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.end = (x, y)
            self.active = False

    def draw_preview(self, img):
        if self.start is not None and self.end is not None:
            cv2.rectangle(img, self.start, self.end, (0, 255, 255), 2)

    def result(self, frame_w, frame_h):
        """Atgriez (x1,y1,x2,y2) [0..1] vai None, ja iezimejums pariak mazs."""
        if self.start is None or self.end is None:
            return None
        x1, y1 = self.start
        x2, y2 = self.end
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None
        return (x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h)


def main():
    global ROI

    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = USE_HALF and device == "cuda"

    print(f"Ierice: {device} (half={half})")
    model = YOLO(MODEL_PATH)
    names = model.names

    # ----- RealSense -----
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS_CAM)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS_CAM)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    # RealSense post-processing - butiski uzlabo depth kvalitati
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    tracks = {}          # {track_id: Track}

    t_fps = deque(maxlen=30)
    t_inf = deque(maxlen=30)
    t_depth = deque(maxlen=30)

    window_name = "YOLO 3D Detection (stable)"
    cv2.namedWindow(window_name)
    roi_selector = RoiSelector(window_name)

    print("ESC = iziet, 'r' = iezimet ROI ar peli, 'c' = notirit ROI")

    try:
        while True:
            t_loop0 = time.perf_counter()

            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            frame = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data())   # uint16, 1 reize
            fh, fw = frame.shape[:2]

            # ---------- YOLO + tracking ----------
            t0 = time.perf_counter()
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CFG,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMGSZ,
                max_det=MAX_DET,
                device=device,
                quantize=16 if half else None,
                verbose=False,
            )[0]
            t_inf.append(time.perf_counter() - t0)

            annotated = frame.copy()
            seen_ids = set()

            t1 = time.perf_counter()

            boxes = results.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().numpy()
                clss = boxes.cls.int().cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for (bx, tid, cls_id, conf) in zip(xyxy, ids, clss, confs):
                    x1, y1, x2, y2 = [int(v) for v in bx]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(fw, x2), min(fh, y2)

                    # 1) geometrija (ieskaitot ROI)
                    if not passes_geometry(x1, y1, x2, y2, fw, fh, ROI):
                        continue

                    # 2) dzilums
                    z = median_depth_np(depth_np, depth_scale, x1, y1, x2, y2)
                    if z <= 0 or z < Z_MIN_M or z > Z_MAX_M:
                        continue

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(
                        intrinsics, [float(cx), float(cy)], z
                    )

                    tid = int(tid)
                    seen_ids.add(tid)
                    if tid not in tracks:
                        tracks[tid] = Track()
                    tracks[tid].update((x1, y1, x2, y2), (X, Y, Z),
                                       int(cls_id), float(conf))

            t_depth.append(time.perf_counter() - t1)

            # ---------- Pazudusie treki ----------
            for tid in list(tracks.keys()):
                if tid not in seen_ids:
                    tracks[tid].missed += 1
                    if tracks[tid].missed > MAX_MISSED:
                        del tracks[tid]

            # ---------- Zimesana ----------
            if ROI is not None:
                rx1, ry1, rx2, ry2 = ROI
                cv2.rectangle(annotated,
                              (int(rx1 * fw), int(ry1 * fh)),
                              (int(rx2 * fw), int(ry2 * fh)),
                              (255, 255, 0), 1)

            for tid, tr in tracks.items():
                if tr.missed > 0 or tr.box is None or tr.hits < 3:
                    continue

                x1, y1, x2, y2 = [int(v) for v in tr.box]
                X, Y, Z = [float(v) for v in tr.xyz]
                label = names[tr.voted_cls]
                stable = tr.is_stable()

                color = (0, 255, 0) if stable else (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated, ((x1 + x2) // 2, (y1 + y2) // 2),
                           4, (0, 0, 255), -1)

                status = "STABLE" if stable else f"{len(tr.xyz_hist)}/{STABLE_FRAMES}"
                cv2.putText(annotated,
                            f"#{tid} {label} {tr.conf:.2f} v={tr.vote_ratio:.2f} [{status}]",
                            (x1, max(15, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(annotated,
                            f"X={X*1000:6.1f} Y={Y*1000:6.1f} Z={Z*1000:6.1f} mm",
                            (x1, min(fh - 6, y2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                if stable:
                    Xm, Ym, Zm = tr.median_xyz()
                    print(f"[STABLE] id={tid} {label} "
                          f"vote={tr.vote_ratio:.2f} "
                          f"X={Xm*1000:.1f} Y={Ym*1000:.1f} Z={Zm*1000:.1f} mm")
                    # -> seit velak: T_base_cam transformacija + socket uz RAPID
                    # (skat. robot/yolo_robot_test.py - protokols un T_base_cam
                    #  pielietosana jau strada tur, jaintegre seit)

            # ---------- ROI iezimesanas rezims ----------
            if roi_selector.dragging or roi_selector.active:
                roi_selector.draw_preview(annotated)
            if not roi_selector.active and roi_selector.start is not None:
                new_roi = roi_selector.result(fw, fh)
                if new_roi is not None:
                    ROI = new_roi
                    print(f"[ROI] iestatits: {ROI}")
                roi_selector.start = None
                roi_selector.end = None

            t_fps.append(time.perf_counter() - t_loop0)
            fps = 1.0 / max(1e-6, float(np.mean(t_fps)))
            inf_ms = float(np.mean(t_inf)) * 1000.0
            dep_ms = float(np.mean(t_depth)) * 1000.0

            cv2.putText(annotated,
                        f"{fps:5.1f} FPS | infer {inf_ms:4.1f} ms | depth {dep_ms:4.1f} ms | trk {len(tracks)}",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord("r"):
                roi_selector.begin()
            if key == ord("c"):
                ROI = None
                print("[ROI] notirits")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
