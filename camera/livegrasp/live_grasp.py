#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_grasp.py — datorredzes programma satveršanas plānošanai.

Ko dara:
  1. Nolasa RealSense D435i krāsu + dziļuma kadru (izlīdzinātu pret krāsu straumi).
  2. Palaiž YOLO detekciju (atbalsta gan -seg, gan parastu detekcijas modeli).
  3. Katram objektam aprēķina virsmas pikseļus, PCA galvenās asis,
     garumu / platumu (mm), rotācijas leņķi un 3D TCP punktu robota bāzē.
  4. Novērtē, kuru objektu var satvert (RG6 atvērums, izolētība, augstums kaudzē).
  5. Uz ekrāna zīmē visus objektus + izceļ LABĀKO satveramo.
  6. Izvada mērķi robot_test programmai (TCP + leņķis + klase).

Palaišana:
    python live_grasp.py                  # tikai vizualizācija
    python live_grasp.py --send           # sūta labāko mērķi uz robotu (5004)
    python live_grasp.py --bag file.bag   # atskaņo ierakstītu .bag failu

Taustiņi logā: q = iziet, s = saglabāt kadru, SPACE = izvadīt mērķi konsolē.
"""

import argparse
import json
import math
import socket
import sys
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("Trūkst pyrealsense2. Instalē: pip install pyrealsense2")
    raise

try:
    from ultralytics import YOLO
except ImportError:
    print("Trūkst ultralytics. Instalē: pip install ultralytics")
    raise


# =====================================================================
# KONFIGURĀCIJA — pielāgo savai videi
# =====================================================================

WEIGHTS = r"runs/detect/train_v4/weights/best.pt"
T_BASE_CAM_PATH = "T_base_cam.npy"          # 4x4 homogēnā matrica, mm

CONF_TH = 0.50
IOU_TH = 0.45
CLASS_NAMES = ["cup", "jar", "bottle", "can"]

# --- RG6 satvērēja parametri (mm) ---
GRIP_OPEN_MAX = 160.0     # maksimālais atvērums
GRIP_OPEN_MIN = 10.0      # minimālais praktiskais objekta platums
GRIP_CLEARANCE = 12.0     # rezerve katrā pusē, lai pirksti neuzduras
GRIP_IDEAL = 60.0         # platums, pie kura satveršana ir visdrošākā

# --- Darba zona robota bāzes koordinātēs (mm) ---
WS_X = (150.0, 700.0)
WS_Y = (-450.0, 450.0)
WS_Z = (-20.0, 400.0)

# --- Dziļuma apstrāde ---
DEPTH_WIN = 9             # 9x9 mediānas logs (viena pikseļa nolasījums neder!)
DEPTH_TOL_MM = 30.0       # cik biezu "slāni" ap mediānu uzskata par objektu
MIN_MASK_PX = 400         # mazākas maskas ignorē

# --- Leņķa pārrēķins kamera -> robots ---
# Attēla y ass rāda uz leju, tāpēc leņķis jāapgriež.
# YAW_OFFSET_DEG kompensē kameras pagriezienu pret bāzes X asi — nomēri!
YAW_OFFSET_DEG = 0.0
YAW_LIMITS_DEG = (-180.0, 180.0)

# --- Vērtēšanas svari ---
W_TOP = 0.35              # cik augstu kaudzē
W_ISO = 0.25              # cik izolēts no kaimiņiem
W_FIT = 0.25              # cik labi platums atbilst satvērējam
W_CONF = 0.15             # YOLO pārliecība

# --- Tīkls ---
ROBOT_IP = "192.168.125.1"
ROBOT_PORT = 5004


# =====================================================================
# 1. MATEMĀTIKA — virsmas pikseļi, PCA, leņķis, izmēri
# =====================================================================

def pca_from_mask(mask):
    """
    Aprēķina objekta orientāciju no binārās maskas ar 2. kārtas centrālajiem
    momentiem (PCA uz pikseļu koordinātēm).

    Formulas:
        N  = pikseļu skaits
        x̄  = (1/N) Σ xi              ȳ = (1/N) Σ yi
        μ20 = (1/N) Σ (xi - x̄)²
        μ02 = (1/N) Σ (yi - ȳ)²
        μ11 = (1/N) Σ (xi - x̄)(yi - ȳ)

        θ = ½ · atan2( 2·μ11 , μ20 - μ02 )      <- galvenās ass leņķis

        λ1,2 = (μ20 + μ02)/2 ± ½·sqrt( (μ20 - μ02)² + 4·μ11² )

    Garumu un platumu NEŅEM no λ (tas ir tikai izkliedes rādītājs) —
    pikseļus projicē uz abām asīm un ņem reālo (max - min) izplētumu.

    Atgriež dict ar: cx, cy, theta_rad, length_px, width_px, e1, e2, lam1, lam2
    """
    ys, xs = np.nonzero(mask)
    n = xs.size
    if n < MIN_MASK_PX:
        return None

    cx = xs.mean()
    cy = ys.mean()
    dx = xs - cx
    dy = ys - cy

    mu20 = float((dx * dx).mean())
    mu02 = float((dy * dy).mean())
    mu11 = float((dx * dy).mean())

    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)

    tmp = math.sqrt((mu20 - mu02) ** 2 + 4.0 * mu11 ** 2)
    lam1 = 0.5 * (mu20 + mu02) + 0.5 * tmp      # lielākā
    lam2 = 0.5 * (mu20 + mu02) - 0.5 * tmp      # mazākā

    e1 = np.array([math.cos(theta), math.sin(theta)])   # garuma ass
    e2 = np.array([-math.sin(theta), math.cos(theta)])  # platuma ass

    pts = np.stack([dx, dy], axis=1)
    proj1 = pts @ e1
    proj2 = pts @ e2
    length_px = float(proj1.max() - proj1.min())
    width_px = float(proj2.max() - proj2.min())

    # ja PCA sajauc asis (gandrīz kvadrātisks objekts) — sakārto
    if width_px > length_px:
        length_px, width_px = width_px, length_px
        e1, e2 = e2, e1
        theta += math.pi / 2.0

    return {
        "cx": float(cx), "cy": float(cy),
        "theta_rad": float(theta),
        "length_px": length_px, "width_px": width_px,
        "e1": e1, "e2": e2,
        "lam1": float(lam1), "lam2": float(lam2),
        "n_px": int(n),
    }


def gripper_yaw_deg(theta_rad):
    """
    Satvērēja pirkstiem jāaizveras PĀRI īsākajai asij, tātad satvērēja
    plakne ir perpendikulāra objekta garuma asij:

        θ_grip(attēlā) = θ + 90°

    Attēla y ass rāda uz leju -> robota Rz = -θ_grip + nobīde.
    """
    theta_img_deg = math.degrees(theta_rad) + 90.0
    yaw = -theta_img_deg + YAW_OFFSET_DEG
    # normalizē uz [-180, 180] un uz simetrisko pusi (satvērējs ir simetrisks)
    while yaw > 90.0:
        yaw -= 180.0
    while yaw < -90.0:
        yaw += 180.0
    return float(yaw)


def px_to_mm(px, depth_mm, fx):
    """Mērogs: mm_uz_pikseli = Z / fx  (perspektīvās projekcijas apgrieztā formula)."""
    return float(px * depth_mm / fx)


# =====================================================================
# 2. DZIĻUMS
# =====================================================================

def median_depth_mm(depth_img_mm, u, v, win=DEPTH_WIN):
    """Mediāna win x win logā, ignorējot nulles (spīdīgas virsmas dod 0)."""
    h, w = depth_img_mm.shape
    r = win // 2
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    patch = depth_img_mm[v0:v1, u0:u1]
    vals = patch[patch > 0]
    if vals.size == 0:
        return 0.0
    return float(np.median(vals))


def mask_from_box_depth(depth_img_mm, box):
    """
    Ja modelis nav -seg, masku veido no dziļuma:
    ņem mediānu kastes centrā un patur pikseļus tuvāk par DEPTH_TOL_MM.
    Tas vienlaikus atdala objektu no fona UN no zemākiem objektiem kaudzē.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = depth_img_mm.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = depth_img_mm[y1:y2, x1:x2]
    valid = roi[roi > 0]
    if valid.size < MIN_MASK_PX // 4:
        return None

    # objekta virsma = tuvākais slānis, tāpēc ņem 25. procentili, ne mediānu
    z_top = float(np.percentile(valid, 25))
    sel = (roi > 0) & (np.abs(roi - z_top) < DEPTH_TOL_MM)

    sel = sel.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    sel = cv2.morphologyEx(sel, cv2.MORPH_OPEN, kernel)
    sel = cv2.morphologyEx(sel, cv2.MORPH_CLOSE, kernel)

    # patur tikai lielāko savienoto komponenti (atdala saskarošos objektus)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sel, connectivity=8)
    if num <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sel = (labels == largest).astype(np.uint8)

    full = np.zeros(depth_img_mm.shape, np.uint8)
    full[y1:y2, x1:x2] = sel
    return full


# =====================================================================
# 3. TRANSFORMĀCIJA KAMERA -> ROBOTA BĀZE
# =====================================================================

def load_T_base_cam(path):
    try:
        T = np.load(path)
        assert T.shape == (4, 4)
        return T
    except Exception as e:
        print(f"BRĪDINĀJUMS: neizdevās ielādēt {path} ({e}). Lietoju vienības matricu.")
        return np.eye(4)


def pixel_to_base(intr, T, u, v, depth_mm):
    """Pikselis + dziļums -> XYZ kameras rāmī (mm) -> XYZ bāzes rāmī (mm)."""
    p_cam_m = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], depth_mm / 1000.0)
    p_cam = np.array([p_cam_m[0] * 1000.0, p_cam_m[1] * 1000.0, p_cam_m[2] * 1000.0, 1.0])
    p_base = T @ p_cam
    return p_base[:3]


# =====================================================================
# 4. SATVERAMĪBAS VĒRTĒJUMS
# =====================================================================

def fit_score(width_mm):
    """1.0 pie ideālā platuma, 0.0 ārpus satvērēja diapazona."""
    usable_max = GRIP_OPEN_MAX - 2 * GRIP_CLEARANCE
    if width_mm < GRIP_OPEN_MIN or width_mm > usable_max:
        return 0.0
    span = max(GRIP_IDEAL - GRIP_OPEN_MIN, usable_max - GRIP_IDEAL)
    return float(max(0.0, 1.0 - abs(width_mm - GRIP_IDEAL) / span))


def isolation_score(mask, others, dilate_px=15):
    """Cik brīva ir josla ap objektu — vai pirksti neuzdursies kaimiņam."""
    if not others:
        return 1.0
    k = np.ones((dilate_px, dilate_px), np.uint8)
    ring = cv2.dilate(mask, k) - mask
    ring_area = float(ring.sum())
    if ring_area < 1:
        return 1.0
    overlap = 0.0
    for om in others:
        overlap += float(np.logical_and(ring > 0, om > 0).sum())
    return float(max(0.0, 1.0 - overlap / ring_area))


def evaluate(objs):
    """Piešķir katram objektam score un atgriež sarakstu, sakārtotu dilstoši."""
    if not objs:
        return []

    zs = [o["depth_mm"] for o in objs]
    z_min, z_max = min(zs), max(zs)
    z_span = max(1.0, z_max - z_min)

    for o in objs:
        # tuvāk kamerai = augstāk kaudzē = labāk
        o["s_top"] = float((z_max - o["depth_mm"]) / z_span)
        o["s_fit"] = fit_score(o["width_mm"])
        o["s_conf"] = float(o["conf"])
        others = [x["mask"] for x in objs if x is not o]
        o["s_iso"] = isolation_score(o["mask"], others)

        reasons = []
        if o["s_fit"] == 0.0:
            reasons.append(f"platums {o['width_mm']:.0f}mm arpus RG6")
        x, y, z = o["p_base"]
        if not (WS_X[0] <= x <= WS_X[1] and WS_Y[0] <= y <= WS_Y[1] and WS_Z[0] <= z <= WS_Z[1]):
            reasons.append("arpus darba zonas")
        if o["depth_mm"] <= 0:
            reasons.append("nav dzilums")

        o["reject"] = reasons
        o["score"] = 0.0 if reasons else (
            W_TOP * o["s_top"] + W_ISO * o["s_iso"] +
            W_FIT * o["s_fit"] + W_CONF * o["s_conf"]
        )

    return sorted(objs, key=lambda o: o["score"], reverse=True)


# =====================================================================
# 5. ZĪMĒŠANA
# =====================================================================

def draw_object(img, o, best=False):
    color = (0, 255, 0) if best else (0, 165, 255)
    thick = 3 if best else 1

    cx, cy = o["cx"], o["cy"]
    e1, e2 = o["e1"], o["e2"]
    L, W = o["length_px"] / 2.0, o["width_px"] / 2.0

    corners = np.array([
        [cx + e1[0] * L + e2[0] * W, cy + e1[1] * L + e2[1] * W],
        [cx + e1[0] * L - e2[0] * W, cy + e1[1] * L - e2[1] * W],
        [cx - e1[0] * L - e2[0] * W, cy - e1[1] * L - e2[1] * W],
        [cx - e1[0] * L + e2[0] * W, cy - e1[1] * L + e2[1] * W],
    ], dtype=np.int32)
    cv2.polylines(img, [corners], True, color, thick)

    # garuma ass (zila) un satvērēja aizvēršanās ass (sarkana)
    p = np.array([cx, cy])
    a1 = (p + e1 * L).astype(int); b1 = (p - e1 * L).astype(int)
    a2 = (p + e2 * (W + 18)).astype(int); b2 = (p - e2 * (W + 18)).astype(int)
    cv2.line(img, tuple(b1), tuple(a1), (255, 120, 0), 1)
    cv2.arrowedLine(img, tuple(a2), tuple(b2), (0, 0, 255), 2, tipLength=0.25)
    cv2.arrowedLine(img, tuple(b2), tuple(a2), (0, 0, 255), 2, tipLength=0.25)
    cv2.circle(img, (int(cx), int(cy)), 4, color, -1)

    lines = [
        f"{o['cls_name']} {o['conf']:.2f}",
        f"L={o['length_mm']:.0f} W={o['width_mm']:.0f} mm",
        f"th={math.degrees(o['theta_rad']):+.1f} yaw={o['yaw_deg']:+.1f}",
        f"Z={o['depth_mm']:.0f}mm  s={o['score']:.2f}",
    ]
    if o["reject"]:
        lines.append("X " + "; ".join(o["reject"]))

    y0 = int(cy - W - 8)
    for i, t in enumerate(lines):
        y = y0 + i * 15
        cv2.putText(img, t, (int(cx) - 60, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (int(cx) - 60, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)


def draw_panel(img, best, n_total):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 74), (30, 30, 30), -1)
    if best is None or best["score"] <= 0.0:
        cv2.putText(img, "NAV SATVERAMA OBJEKTA", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, f"atrasti: {n_total}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return
    x, y, z = best["p_base"]
    cv2.putText(img, f"LABAKAIS: {best['cls_name']}  score {best['score']:.2f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img,
                f"TCP base: X={x:.1f} Y={y:.1f} Z={z:.1f} mm   Rz={best['yaw_deg']:+.1f} deg   "
                f"atverums={best['width_mm'] + 2 * GRIP_CLEARANCE:.0f} mm",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# =====================================================================
# 6. IZVADE ROBOTAM
# =====================================================================

def target_message(o):
    """
    Formāts jāsaskaņo ar ParseMessage procedūru ArtCamera.mod!
    Pašlaik: "PICK,x,y,z,rz,klase,atverums;"
    """
    x, y, z = o["p_base"]
    return (f"PICK,{x:.2f},{y:.2f},{z:.2f},{o['yaw_deg']:.2f},"
            f"{o['cls_id']},{o['width_mm'] + 2 * GRIP_CLEARANCE:.1f};")


def target_dict(o):
    x, y, z = o["p_base"]
    return {
        "class": o["cls_name"], "class_id": o["cls_id"], "conf": round(o["conf"], 3),
        "x_mm": round(float(x), 2), "y_mm": round(float(y), 2), "z_mm": round(float(z), 2),
        "rz_deg": round(o["yaw_deg"], 2),
        "length_mm": round(o["length_mm"], 1), "width_mm": round(o["width_mm"], 1),
        "grip_open_mm": round(o["width_mm"] + 2 * GRIP_CLEARANCE, 1),
        "score": round(o["score"], 3),
    }


def send_target(msg, ip=ROBOT_IP, port=ROBOT_PORT, timeout=3.0):
    try:
        with socket.create_connection((ip, port), timeout=timeout) as s:
            s.sendall(msg.encode("ascii"))
            s.settimeout(timeout)
            try:
                return s.recv(256).decode("ascii", "ignore")
            except socket.timeout:
                return "(nav atbildes)"
    except Exception as e:
        return f"KLUDA: {e}"


# =====================================================================
# 7. GALVENAIS CIKLS
# =====================================================================

def build_pipeline(bag=None):
    pipe = rs.pipeline()
    cfg = rs.config()
    if bag:
        cfg.enable_device_from_file(bag, repeat_playback=True)
    else:
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    return pipe, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--bag", default=None)
    ap.add_argument("--send", action="store_true", help="sutit labako merki robotam")
    ap.add_argument("--conf", type=float, default=CONF_TH)
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = model.names if hasattr(model, "names") else {i: n for i, n in enumerate(CLASS_NAMES)}
    T = load_T_base_cam(T_BASE_CAM_PATH)

    pipe, depth_scale = build_pipeline(args.bag)
    align = rs.align(rs.stream.color)          # obligati! citadi dzilums nesakrit
    print(f"Depth scale: {depth_scale}  ->  1 vieniba = {depth_scale * 1000:.3f} mm")

    last_sent = 0.0
    try:
        while True:
            frames = align.process(pipe.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth_mm = depth_raw * depth_scale * 1000.0

            intr = color_frame.profile.as_video_stream_profile().intrinsics
            fx = intr.fx

            res = model.predict(color, conf=args.conf, iou=IOU_TH, verbose=False)[0]

            objs = []
            has_masks = getattr(res, "masks", None) is not None
            boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
            clss = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else []
            confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else []

            if has_masks:
                md = res.masks.data.cpu().numpy()
            
            for i in range(len(boxes)):
                if has_masks:
                    m = cv2.resize(md[i], (color.shape[1], color.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                    mask = (m > 0.5).astype(np.uint8)
                    mask[depth_mm <= 0] = 0
                else:
                    mask = mask_from_box_depth(depth_mm, boxes[i])
                if mask is None or mask.sum() < MIN_MASK_PX:
                    continue

                geo = pca_from_mask(mask)
                if geo is None:
                    continue

                u, v = int(round(geo["cx"])), int(round(geo["cy"]))
                z = median_depth_mm(depth_mm, u, v)
                if z <= 0:
                    ys, xs = np.nonzero(mask)
                    vals = depth_mm[ys, xs]
                    vals = vals[vals > 0]
                    z = float(np.median(vals)) if vals.size else 0.0
                if z <= 0:
                    continue

                geo.update({
                    "cls_id": int(clss[i]),
                    "cls_name": names.get(int(clss[i]), str(clss[i])),
                    "conf": float(confs[i]),
                    "mask": mask,
                    "depth_mm": z,
                    "length_mm": px_to_mm(geo["length_px"], z, fx),
                    "width_mm": px_to_mm(geo["width_px"], z, fx),
                    "yaw_deg": gripper_yaw_deg(geo["theta_rad"]),
                    "p_base": pixel_to_base(intr, T, u, v, z),
                })
                objs.append(geo)

            ranked = evaluate(objs)
            best = ranked[0] if ranked and ranked[0]["score"] > 0 else None

            vis = color.copy()
            for o in ranked:
                draw_object(vis, o, best=(o is best))
            draw_panel(vis, best, len(ranked))

            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_mm, alpha=0.045), cv2.COLORMAP_JET)
            cv2.imshow("live_grasp", vis)
            cv2.imshow("depth", depth_vis)

            if args.send and best is not None and time.time() - last_sent > 2.0:
                msg = target_message(best)
                print(">>", msg, "->", send_target(msg))
                last_sent = time.time()

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"grasp_{ts}.png", vis)
                print(f"Saglabats grasp_{ts}.png")
            if k == 32 and best is not None:
                print(json.dumps(target_dict(best), indent=2, ensure_ascii=False))
                print(target_message(best))
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
