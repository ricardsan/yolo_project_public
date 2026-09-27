#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment_backgrounds.py

Paplasina esoso YOLO datu kopu (images/ + labels/, viss uz balta/pelekaina
fona) ar sintetiskiem variantiem uz dazadiem foniem un apgaismojumiem, lai
modelis nepieradistu tikai pie studijas fona.

NEAIZTIEK oriģinālos failus - lasa no images/+labels/ un raksta jaunu
neatkarīgu kopiju uz dataset_v4/.

Pieeja: fona AIZVIETOSANA, nevis pilna copy-paste parkomponēšana.
  1. Katram bbox no YOLO labela ar GrabCut (inicializēts no bbox
     taisnstūra) izdala objekta masku - strādā labi, jo fons attēlā ir
     vienlaidus.
  2. Objektu maska tiek atstāta savā vietā (bbox koordinātas NEMAINĀS,
     tāpēc labeļus var vienkārši nokopēt), bet fons ap to tiek aizstāts
     ar citu fonu (foto no --backgrounds mapes, ja tāda dota, citādi
     ģenerēti sintētiski foni).
  3. Kompozītam attēlam pievienots neliels spilgtuma/kontrasta/trokšņa
     variants, lai simulētu atšķirīgu apgaismojumu - tas parasti ietekmē
     precizitāti vairāk nekā pats fona krāsu tonis.
  4. Oriģinālais (balta fona) attēls arī tiek nokopēts uz dataset_v4, lai
     saglabātu reālo domēnu maisījumā ar sintētiskajiem variantiem.

Lietošana:
  python augment_backgrounds.py
  python augment_backgrounds.py --backgrounds backgrounds --variants 4
  python augment_backgrounds.py --variants 3 --seed 42 --out dataset_v4
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

if sys.platform == "win32":
    # Windows konsoles noklusētais cp1252 kodējums nespēj izvadīt latviešu
    # diakritiskās zīmes -- pārslēdz stdout/stderr uz UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

CLASSES = ["glass", "metal", "paper", "plastic"]


def read_yolo_labels(label_path, img_w, img_h):
    """Return list of (cls, x1, y1, x2, y2) in pixel coords."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, bw, bh = parts
        cls = int(float(cls))
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
        x1 = max(0, int((xc - bw / 2) * img_w))
        y1 = max(0, int((yc - bh / 2) * img_h))
        x2 = min(img_w, int((xc + bw / 2) * img_w))
        y2 = min(img_h, int((yc + bh / 2) * img_h))
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def segment_objects(img, boxes, pad=12, max_dim=900):
    """GrabCut per bbox, seeded from the box rectangle. Returns a single
    0..255 uint8 alpha mask covering all objects in the image.

    GrabCut cost scales with pixel count, and the source photos are
    full-res phone photos (~4000px) -- running it at native resolution
    takes tens of seconds per image. Downscaling to max_dim before
    GrabCut and upscaling the resulting mask afterwards keeps quality
    (edges get smoothed anyway) while cutting runtime by ~15-20x."""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        work = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        work_boxes = [(c, int(x1 * scale), int(y1 * scale),
                       int(x2 * scale), int(y2 * scale))
                      for c, x1, y1, x2, y2 in boxes]
    else:
        work = img
        work_boxes = boxes
    wh, ww = work.shape[:2]

    work_mask = np.zeros((wh, ww), np.uint8)
    work_pad = max(2, int(pad * scale))

    for _, x1, y1, x2, y2 in work_boxes:
        rx1 = max(0, x1 - work_pad)
        ry1 = max(0, y1 - work_pad)
        rx2 = min(ww, x2 + work_pad)
        ry2 = min(wh, y2 + work_pad)
        rect = (rx1, ry1, rx2 - rx1, ry2 - ry1)
        if rect[2] < 4 or rect[3] < 4:
            continue

        gc_mask = np.zeros((wh, ww), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(work, gc_mask, rect, bgd_model, fgd_model, 5,
                        cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            # Deģenerēts reģions (piem. viendabīga zona) - krīti atpakaļ
            # uz pilnu bbox kā masku.
            work_mask[ry1:ry2, rx1:rx2] = 255
            continue

        obj = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                        255, 0).astype(np.uint8)
        work_mask = cv2.bitwise_or(work_mask, obj)

    # Notīra troksni, izgludina malas mazā izšķirtspējā (lēti).
    work_mask = cv2.morphologyEx(work_mask, cv2.MORPH_OPEN,
                                  np.ones((5, 5), np.uint8))
    work_mask = cv2.morphologyEx(work_mask, cv2.MORPH_CLOSE,
                                  np.ones((9, 9), np.uint8))

    if scale < 1.0:
        full_mask = cv2.resize(work_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        full_mask = work_mask
    full_mask = cv2.GaussianBlur(full_mask, (9, 9), 0)
    return full_mask


def load_backgrounds(bg_dir, size_hint):
    bgs = []
    if bg_dir is not None:
        bg_dir = Path(bg_dir)
        if bg_dir.exists():
            for p in sorted(bg_dir.glob("*")):
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                im = cv2.imread(str(p))
                if im is not None:
                    bgs.append(im)
    if not bgs:
        print("[INFO] Nav foto fonu -- izmantoju generetus sintetiskos fonus.")
    return bgs


def synthetic_background(h, w, rng):
    """Vienkāršs krāsains/tekstūrēts fons, ja foto fonu nav (galds var
    būt koks, tumšāks plastmasas galds, u.tml.)."""
    base_color = np.array([
        rng.randint(60, 230),
        rng.randint(60, 230),
        rng.randint(60, 230),
    ], dtype=np.float64)
    bg = np.ones((h, w, 3), np.float64) * base_color

    # Viegls gradients + troksnis, lai fons neizskatītos plakans.
    grad = np.linspace(-25, 25, w)[None, :, None]
    bg = bg + grad
    noise = rng.normal(0, 6, (h, w, 1))
    bg = bg + noise
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=3)
    return bg


def fit_background(bg, h, w, rng):
    bh, bw = bg.shape[:2]
    scale = max(h / bh, w / bw) * rng.uniform(1.0, 1.15)
    bg = cv2.resize(bg, (int(bw * scale) + 1, int(bh * scale) + 1))
    bh, bw = bg.shape[:2]
    x0 = rng.randint(0, max(1, bw - w + 1))
    y0 = rng.randint(0, max(1, bh - h + 1))
    return bg[y0:y0 + h, x0:x0 + w].copy()


def relight(img, rng):
    img = img.astype(np.float32)
    alpha = rng.uniform(0.8, 1.2)   # kontrasts
    beta = rng.uniform(-25, 25)     # spilgtums
    img = img * alpha + beta
    noise = rng.normal(0, rng.uniform(1, 6), img.shape)
    img = img + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def composite(img, mask, bg):
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    out = img.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def process_split(split, images_dir, labels_dir, out_images_dir,
                   out_labels_dir, backgrounds, variants, rng, max_out_dim):
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in images_dir.glob("*")
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    print(f"[{split}] {len(img_files)} avota attēli")

    n_written = 0
    for i, img_path in enumerate(img_files, 1):
        label_path = labels_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            print("  ! Nevar nolasit:", img_path)
            continue

        # Avota foto ir ~4000px telefona attēli -- kompozīcija/relight uz
        # pilnas izšķirtspējas ir ļoti lēna un YOLO treniņam nevajadzīga
        # (imgsz parasti 640-1280). Samazina reizi uzreiz; YOLO labeļi ir
        # normalizēti, tāpēc koordinātas nemainās.
        h0, w0 = img.shape[:2]
        scale_out = min(1.0, max_out_dim / max(h0, w0))
        if scale_out < 1.0:
            img = cv2.resize(img, (int(w0 * scale_out), int(h0 * scale_out)),
                              interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        boxes = read_yolo_labels(label_path, w, h)

        # 1) Oriģināls (balta fona) attēls -- saglabā reālo domēnu.
        cv2.imwrite(str(out_images_dir / img_path.name), img)
        if label_path.exists():
            (out_labels_dir / label_path.name).write_text(
                label_path.read_text())
        n_written += 1

        if not boxes:
            continue  # nav ko segmentēt -- tikai oriģināls tiek pievienots

        mask = segment_objects(img, boxes)

        # 2) Sintētiskie fona varianti.
        for v in range(variants):
            if backgrounds:
                bg = backgrounds[rng.randint(0, len(backgrounds) - 1)]
                bg = fit_background(bg, h, w, rng)
            else:
                bg = synthetic_background(h, w, rng)

            out_img = composite(img, mask, bg)
            out_img = relight(out_img, rng)

            out_name = f"{img_path.stem}_bg{v}{img_path.suffix}"
            cv2.imwrite(str(out_images_dir / out_name), out_img)
            if label_path.exists():
                (out_labels_dir / f"{img_path.stem}_bg{v}.txt").write_text(
                    label_path.read_text())
            n_written += 1

        if i % 50 == 0:
            print(f"  ... {i}/{len(img_files)}")

    print(f"[{split}] uzrakstīti {n_written} attēli -> {out_images_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=".",
                     help="dataset700 saknes mape (noklusējums: pašreizējā)")
    ap.add_argument("--out", default="dataset_v4",
                     help="izvades apakšmape base iekšpusē")
    ap.add_argument("--backgrounds", default="backgrounds",
                     help="mape ar fotogrāfētiem fona attēliem (nav "
                          "obligāta -- ja nav, izmanto ģenerētus fonus)")
    ap.add_argument("--variants", type=int, default=3,
                     help="cik sintētiskus fona variantus ģenerēt uz katru "
                          "avota attēlu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-out-dim", type=int, default=1600,
                     help="izejas attēlu maksimālā mala pikseļos (avota "
                          "foto ir ~4000px -- samazināšana pirms "
                          "kompozīcijas paātrina apstrādi ~10x un "
                          "YOLO treniņam pilna izšķirtspēja nav vajadzīga)")
    args = ap.parse_args()

    base = Path(args.base)
    out_base = base / args.out
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    class NpCompatRng:
        """random.Random ērtībai izmanto arī np gadījuma skaitļus."""
        def randint(self, a, b):
            return rng.randint(a, b)

        def uniform(self, a, b):
            return rng.uniform(a, b)

        def normal(self, mean, std, size):
            return np_rng.normal(mean, std, size)

    r = NpCompatRng()

    backgrounds = load_backgrounds(args.backgrounds, None)
    print(f"Fona attēlu skaits: {len(backgrounds)}")

    for split in ("train", "val"):
        process_split(
            split,
            base / "images" / split,
            base / "labels" / split,
            out_base / "images" / split,
            out_base / "labels" / split,
            backgrounds,
            args.variants,
            r,
            args.max_out_dim,
        )

    yaml_path = base / f"data_{args.out.split('_')[-1]}.yaml"
    yaml_path.write_text(
        f"path: {out_base.resolve()}\n\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: glass\n"
        "  1: metal\n"
        "  2: paper\n"
        "  3: plastic\n"
    )
    print(f"\nGatavs. Konfigurācija: {yaml_path}")
    print("Ieteikums: pirms treniņa palaid check_dataset.py (ar mainītiem "
          "ceļiem) uz dataset_v4, lai vizuāli pārbaudītu masku/kompozīcijas "
          "kvalitāti dažiem attēliem.")


if __name__ == "__main__":
    main()
