#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_split_yaml.py — sadala Label Studio YOLO eksportu train/val un izveido data.yaml.

Palaisana (no jebkuras mapes):
    python make_split_yaml.py C:\\Users\\jager\\Documents\\dataset700\\dataset_seg
    python make_split_yaml.py <mape> --val 0.2 --seed 0

Gaida:  <mape>/images/*.jpg  <mape>/labels/*.txt  <mape>/classes.txt
Izveido: images/train, images/val, labels/train, labels/val, data.yaml
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    img_dir, lab_dir = root / "images", root / "labels"
    if not img_dir.is_dir() or not lab_dir.is_dir():
        print(f"Nav atrastas mapes images/ un labels/ zem {root}")
        return 1

    if (img_dir / "train").exists():
        print("images/train jau eksiste — sadalijums jau izdarits, tikai rakstu data.yaml")
        pairs = None
    else:
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        pairs, no_label = [], 0
        for im in imgs:
            lb = lab_dir / (im.stem + ".txt")
            if lb.exists() and lb.stat().st_size > 0:
                pairs.append((im, lb))
            else:
                no_label += 1
        if not pairs:
            print("Neviens attels nesakrit ar etiketi (vienadi nosaukumi bez paplasinajuma?)")
            return 1

        # parbaude: poligoni vai kastes?
        first = pairs[0][1].read_text(encoding="utf-8").split("\n")[0].split()
        if len(first) == 5:
            print("BRIDINAJUMS: etiketes rinda ir 5 skaitli -> tas ir BBOX, nevis poligoni!")
            print("Segmentacijai eksporte JSON un lieto ls2yoloseg.py.")
            return 1
        print(f"Etiketes formats OK: {len(first)} vertibas rinda (poligoni)")

        random.seed(args.seed)
        random.shuffle(pairs)
        n_val = int(len(pairs) * args.val)
        for split in ("train", "val"):
            (img_dir / split).mkdir(exist_ok=True)
            (lab_dir / split).mkdir(exist_ok=True)
        for k, (im, lb) in enumerate(pairs):
            split = "val" if k < n_val else "train"
            shutil.move(str(im), str(img_dir / split / im.name))
            shutil.move(str(lb), str(lab_dir / split / lb.name))
        print(f"Sadalits: train {len(pairs) - n_val} / val {n_val}"
              + (f"  (bez etiketes izlaisti: {no_label})" if no_label else ""))

    cls_file = root / "classes.txt"
    if cls_file.exists():
        names = [l.strip() for l in cls_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        names = ["glass", "metal", "paper", "plastic"]
        print("classes.txt nav — lietoju glass/metal/paper/plastic (PARBAUDI SECIBU!)")

    yaml = (f"path: {root.resolve().as_posix()}\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: {len(names)}\n"
            f"names: {names}\n")
    (root / "data.yaml").write_text(yaml, encoding="utf-8")
    print("\n" + yaml)
    print(f"Gatavs: {root / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
