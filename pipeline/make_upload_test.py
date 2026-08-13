# -*- coding: utf-8 -*-
"""Write a clean/defective image pair to disk for an end-to-end upload test."""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from engine import white_bg_mask, crop_align, part_regions
from make_defects import GENERATORS

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "data", "sku", "Superstar", "01_0031.jpg")
DST = os.path.join(ROOT, "uploadtest")
os.makedirs(DST, exist_ok=True)

bgr = cv2.imread(SRC)
mask = white_bg_mask(bgr)
bgr, mask, _ = crop_align(bgr, mask)
parts = part_regions(mask)
rng = np.random.default_rng(7)

cv2.imwrite(os.path.join(DST, "clean.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
for name in ("upper_contamination", "excess_cement"):
    for _ in range(8):
        r = GENERATORS[name](bgr, parts, rng)
        if r is not None and r[1].sum() > 60:
            break
    cv2.imwrite(os.path.join(DST, f"{name}.jpg"), r[0], [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(DST, f"{name}_gt.png"), r[1] * 255)
    print(name, "gt px", int(r[1].sum()))
print("written to", DST)
