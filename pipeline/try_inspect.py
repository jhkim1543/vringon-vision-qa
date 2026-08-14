# -*- coding: utf-8 -*-
"""Smoke-run the inspection items on a few real lateral photos."""
import os, sys, glob, json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from engine import white_bg_mask, crop_align, part_regions
from inspect_items import inspect

ROOT = os.path.join(os.path.dirname(__file__), "..")
SKU = os.path.join(ROOT, "data", "sku")

files = []
for sku in ("Superstar", "Stan-Smith", "Gazelle"):
    files += sorted(glob.glob(os.path.join(SKU, sku, "*.jpg")))[:2]

for f in files:
    bgr = cv2.imread(f)
    mask = white_bg_mask(bgr)
    bgr, mask, _ = crop_align(bgr, mask)
    parts = part_regions(mask)
    recs, summary = inspect(bgr, mask, parts, view="lateral")
    print(f"\n=== {os.path.relpath(f, SKU)}  {summary}")
    for r in recs:
        v = "—" if r["measured"] is None else f'{r["measured"]:>8.2f}'
        u = "" if not r.get("uncertainty") else f' ±{r["uncertainty"]}'
        print(f'  {r["no"]:>2}. {r["name_en"]:<28} {r["status"]:<11} {v}{u} '
              f'{r["units"]}  {r["note"][:46]}')
