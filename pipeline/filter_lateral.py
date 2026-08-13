# -*- coding: utf-8 -*-
"""Filter sneaker parquet images to clean white-background lateral views.

Heuristic: border pixels near-white, landscape-ish shoe bbox, single connected
foreground blob whose aspect (w/h) is in the lateral-view range.
"""
import os, io, sys
import numpy as np
import cv2
import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(ROOT, "sku")

MODELS = ["Superstar", "Stan Smith", "Gazelle"]

def white_bg_mask(bgr):
    """Return foreground mask if background is near-white, else None."""
    h, w = bgr.shape[:2]
    border = np.concatenate([bgr[0, :], bgr[-1, :], bgr[:, 0], bgr[:, -1]], axis=0)
    if border.mean() < 225 or border.std() > 26:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    fg = ((gray < 232) | (sat > 28)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return None
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (lab == big).astype(np.uint8)
    area = stats[big, cv2.CC_STAT_AREA]
    if area < 0.12 * h * w or area > 0.85 * h * w:
        return None
    return mask

def is_lateral(mask):
    ys, xs = np.where(mask > 0)
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    ar = bw / bh
    if not (1.55 <= ar <= 3.2):
        return False
    # lateral single-shoe: fill ratio of bbox moderate (shoe profile ~0.5-0.75)
    fill = mask.sum() / (bw * bh)
    return 0.42 <= fill <= 0.82

def main():
    os.makedirs(OUT, exist_ok=True)
    kept = {m: 0 for m in MODELS}
    for b in range(1, 4):
        p = os.path.join(ROOT, "sneakers", f"dataset_batch_{b:02d}.parquet")
        if not os.path.exists(p):
            from huggingface_hub import hf_hub_download
            hf_hub_download("ipogorelov/sneakers", f"dataset_batch_{b:02d}.parquet",
                            repo_type="dataset", local_dir=os.path.join(ROOT, "sneakers"))
        t = pq.read_table(p).to_pandas()
        for i, row in t.iterrows():
            if row["model"] not in MODELS:
                continue
            img = Image.open(io.BytesIO(row["image"])).convert("RGB")
            if img.size[0] < 300:
                continue
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            mask = white_bg_mask(bgr)
            if mask is None or not is_lateral(mask):
                continue
            mdir = os.path.join(OUT, row["model"].replace(" ", "-"))
            os.makedirs(mdir, exist_ok=True)
            k = kept[row["model"]]
            cv2.imwrite(os.path.join(mdir, f"{b:02d}_{i:04d}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(os.path.join(mdir, f"{b:02d}_{i:04d}_mask.png"), mask * 255)
            kept[row["model"]] += 1
        print(f"batch {b} done, kept: {kept}", flush=True)

if __name__ == "__main__":
    main()
