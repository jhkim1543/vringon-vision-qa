# -*- coding: utf-8 -*-
"""Sort the sneaker photos into camera views the inspection spec needs.

The Air Max 90 spec measures different items from different cameras: eyestay
opening width from the top (CAM3), heel height/overlay/centre from the rear
(CAM4), the strobel from below (CAM5), mudguard distances from the medial and
lateral sides (CAM6/7). Free-form product photos are mostly lateral, so this
pass finds whatever other views the dataset actually contains.

View is inferred from silhouette geometry, which is stable across colourways:
  lateral  wide and low, one long toe-to-heel sweep, aspect ~1.8-3.0
  top      long but symmetric about its own long axis, waisted in the middle
  rear     nearly as tall as wide, silhouette widest at the bottom
  bottom   long, strongly waisted (arch), near mirror-symmetric
  angled   anything that does not commit
"""
import os, sys, io, glob, json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from engine import white_bg_mask

ROOT = os.path.join(os.path.dirname(__file__), "..")
SNEAK = os.path.join(ROOT, "data", "sneakers")
OUT = os.path.join(ROOT, "data", "views")

MIN_SIDE = 200
MIN_FILL = 0.06


def silhouette_features(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) < 200:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    m = mask[y0:y1 + 1, x0:x1 + 1]
    area = float(m.sum())
    fill = area / (w * h)
    if fill < MIN_FILL:
        return None
    aspect = w / h

    # column heights along the long axis, normalised
    col = m.sum(axis=0).astype(np.float32)
    col = col / (col.max() + 1e-6)
    n = len(col)
    third = max(1, n // 3)
    left, mid, right = col[:third].mean(), col[third:2 * third].mean(), col[2 * third:].mean()
    waist = mid / (0.5 * (left + right) + 1e-6)     # <1 => pinched middle (arch)

    # vertical mass distribution: rear views sit heavy at the bottom
    row = m.sum(axis=1).astype(np.float32)
    row = row / (row.max() + 1e-6)
    half = max(1, len(row) // 2)
    bottom_heavy = row[half:].mean() / (row[:half].mean() + 1e-6)

    # symmetry of the silhouette about its own horizontal mid-line
    flip = m[::-1, :]
    sym_h = float((m & flip).sum()) / max(area, 1)
    # symmetry about the vertical mid-line
    flip_v = m[:, ::-1]
    sym_v = float((m & flip_v).sum()) / max(area, 1)

    return dict(w=int(w), h=int(h), aspect=float(aspect), fill=float(fill),
                waist=float(waist), bottom_heavy=float(bottom_heavy),
                sym_h=float(sym_h), sym_v=float(sym_v))


def classify(f):
    a, waist, sym_h, bh = f["aspect"], f["waist"], f["sym_h"], f["bottom_heavy"]
    if a < 1.45:
        # squarish: rear or front view
        return "rear" if bh > 1.05 else "front"
    if a > 1.7 and sym_h > 0.80:
        # long and mirror-symmetric top-to-bottom => looking down the shoe
        return "bottom" if waist < 0.86 else "top"
    if 1.6 <= a <= 3.4 and sym_h < 0.72:
        return "lateral"
    return "angled"


def main():
    os.makedirs(OUT, exist_ok=True)
    import pandas as pd
    from PIL import Image

    rows = []
    counts = {}
    for pq in sorted(glob.glob(os.path.join(SNEAK, "*.parquet"))):
        df = pd.read_parquet(pq)
        tag = os.path.splitext(os.path.basename(pq))[0][-2:]
        for i, r in df.iterrows():
            b = r["image"]
            if isinstance(b, dict):
                b = b["bytes"]
            try:
                im = Image.open(io.BytesIO(b)).convert("RGB")
            except Exception:
                continue
            if min(im.size) < MIN_SIDE:
                continue
            bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            mask = white_bg_mask(bgr)
            feat = silhouette_features(mask)
            if feat is None:
                continue
            view = classify(feat)
            key = f"{tag}_{i:04d}"
            d = os.path.join(OUT, view, str(r["model"]).replace(" ", "-"))
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, key + ".jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts[view] = counts.get(view, 0) + 1
            rows.append({"key": key, "model": str(r["model"]), "view": view, **feat})
        print(f"{os.path.basename(pq)} done: {counts}", flush=True)

    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"counts": counts, "items": rows}, f, ensure_ascii=False)
    print("VIEW COUNTS:", json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
