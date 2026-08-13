# -*- coding: utf-8 -*-
"""Grid-search the review-tier threshold factor on saved z heatmaps."""
import os, sys, json, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from run_samples import MIN_AREA, _bbox_overlap

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "assets", "samples")

def comps_at(z, mask, thr):
    er = max(3, int(min(mask.shape) * 0.02)) | 1
    inner = cv2.erode(mask, np.ones((er, er), np.uint8))
    dm = ((z > thr) & (inner > 0)).astype(np.uint8)
    dm = cv2.morphologyEx(dm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(dm)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        x, y, w, h = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3]
        out.append(((x, y, w, h), float(z[lab == i].max())))
    return out

samples = []
for sdir in sorted(glob.glob(os.path.join(OUT, "*"))):
    zp = os.path.join(sdir, "z.npz")
    rp = os.path.join(sdir, "result.json")
    if not (os.path.exists(zp) and os.path.exists(rp)):
        continue
    with open(rp, encoding="utf-8") as f:
        rec = json.load(f)
    a = np.load(zp)
    gt_mask = None
    gp = os.path.join(sdir, "gt.png")
    if rec.get("gt") and os.path.exists(gp):
        gtc = cv2.imread(gp, cv2.IMREAD_UNCHANGED)
        gt_mask = (gtc[..., 3] > 0).astype(np.uint8)
    samples.append((rec["id"], rec["kind"], rec["det_thr"],
                    a["z"].astype(np.float32), a["mask"], gt_mask))

print(f"{len(samples)} samples loaded", flush=True)
for f in [1.0, 0.9, 0.85, 0.8, 0.75, 0.7]:
    loc = tot = 0
    norm_dets = []
    for sid, kind, thr, z, mask, gt in samples:
        cs = comps_at(z, mask, thr * f)
        if kind == "defect" and gt is not None:
            tot += 1
            if any(_bbox_overlap(b, gt) for b, _ in cs):
                loc += 1
        elif kind == "normal":
            norm_dets.append(len(cs))
    print(f"f={f:.2f}  defect_loc={loc}/{tot}  "
          f"normal_alarm={sum(1 for x in norm_dets if x)}/{len(norm_dets)}  "
          f"normal_mean_dets={np.mean(norm_dets):.1f}  max={max(norm_dets)}",
          flush=True)
