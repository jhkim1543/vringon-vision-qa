# -*- coding: utf-8 -*-
"""Quick separation experiment on one test key with colorway-gated refs."""
import sys, os, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
import run_samples as rs
from engine import PatchCore, part_regions
from make_defects import GENERATORS

RNG = np.random.default_rng(42)

cache = {}
for sku_dir in glob.glob(os.path.join(rs.SKU_DIR, "*")):
    sku = os.path.basename(sku_dir)
    for f in glob.glob(os.path.join(sku_dir, "*.jpg")):
        key = os.path.splitext(os.path.basename(f))[0]
        bgr, mask, _ = rs.load_aligned(f)
        cache[(sku, key)] = (bgr, mask, rs.fg_hist(bgr, mask))

sku, key = "Superstar", "01_0031"
bgr, mask, hist = cache[(sku, key)]
sims = sorted(((float(cv2.compareHist(hist, c[2], cv2.HISTCMP_CORREL)), k, c)
               for (s, k), c in cache.items() if s == sku and k != key),
              key=lambda x: -x[0])
picked = rs.pick_refs(sims)
print("refs:", [(k, round(v, 2)) for v, k, _ in picked], flush=True)
refs = [(c[0], c[1]) for _, _, c in picked]

pc = PatchCore().fit([r[0] for r in refs], [r[1] for r in refs])
print("det_thr:", round(pc.det_thr, 2),
      "ref_maxes:", [round(x, 2) for x in pc.ref_maxes], flush=True)

h, w = mask.shape
z_ok = pc.heatmap(bgr, mask, out_size=(w, h))
print("OK   z_max %.2f  p99.9 %.2f" % (z_ok[mask > 0].max(),
      np.percentile(z_ok[mask > 0], 99.9)), flush=True)

parts = part_regions(mask)
for dclass in ["upper_contamination", "loose_thread", "toe_scuff"]:
    r = GENERATORS[dclass](bgr, parts, RNG)
    if r is None:
        print(dclass, "synth failed"); continue
    dbgr, gt, _, _ = r
    z = pc.heatmap(dbgr, mask, out_size=(w, h))
    gm = (gt > 0) & (mask > 0)
    print("%-20s z_max %.2f  gt_max %.2f  gt_mean %.2f" % (
        dclass, z[mask > 0].max(), z[gm].max(), z[gm].mean()), flush=True)
