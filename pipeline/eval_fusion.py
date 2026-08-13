# -*- coding: utf-8 -*-
"""AND-fusion of cross-reference and within-part self-reference anomaly maps.

The two detectors fail in opposite directions:
  cross-reference  — knows laces/stripes/logo are normal (they are in every
                     reference), but fires on pose, lighting and unit-to-unit
                     differences between physically different shoes.
  self-reference   — immune to pose/lighting (everything cancels inside one
                     image), but fires on every structural element.
A real defect is the only thing that is anomalous under BOTH, so the fused
score is the elementwise minimum of the two normalised maps.
"""
import os, sys, json, glob
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from engine import part_regions, self_ref_map

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "assets", "samples")
SELF_SIGMA = 4.0     # robust-z at which a self-reference deviation counts as 1.0

def sdir_of(sid):
    return os.path.join(OUT, sid)

def load_maps(sid):
    """(bgr, mask, gt, z_cross_normalised, z_self_normalised)."""
    sd = sdir_of(sid)
    bgr = cv2.imread(os.path.join(sd, "image.jpg"))
    a = np.load(os.path.join(sd, "z.npz"))
    z_cross, mask = a["z"].astype(np.float32), a["mask"]
    with open(os.path.join(sd, "result.json"), encoding="utf-8") as f:
        thr = json.load(f)["det_thr"]
    sp = os.path.join(sd, "zself.npz")
    if os.path.exists(sp):
        z_self = np.load(sp)["z"].astype(np.float32)
    else:
        z_self = self_ref_map(bgr, part_regions(mask), mask)
        np.savez_compressed(sp, z=z_self.astype(np.float16))
    gp = os.path.join(sd, "gt.png")
    gt = (cv2.imread(gp, cv2.IMREAD_UNCHANGED)[..., 3] > 0) if os.path.exists(gp) else None
    return bgr, mask, gt, z_cross / max(thr, 1e-6), z_self / SELF_SIGMA

with open(os.path.join(OUT, "manifest.json"), encoding="utf-8") as f:
    manifest = json.load(f)["samples"]
pairs = []
for r in manifest:
    if r["kind"] == "defect":
        stem = r["id"].rsplit("_" + r["gt"]["type"], 1)[0]
        if os.path.isdir(sdir_of(stem + "_ok")):
            pairs.append((stem, stem + "_ok", r["id"]))

FUSIONS = {
    "cross only": lambda c, s: c,
    "self only": lambda c, s: s,
    "min (AND)": lambda c, s: np.minimum(c, s),
    "geo mean": lambda c, s: np.sqrt(np.clip(c, 0, None) * np.clip(s, 0, None)),
}

def peak_on_gt(z, mask, gt, frac=0.999):
    thr = np.quantile(z[mask > 0], frac)
    return bool(((z >= thr) & (mask > 0) & gt).any())

cache = {}
for _, ok_id, d_id in pairs:
    for sid in (ok_id, d_id):
        if sid not in cache:
            cache[sid] = load_maps(sid)
            print(f"  maps ready: {sid}", flush=True)

print()
for name, fuse in FUSIONS.items():
    scores, labels, lifts, hits, aurocs = [], [], [], [], []
    for stem, ok_id, d_id in sorted(pairs):
        _, mo, _, co, so_ = cache[ok_id]
        _, md, gt, cd, sd_ = cache[d_id]
        zo, zd = fuse(co, so_), fuse(cd, sd_)
        a, b = float(zo[mo > 0].max()), float(zd[md > 0].max())
        scores += [a, b]; labels += [0, 1]; lifts.append(b - a)
        aurocs.append(roc_auc_score(gt[md > 0].astype(int), zd[md > 0]))
        hits.append(peak_on_gt(zd, md, gt))
    print(f"{name:<12} imgAUROC {roc_auc_score(labels, scores):.3f}  "
          f"pxAUROC {np.mean(aurocs):.4f}  "
          f"lift+ {sum(1 for l in lifts if l > 0)}/{len(lifts)}  "
          f"peak_on_defect {sum(hits)}/{len(hits)}")
