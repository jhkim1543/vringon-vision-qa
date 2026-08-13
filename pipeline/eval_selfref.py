# -*- coding: utf-8 -*-
"""Decisive experiment: within-part self-reference vs cross-image memory bank.

Scores every (normal, defect) twin with both methods and reports the metrics
that actually matter to an operator: does the defect image score higher than
its own clean twin, and does the single hottest spot land on the real defect.
"""
import os, sys, json, glob
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from engine import part_regions, self_ref_map

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "assets", "samples")

K_FRACS = [float(x) for x in (sys.argv[1:] or ["0.06"])]

recs = {}
for sdir in sorted(glob.glob(os.path.join(OUT, "*"))):
    rp = os.path.join(sdir, "result.json")
    if os.path.isdir(sdir) and os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            recs[json.load(f)["id"]] = sdir

def load(sid):
    sdir = recs[sid]
    bgr = cv2.imread(os.path.join(sdir, "image.jpg"))
    mask = np.load(os.path.join(sdir, "z.npz"))["mask"]
    gp = os.path.join(sdir, "gt.png")
    gt = None
    if os.path.exists(gp):
        gt = (cv2.imread(gp, cv2.IMREAD_UNCHANGED)[..., 3] > 0)
    return bgr, mask, gt

with open(os.path.join(OUT, "manifest.json"), encoding="utf-8") as f:
    manifest = json.load(f)["samples"]
pairs = []
for r in manifest:
    if r["kind"] == "defect":
        stem = r["id"].rsplit("_" + r["gt"]["type"], 1)[0]
        if stem + "_ok" in recs:
            pairs.append((stem, stem + "_ok", r["id"], r["gt"]["type"]))

def peak_on_gt(z, mask, gt, frac=0.999):
    """Is the hottest region of the map on the injected defect?"""
    zin = z[mask > 0]
    thr = np.quantile(zin, frac)
    hot = (z >= thr) & (mask > 0)
    return bool((hot & gt).any())

for kf in K_FRACS:
    print(f"\n===== self-reference, k_frac={kf} =====")
    print(f"{'sample':<22}{'z_ok':>7}{'z_def':>7}{'lift':>7}{'pxAUROC':>9}  peak_on_gt")
    scores, labels, lifts, hits, aurocs = [], [], [], [], []
    for stem, ok_id, d_id, dtype in sorted(pairs):
        bo, mo, _ = load(ok_id)
        bd, md, gt = load(d_id)
        parts = part_regions(mo)
        zo = self_ref_map(bo, parts, mo)
        zd = self_ref_map(bd, part_regions(md), md)
        so, sd = float(zo[mo > 0].max()), float(zd[md > 0].max())
        au = roc_auc_score(gt[md > 0].astype(int), zd[md > 0]) if gt is not None else float("nan")
        hit = peak_on_gt(zd, md, gt) if gt is not None else False
        scores += [so, sd]; labels += [0, 1]
        lifts.append(sd - so); hits.append(hit); aurocs.append(au)
        print(f"{stem[:21]:<22}{so:>7.2f}{sd:>7.2f}{sd-so:>+7.2f}{au:>9.4f}  {'Y' if hit else 'N'}")
    print(f"paired lift positive : {sum(1 for l in lifts if l > 0)}/{len(lifts)}")
    print(f"image-level AUROC    : {roc_auc_score(labels, scores):.3f}")
    print(f"mean pixel AUROC     : {np.nanmean(aurocs):.4f}")
    print(f"peak lands on defect : {sum(hits)}/{len(hits)}")
