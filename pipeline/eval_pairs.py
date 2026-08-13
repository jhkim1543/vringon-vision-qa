# -*- coding: utf-8 -*-
"""Honest discrimination check: does the engine actually separate normal from defective?

Every defect sample is synthesized from the SAME source photo as its
"_ok" twin and scored against the SAME reference bank, so a paired
comparison isolates the defect as the only changed variable.
"""
import os, json, glob
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "assets", "samples")

recs = {}
for sdir in sorted(glob.glob(os.path.join(OUT, "*"))):
    rp = os.path.join(sdir, "result.json")
    if os.path.isdir(sdir) and os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            r = json.load(f)
        recs[r["id"]] = r

pairs = []
for rid, r in recs.items():
    if r["kind"] != "defect":
        continue
    stem = rid.rsplit("_" + r["gt"]["type"], 1)[0]
    ok = recs.get(stem + "_ok")
    if ok:
        pairs.append((stem, ok, r))

print(f"{len(pairs)} matched (normal, defect) pairs\n")
print(f"{'sample':<26}{'thr':>6}{'z_ok':>7}{'z_def':>7}{'lift':>7}  {'verdict ok->def':<22}{'GT'}")
wins = 0
for stem, ok, d in sorted(pairs):
    zo, zd, thr = ok["z_max"], d["z_max"], d["det_thr"]
    lift = zd - zo
    wins += lift > 0
    gt = f"loc={'Y' if d['gt']['localized'] else 'N'} type={'Y' if d['gt']['type_match'] else 'N'} IoU={d['gt']['iou']:.2f}"
    print(f"{stem[:25]:<26}{thr:>6.2f}{zo:>7.2f}{zd:>7.2f}{lift:>+7.2f}  "
          f"{ok['verdict']:>7} -> {d['verdict']:<11}{gt}")

print(f"\npaired z_max lift positive: {wins}/{len(pairs)}")

# image-level ranking over all 20 samples
y = [1 if r["kind"] == "defect" else 0 for r in recs.values()]
s_abs = [r["z_max"] for r in recs.values()]
s_rel = [r["z_max"] / r["det_thr"] for r in recs.values()]
print(f"image-level AUROC (raw z_max)          : {roc_auc_score(y, s_abs):.3f}")
print(f"image-level AUROC (z_max / det_thr)    : {roc_auc_score(y, s_rel):.3f}")

# how the verdicts actually land
from collections import Counter
for kind in ("normal", "defect"):
    c = Counter(r["verdict"] for r in recs.values() if r["kind"] == kind)
    print(f"{kind:<8} verdicts: {dict(c)}")

# Operator-facing question: is the TOP-ranked flag the real defect, or noise?
import cv2
top1 = ranked = 0
for stem, ok, d in pairs:
    gp = os.path.join(OUT, d["id"], "gt.png")
    if not (d["detections"] and os.path.exists(gp)):
        continue
    gt = cv2.imread(gp, cv2.IMREAD_UNCHANGED)[..., 3] > 0
    ranked += 1
    x, y_, w, h = d["detections"][0]["bbox"]
    if gt[y_:y_ + h, x:x + w].any():
        top1 += 1
print(f"top-1 flagged region IS the injected defect: {top1}/{ranked}")
