# -*- coding: utf-8 -*-
"""Find an image-level score that survives reference mismatch.

Localisation already works (pixel AUROC 0.946). What fails is the PASS/FAIL
decision: a clean shoe whose references match poorly gets a uniformly raised
map, so its absolute z_max rivals a real defect's. A defect is a LOCALISED
bump, so scores measuring peak prominence — the peak relative to the image's
own bulk — should separate the two. All candidates are evaluated on the
already-computed maps, so this is a pure scoring-rule comparison.
"""
import os, sys, json, glob
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "assets", "samples")

def load(sid):
    sd = os.path.join(OUT, sid)
    a = np.load(os.path.join(sd, "z.npz"))
    z, mask = a["z"].astype(np.float32), a["mask"]
    sp = os.path.join(sd, "zself.npz")
    zs = np.load(sp)["z"].astype(np.float32) if os.path.exists(sp) else None
    return z, mask, zs

with open(os.path.join(OUT, "manifest.json"), encoding="utf-8") as f:
    manifest = json.load(f)["samples"]
pairs = []
for r in manifest:
    if r["kind"] == "defect":
        stem = r["id"].rsplit("_" + r["gt"]["type"], 1)[0]
        if os.path.isdir(os.path.join(OUT, stem + "_ok")):
            pairs.append((stem, stem + "_ok", r["id"]))

def tophat(z, mask, frac=0.22):
    """Local prominence: peak minus a large-scale background estimate."""
    ks = max(9, int(min(z.shape) * frac)) | 1
    bg = cv2.morphologyEx(z, cv2.MORPH_OPEN, np.ones((ks, ks), np.uint8))
    return z - bg

def stats(z, mask):
    v = z[mask > 0]
    return v, np.percentile(v, [50, 90, 95, 99])

SCORES = {}
SCORES["z_max (current)"] = lambda z, m, zs: z[m > 0].max()
SCORES["z_max - p50"] = lambda z, m, zs: (lambda v, q: v.max() - q[0])(*stats(z, m))
SCORES["z_max - p99"] = lambda z, m, zs: (lambda v, q: v.max() - q[3])(*stats(z, m))
SCORES["(max-p50)/(p99-p50)"] = lambda z, m, zs: (lambda v, q: (v.max() - q[0]) / max(q[3] - q[0], 1e-6))(*stats(z, m))
SCORES["(max-p95)/(p95-p50)"] = lambda z, m, zs: (lambda v, q: (v.max() - q[2]) / max(q[2] - q[0], 1e-6))(*stats(z, m))
SCORES["tophat max"] = lambda z, m, zs: tophat(z, m)[m > 0].max()
SCORES["tophat max x zmax"] = lambda z, m, zs: tophat(z, m)[m > 0].max() * max(z[m > 0].max(), 0)
SCORES["tophat, self-gated"] = lambda z, m, zs: (tophat(z, m) * np.clip(zs / 4.0, 0, 1.5))[m > 0].max() if zs is not None else np.nan

rows = []
for name, fn in SCORES.items():
    vals, labels, lifts = [], [], []
    for stem, ok_id, d_id in sorted(pairs):
        zo, mo, so = load(ok_id)
        zd, md, sd = load(d_id)
        a, b = float(fn(zo, mo, so)), float(fn(zd, md, sd))
        vals += [a, b]; labels += [0, 1]; lifts.append(b - a)
    if np.isnan(vals).any():
        continue
    rows.append((roc_auc_score(labels, vals), sum(1 for l in lifts if l > 0), name))

print(f"{'image-level score':<24}{'imgAUROC':>10}{'paired lift+':>14}")
for auc, lp, name in sorted(rows, reverse=True):
    print(f"{name:<24}{auc:>10.3f}{lp:>10}/{len(pairs)}")
