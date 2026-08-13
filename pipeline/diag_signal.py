# -*- coding: utf-8 -*-
"""Decisive diagnostic: is the defect signal the strongest thing in the image?

For each defect sample, compare the peak INSIDE the injected defect against
the peak OUTSIDE it (reference-mismatch noise), and against the same image's
clean twin. If in_gt > out_gt the defect is the dominant signal and a decision
rule can work; if not, no scoring rule can recover it and the limit is the
reference data itself.
"""
import os, sys, json
import numpy as np
import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "assets", "samples")

def load(sid):
    sd = os.path.join(OUT, sid)
    a = np.load(os.path.join(sd, "z.npz"))
    z, mask = a["z"].astype(np.float32), a["mask"]
    with open(os.path.join(sd, "result.json"), encoding="utf-8") as f:
        rec = json.load(f)
    gp = os.path.join(sd, "gt.png")
    gt = (cv2.imread(gp, cv2.IMREAD_UNCHANGED)[..., 3] > 0) if os.path.exists(gp) else None
    return z, mask, gt, rec

with open(os.path.join(OUT, "manifest.json"), encoding="utf-8") as f:
    manifest = json.load(f)["samples"]

print(f"{'sample':<24}{'thr':>6}{'in_gt':>7}{'out_gt':>8}{'clean_max':>10}"
      f"{'gt_area%':>9}   verdict")
dom = clean_beat = 0
n = 0
for r in manifest:
    if r["kind"] != "defect":
        continue
    stem = r["id"].rsplit("_" + r["gt"]["type"], 1)[0]
    z, mask, gt, rec = load(r["id"])
    zc, mc, _, _ = load(stem + "_ok")
    fg = mask > 0
    in_gt = float(z[gt & fg].max())
    out_gt = float(z[(~gt) & fg].max())
    clean_max = float(zc[mc > 0].max())
    area = 100.0 * (gt & fg).sum() / fg.sum()
    n += 1
    dom += in_gt > out_gt
    clean_beat += in_gt > clean_max
    flag = "  <-- defect dominates" if in_gt > out_gt else ""
    print(f"{stem[:23]:<24}{rec['det_thr']:>6.2f}{in_gt:>7.2f}{out_gt:>8.2f}"
          f"{clean_max:>10.2f}{area:>9.2f}   {rec['verdict']}{flag}")

print(f"\ndefect peak beats in-image noise peak : {dom}/{n}")
print(f"defect peak beats its clean twin's peak: {clean_beat}/{n}")
