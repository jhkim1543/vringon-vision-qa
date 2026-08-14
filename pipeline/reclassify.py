# -*- coding: utf-8 -*-
"""Re-derive detections from saved z.npz heatmaps (no PatchCore re-run).

Used after improving the rule-based defect classifier: recomputes
components, typing, severity, verdict, GT metrics; rewrites result.json,
manifest.json and metrics.json in docs/assets/samples/.
"""
import os, sys, json, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from engine import part_regions
from run_samples import extract_detections, _bbox_overlap

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "assets", "samples")

def rework(sdir):
    rid = os.path.basename(sdir)
    rp = os.path.join(sdir, "result.json")
    zp = os.path.join(sdir, "z.npz")
    if not (os.path.exists(rp) and os.path.exists(zp)):
        return None
    with open(rp, encoding="utf-8") as f:
        rec = json.load(f)
    arr = np.load(zp)
    z, mask = arr["z"].astype(np.float32), arr["mask"]
    bgr = cv2.imread(os.path.join(sdir, "image.jpg"))
    parts = part_regions(mask)
    thr = rec["det_thr"]

    dets, det_mask = extract_detections(z, mask, bgr, parts, thr)
    sev = [d["severity"] for d in dets]
    rec["detections"] = dets
    rec["verdict"] = "fail" if "major" in sev else ("review" if sev else "pass")

    if rec.get("gt"):
        gtc = cv2.imread(os.path.join(sdir, "gt.png"), cv2.IMREAD_UNCHANGED)
        gt_mask = (gtc[..., 3] > 0).astype(np.uint8)
        gt_type = rec["gt"]["type"]
        zin = z[mask > 0]; gin = gt_mask[mask > 0] > 0
        auroc = None
        if 0 < gin.sum() < gin.size:
            from sklearn.metrics import roc_auc_score
            auroc = round(float(roc_auc_score(gin.astype(int), zin)), 4)
        pred = det_mask > 0
        inter = float((pred & (gt_mask > 0)).sum())
        union = float((pred | (gt_mask > 0)).sum())
        rec["gt"] = {"type": gt_type, "pixel_auroc": auroc,
                     "iou": round(inter / union if union else 0.0, 3),
                     "localized": bool(any(_bbox_overlap(d["bbox"], gt_mask) for d in dets)),
                     "type_match": bool(any(d["type"] == gt_type for d in dets))}

    with open(rp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(f"[{rid}] verdict={rec['verdict']} dets={len(dets)} gt={rec.get('gt')}",
          flush=True)
    return rec

def main():
    recs = {}
    for sdir in sorted(glob.glob(os.path.join(OUT, "*"))):
        if os.path.isdir(sdir):
            r = rework(sdir)
            if r:
                recs[r["id"]] = r

    mp = os.path.join(OUT, "manifest.json")
    with open(mp, encoding="utf-8") as f:
        manifest = json.load(f)
    # result.json carries no track label, so merging must preserve the one the
    # manifest already has — otherwise every rig sample silently becomes field
    merged = []
    for s in manifest["samples"]:
        r = recs.get(s["id"])
        if r is None:
            merged.append(s)
            continue
        r = dict(r)
        r["track"] = s.get("track", "rig" if s["id"].startswith("rig_") else "field")
        merged.append(r)
    manifest["samples"] = merged
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    records = manifest["samples"]
    defs = [r for r in records if r["kind"] == "defect" and r.get("gt")]
    norms = [r for r in records if r["kind"] == "normal"]
    agg = {
        "n_samples": len(records),
        "n_defect": len(defs), "n_normal": len(norms),
        "mean_pixel_auroc": round(float(np.mean([r["gt"]["pixel_auroc"] for r in defs
                                                 if r["gt"]["pixel_auroc"]])), 4) if defs else None,
        "localization_rate": round(float(np.mean([r["gt"]["localized"] for r in defs])), 3) if defs else None,
        "type_match_rate": round(float(np.mean([r["gt"]["type_match"] for r in defs])), 3) if defs else None,
        "normal_false_alarm": round(float(np.mean([1 if r["detections"] else 0 for r in norms])), 3) if norms else None,
        "normal_fail_rate": round(float(np.mean([1 if r["verdict"] == "fail" else 0 for r in norms])), 3) if norms else None,
        "mean_iou": round(float(np.mean([r["gt"]["iou"] for r in defs])), 3) if defs else None,
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(agg, f, indent=1)
    print("AGGREGATE:", json.dumps(agg), flush=True)

if __name__ == "__main__":
    main()
