# -*- coding: utf-8 -*-
"""Per-track metrics that cannot be inflated by flagging more regions.

The earlier `localization_rate` counted a hit whenever ANY of the (often 5-7)
flagged regions touched the ground truth, so over-flagging raised it. Every
measure here is decision-relevant instead:

  image_auroc        can the system tell a defective image from a clean one
  top1_region_is_gt  the region an operator looks at first is the real defect
  defect_dominates   the defect outranks the reference-mismatch noise
  pixel_auroc        the heatmap ranks defect pixels above clean ones

Reported separately for the two acquisition tracks, because the whole point
of the demo is that the same engine only becomes decisive under a fixed rig.
"""
import os, json, glob
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "docs", "assets", "samples")

with open(os.path.join(OUT, "manifest.json"), encoding="utf-8") as f:
    manifest = json.load(f)
by_id = {s["id"]: s for s in manifest["samples"]}


def gt_of(sid):
    gp = os.path.join(OUT, sid, "gt.png")
    if not os.path.exists(gp):
        return None
    return cv2.imread(gp, cv2.IMREAD_UNCHANGED)[..., 3] > 0


def z_of(sid):
    a = np.load(os.path.join(OUT, sid, "z.npz"))
    return a["z"].astype(np.float32), a["mask"] > 0


def track_metrics(samples):
    recs = {s["id"]: s for s in samples}
    pairs = []
    for sid, r in recs.items():
        if r["kind"] == "defect" and r.get("gt"):
            stem = sid.rsplit("_" + r["gt"]["type"], 1)[0]
            if stem + "_ok" in recs:
                pairs.append((stem + "_ok", sid))

    top1 = ranked = dominates = 0
    img_scores, img_labels = [], []
    for ok_id, d_id in pairs:
        zd, md = z_of(d_id)
        zo, mo = z_of(ok_id)
        gt = gt_of(d_id)
        img_scores += [float(zo[mo].max()) / recs[ok_id]["det_thr"],
                       float(zd[md].max()) / recs[d_id]["det_thr"]]
        img_labels += [0, 1]
        if gt is not None and (gt & md).any() and ((~gt) & md).any():
            dominates += float(zd[gt & md].max()) > float(zd[(~gt) & md].max())
        dets = recs[d_id]["detections"]
        if dets and gt is not None:
            ranked += 1
            x, y, w, h = dets[0]["bbox"]
            top1 += bool(gt[y:y + h, x:x + w].any())

    defs = [r for r in recs.values() if r["kind"] == "defect" and r.get("gt")]
    norms = [r for r in recs.values() if r["kind"] == "normal"]
    n_pairs = max(len(pairs), 1)
    return {
        "n_samples": len(recs), "n_defect": len(defs), "n_normal": len(norms),
        "mean_pixel_auroc": round(float(np.mean([r["gt"]["pixel_auroc"] for r in defs
                                                 if r["gt"]["pixel_auroc"]])), 4),
        "mean_iou": round(float(np.mean([r["gt"]["iou"] for r in defs])), 3),
        "image_auroc": round(float(roc_auc_score(img_labels, img_scores)), 3)
        if len(set(img_labels)) > 1 else None,
        "top1_region_is_gt": round(top1 / max(ranked, 1), 3),
        "defect_dominates": round(dominates / n_pairs, 3),
        "type_match_rate": round(float(np.mean([r["gt"]["type_match"] for r in defs])), 3),
        "defect_flagged_rate": round(float(np.mean([r["verdict"] != "pass" for r in defs])), 3),
        "normal_pass_rate": round(float(np.mean([r["verdict"] == "pass" for r in norms])), 3),
        "normal_fail_rate": round(float(np.mean([r["verdict"] == "fail" for r in norms])), 3),
    }


tracks = {}
for name in ("rig", "field"):
    sub = [s for s in manifest["samples"] if s.get("track", "field") == name]
    if sub:
        tracks[name] = track_metrics(sub)

agg = {
    "tracks": tracks,
    "headline_track": "rig" if "rig" in tracks else "field",
    "limitation": ("자유 촬영 트랙은 레퍼런스가 동일 SKU의 서로 다른 실물 개체 사진이라 "
                   "개체·포즈 차이가 결함 신호와 같은 크기입니다. 위치 표시는 유효하지만 "
                   "PASS/FAIL 자동 판정은 성립하지 않습니다. 같은 엔진을 고정 리그 조건에 "
                   "두면 판정이 성립하며, 이 대비가 곧 촬영 통제가 전제라는 근거입니다."),
}
# keep the flat keys the first version of the UI read
agg.update({k: v for k, v in tracks.get("field", {}).items()})
with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(agg, f, ensure_ascii=False, indent=1)
print(json.dumps(agg, ensure_ascii=False, indent=1))
