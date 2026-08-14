# -*- coding: utf-8 -*-
"""Run the full QA pipeline on real sneaker photos and export demo assets.

Per sample:
  silhouette -> part regions -> colorway reference retrieval -> PatchCore
  -> anomaly heatmap -> region extraction -> rule-based defect typing
  -> severity/verdict -> overlays + JSON
"""
import os, sys, json, glob, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
import engine
from engine import (white_bg_mask, crop_align, part_regions, PART_COLORS,
                    part_of_point, PatchCore)
from make_defects import GENERATORS

ROOT = os.path.join(os.path.dirname(__file__), "..")
SKU_DIR = os.path.join(ROOT, "data", "sku")
OUT = os.path.join(ROOT, "docs", "assets", "samples")
os.makedirs(OUT, exist_ok=True)

Z_PIX = 3.5          # patch-z threshold for defect pixels (robust MAD z)
MIN_AREA = 60        # px at display resolution
RNG = np.random.default_rng(42)

# SKU test images (clean lateral single-shoe, chosen from contact sheet)
TESTS = {                       # verified single-shoe photographs only
    "Superstar": ["01_0031", "01_0035", "01_0047", "01_0054"],
    "Stan-Smith": ["01_0147", "01_0157", "01_0154"],
    "Gazelle": ["01_0207", "01_0242", "01_0246"],
}
# which defects to synthesize on which test image (part-aware variety)
DEFECT_PLAN = {
    ("Superstar", "01_0031"): ["upper_contamination"],
    ("Superstar", "01_0035"): ["excess_cement"],
    ("Superstar", "01_0047"): ["loose_thread"],
    ("Superstar", "01_0054"): ["toe_scuff"],
    ("Stan-Smith", "01_0147"): ["bottom_contamination"],
    ("Stan-Smith", "01_0157"): ["excess_cement"],
    ("Stan-Smith", "01_0154"): ["upper_contamination"],
    ("Gazelle", "01_0207"): ["loose_thread"],
    ("Gazelle", "01_0242"): ["toe_scuff"],
    ("Gazelle", "01_0246"): ["bottom_contamination"],
}

def fg_hist(bgr, mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], mask, [12, 6, 6],
                     [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, None).flatten()

SIM_MIN = 0.35   # colorway similarity threshold for reference selection
REF_MAX = 8
REF_MIN = 3

def pick_refs(sims):
    """sims: sorted desc list of (sim, key, cacheval). Colorway-gated top-K."""
    good = [x for x in sims if x[0] >= SIM_MIN][:REF_MAX]
    if len(good) < REF_MIN:
        good = sims[:REF_MIN]
    return good

def load_aligned(path):
    bgr = cv2.imread(path)
    mask = white_bg_mask(bgr)
    return crop_align(bgr, mask)

def heat_png(z, mask, thr=3.5):
    """VRINGON-tone heatmap: transparent -> blue -> orange -> red."""
    h, w = z.shape
    img = np.zeros((h, w, 4), np.uint8)
    t = np.clip((z - 0.55 * thr) / (1.6 * thr), 0, 1)  # visible below thr, red above
    stops = [  # t, BGRA (vringon blue05, orange04, red05)
        (0.00, (250, 108, 93, 0)),
        (0.35, (250, 108, 93, 110)),
        (0.65, (0, 130, 251, 170)),
        (1.00, (80, 69, 255, 220)),
    ]
    out = np.zeros((h, w, 4), np.float32)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]; t1, c1 = stops[i + 1]
        sel = (t >= t0) & (t <= t1)
        f = np.zeros_like(t); f[sel] = (t[sel] - t0) / (t1 - t0 + 1e-9)
        for c in range(4):
            out[..., c][sel] = c0[c] + (c1[c] - c0[c]) * f[sel]
    out[..., 3] *= (mask > 0)
    return out.astype(np.uint8)

def parts_png(parts, mask):
    h, w = mask.shape
    img = np.zeros((h, w, 4), np.uint8)
    for name, col in PART_COLORS.items():
        m = parts[name] > 0
        img[m] = (*col, 150 if name != "cement_boundary" else 200)
    return img

ELONG_T = 3.2

def classify_region(bgr, parts, comp_mask, cx, cy):
    part = part_of_point(parts, cx, cy)
    ys, xs = np.where(comp_mask)
    area = len(ys)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    elong = max(bw, bh) / max(1, min(bw, bh))
    thin = area / max(1, bw * bh) < 0.35 and max(bw, bh) > 24
    mean_bgr = bgr[comp_mask > 0].mean(axis=0)
    b, g, r = mean_bgr
    yellowish = (r > 140 and g > 120 and r - b > 35)
    # contrast vs the whole part's median (a ring right around the component
    # sits inside the defect blob itself and washes the contrast out)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grown = cv2.dilate(comp_mask, np.ones((17, 17), np.uint8))
    bgpix = (parts.get(part, comp_mask) > 0) & (grown == 0)
    delta = (float(np.median(gray[bgpix]) - gray[comp_mask > 0].mean())
             if bgpix.sum() > 40 else 0.0)
    darker = delta > 14          # region darker than surroundings
    contrasty = abs(delta) > 14
    if part == "cement_boundary" and yellowish:
        dtype = "excess_cement"
    elif (thin or elong > ELONG_T) and part in ("upper", "collar", "heel") and area < 900:
        dtype = "loose_thread"
    elif part == "toe" and contrasty:
        dtype = "toe_scuff"
    elif part in ("outsole", "midsole") and darker:
        dtype = "bottom_contamination"
    elif part in ("upper", "collar", "heel") and darker:
        dtype = "upper_contamination"
    elif part == "cement_boundary":
        dtype = "excess_cement"
    else:
        dtype = "surface_anomaly"
    return dtype, part, (int(x0), int(y0), int(bw), int(bh))

REVIEW_F = 0.8   # candidate tier: REVIEW_F*thr < z <= thr -> low-confidence flag

def severity_of(dtype, area_pct, zmax, thr=3.5):
    if zmax <= thr:
        return "review"          # candidate tier
    if dtype == "surface_anomaly":
        return "review"
    return "major" if zmax > 2.0 * thr or area_pct > 2.5 else "minor"

def extract_detections(z, mask, bgr, parts, thr):
    """Two-tier region extraction: primary (z>thr) + candidate (z>REVIEW_F*thr)."""
    er = max(3, int(min(mask.shape) * 0.02)) | 1
    inner = cv2.erode(mask, np.ones((er, er), np.uint8))
    det_mask = ((z > REVIEW_F * thr) & (inner > 0)).astype(np.uint8)
    det_mask = cv2.morphologyEx(det_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(det_mask)
    dets = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        comp = (lab == i).astype(np.uint8)
        cx, cy = int(cents[i][0]), int(cents[i][1])
        zmax = float(z[comp > 0].max())
        area_pct = 100.0 * stats[i, cv2.CC_STAT_AREA] / max(1, mask.sum())
        dtype, part, bbox = classify_region(bgr, parts, comp, cx, cy)
        if zmax > thr:
            conf = float(np.clip(0.5 + 0.5 * (zmax - thr) / max(thr, 1e-6), 0.05, 0.99))
        else:
            conf = float(np.clip(0.3 + (zmax - REVIEW_F * thr)
                                 / max((1 - REVIEW_F) * thr, 1e-6) * 0.2, 0.2, 0.5))
        dets.append({
            "type": dtype, "part": part, "bbox": bbox,
            "area_pct": round(area_pct, 3), "z_max": round(zmax, 2),
            "confidence": round(conf, 2),
            "severity": severity_of(dtype, area_pct, zmax, thr),
        })
    dets.sort(key=lambda d: -d["z_max"])
    return dets, det_mask


def analyze(sample_id, bgr, mask, refs, meta, gt=None, pc=None):
    t0 = time.time()
    parts = part_regions(mask)
    if pc is None:
        pc = PatchCore().fit([r[0] for r in refs], [r[1] for r in refs])
    t_fit = time.time() - t0
    h, w = mask.shape
    t1 = time.time()
    z = pc.heatmap(bgr, mask, out_size=(w, h))
    t_inf = time.time() - t1

    thr = pc.det_thr or Z_PIX
    dets, det_mask = extract_detections(z, mask, bgr, parts, thr)

    sev = [d["severity"] for d in dets]
    verdict = "fail" if "major" in sev else ("review" if sev else "pass")

    # ground-truth metrics for synthetic defects
    gt_info = None
    if gt is not None:
        gt_mask, gt_type = gt
        zin = z[mask > 0]; gin = gt_mask[mask > 0] > 0
        if gin.sum() > 0 and gin.sum() < gin.size:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(gin.astype(int), zin))
        else:
            auroc = None
        pred = det_mask > 0
        inter = float((pred & (gt_mask > 0)).sum())
        union = float((pred | (gt_mask > 0)).sum())
        iou = inter / union if union else 0.0
        hit = any(_bbox_overlap(d["bbox"], gt_mask) for d in dets)
        type_ok = any(d["type"] == gt_type for d in dets)
        gt_info = {"type": gt_type, "pixel_auroc": None if auroc is None else round(auroc, 4),
                   "iou": round(iou, 3), "localized": bool(hit), "type_match": bool(type_ok)}

    sdir = os.path.join(OUT, sample_id)
    os.makedirs(sdir, exist_ok=True)
    np.savez_compressed(os.path.join(sdir, "z.npz"), z=z.astype(np.float16),
                        mask=mask.astype(np.uint8))
    cv2.imwrite(os.path.join(sdir, "image.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(sdir, "parts.png"), parts_png(parts, mask))
    cv2.imwrite(os.path.join(sdir, "heat.png"), heat_png(z, mask, thr))
    if gt is not None:
        gtc = np.zeros((h, w, 4), np.uint8); gtc[gt[0] > 0] = (80, 69, 255, 160)
        cv2.imwrite(os.path.join(sdir, "gt.png"), gtc)

    part_stats = {k: round(100.0 * v.sum() / max(1, mask.sum()), 1)
                  for k, v in parts.items()}
    rec = {
        "id": sample_id, "w": w, "h": h,
        "sku": meta["sku"], "colorway_refs": meta["n_refs"],
        "kind": meta["kind"],
        "verdict": verdict, "detections": dets, "gt": gt_info,
        "parts_pct": part_stats,
        "z_mean": round(float(z[mask > 0].mean()), 2),
        "z_max": round(float(z[mask > 0].max()), 2),
        "det_thr": round(float(thr), 2),
        "t_bank_s": round(t_fit, 1), "t_infer_s": round(t_inf, 1),
    }
    with open(os.path.join(sdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print(f"[{sample_id}] verdict={verdict} dets={len(dets)} "
          f"gt={gt_info}", flush=True)
    return rec

def _bbox_overlap(bbox, gt_mask):
    x, y, w, h = bbox
    return (gt_mask[y:y + h, x:x + w] > 0).any()

def main():
    from curation import filter_files
    all_imgs = {}
    for sku_dir in glob.glob(os.path.join(SKU_DIR, "*")):
        sku = os.path.basename(sku_dir)
        # pair photographs poison the reference bank as well as the test set
        for f in filter_files(sku, glob.glob(os.path.join(sku_dir, "*.jpg"))):
            key = os.path.splitext(os.path.basename(f))[0]
            all_imgs.setdefault(sku, {})[key] = f

    # preload aligned crops + histograms
    cache = {}
    for sku, files in all_imgs.items():
        for key, path in files.items():
            bgr, mask, _ = load_aligned(path)
            cache[(sku, key)] = (bgr, mask, fg_hist(bgr, mask))

    records = []
    for sku, keys in TESTS.items():
        for key in keys:
            bgr, mask, hist = cache[(sku, key)]
            # retrieval: same SKU, colorway similarity, exclude self
            cands = [(k, c) for (s, k), c in cache.items() if s == sku and k != key]
            sims = [(float(cv2.compareHist(hist, c[2], cv2.HISTCMP_CORREL)), k, c)
                    for k, c in cands]
            sims.sort(key=lambda x: -x[0])
            refs = [(c[0], c[1]) for _, _, c in pick_refs(sims)]
            meta = {"sku": sku, "n_refs": len(refs)}
            pc = PatchCore().fit([r[0] for r in refs], [r[1] for r in refs])

            # normal (held-out) sample
            records.append(analyze(f"{sku}_{key}_ok", bgr, mask, refs,
                                   {**meta, "kind": "normal"}, pc=pc))
            # defective versions
            for dclass in DEFECT_PLAN.get((sku, key), []):
                parts = part_regions(mask)
                for attempt in range(6):
                    r = GENERATORS[dclass](bgr, parts, RNG)
                    if r is not None and r[1].sum() > 40:
                        break
                if r is None:
                    print("!! could not synthesize", dclass, sku, key); continue
                dbgr, gt_mask, gt_type, _ = r
                records.append(analyze(f"{sku}_{key}_{dclass}", dbgr, mask, refs,
                                       {**meta, "kind": "defect"},
                                       gt=(gt_mask, gt_type), pc=pc))

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"samples": records}, f, ensure_ascii=False, indent=1)

    # aggregate quality metrics
    defs = [r for r in records if r["kind"] == "defect" and r["gt"]]
    norms = [r for r in records if r["kind"] == "normal"]
    agg = {
        "n_samples": len(records),
        "n_defect": len(defs), "n_normal": len(norms),
        "mean_pixel_auroc": round(np.mean([r["gt"]["pixel_auroc"] for r in defs
                                           if r["gt"]["pixel_auroc"]]), 4) if defs else None,
        "localization_rate": round(np.mean([r["gt"]["localized"] for r in defs]), 3) if defs else None,
        "type_match_rate": round(np.mean([r["gt"]["type_match"] for r in defs]), 3) if defs else None,
        "normal_false_alarm": round(np.mean([1 if r["detections"] else 0 for r in norms]), 3) if norms else None,
        "mean_iou": round(np.mean([r["gt"]["iou"] for r in defs]), 3) if defs else None,
    }
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=1)
    print("AGGREGATE:", json.dumps(agg), flush=True)

if __name__ == "__main__":
    main()
