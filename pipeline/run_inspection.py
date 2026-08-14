# -*- coding: utf-8 -*-
"""Golden-sample statistics, guard-banded verdicts, and demo assets.

We have no Nike drawing and no tolerance table, so absolute millimetre pass/fail
is not available and inventing one is the fastest way to lose a QA engineer's
trust. Instead each characteristic is judged against the spread of a golden set
of the same style, in per-mille of the part's own reference length:

  centre = median of the golden set          (not the mean: one bad golden
  spread = 1.4826 x MAD                       sample would widen every limit)
  limits = centre +/- k x spread              k is published, not hidden

The verdict has three states, never two. The guard band is the measurement
uncertainty u: inside the limits by more than u is PASS, outside by more than u
is FAIL, and the band in between is REVIEW because the measurement genuinely
cannot tell. That is standard metrology practice and it turns every borderline
case from an embarrassment into an honest answer.
"""
import os, sys, glob, json, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from engine import white_bg_mask, crop_align, part_regions
from inspect_items import inspect
from spec_airmax90 import ITEMS, BY_ID, CAMERAS, RIG

ROOT = os.path.join(os.path.dirname(__file__), "..")
SKU_DIR = os.path.join(ROOT, "data", "sku")
OUT = os.path.join(ROOT, "docs", "assets", "inspect")

K_SIGMA = 3.0          # published, chosen — not derived from a spec
MIN_SIGMA = 2.0        # ‰ floor: never claim a tolerance tighter than this
TESTS = {
    "Superstar": ["01_0031", "01_0035", "01_0047"],
    "Stan-Smith": ["01_0113", "01_0124", "01_0154"],
    "Gazelle": ["01_0207", "01_0242", "01_0246"],
}


def load(path):
    bgr = cv2.imread(path)
    mask = white_bg_mask(bgr)
    bgr, mask, _ = crop_align(bgr, mask)
    return bgr, mask, part_regions(mask)


def measure(path):
    bgr, mask, parts = load(path)
    recs, summary = inspect(bgr, mask, parts, view="lateral")
    return bgr, mask, recs, summary


def golden_stats(sku, exclude):
    """Centre and robust spread per item, from the rest of the style's library."""
    vals = {}
    files = [f for f in sorted(glob.glob(os.path.join(SKU_DIR, sku, "*.jpg")))
             if os.path.splitext(os.path.basename(f))[0] not in exclude]
    used = 0
    for f in files:
        try:
            _, _, recs, _ = measure(f)
        except Exception as e:
            print(f"  golden skip {os.path.basename(f)}: {e}", flush=True)
            continue
        used += 1
        for r in recs:
            if r["measured"] is not None:
                vals.setdefault(r["item_id"], []).append(r["measured"])
    stats = {}
    for k, v in vals.items():
        a = np.array(v, float)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med))) * 1.4826
        stats[k] = {"n": len(a), "centre": round(med, 2),
                    "sigma": round(max(mad, MIN_SIGMA), 2)}
    return stats, used


def judge(rec, stats):
    """Three-state, guard-banded verdict."""
    st = stats.get(rec["item_id"])
    if rec["measured"] is None or st is None:
        rec["nominal"] = None if not st else st["centre"]
        rec["tol_lower"] = rec["tol_upper"] = None
        rec["deviation"] = None
        if rec["status"] == "MEASURED":
            rec["status"] = "NO_GOLDEN"
            rec["note"] += " 골든 통계가 없어 판정하지 않았습니다."
        return rec
    # distances and capability percentages are non-negative by construction, so
    # a lower limit below zero is not a tolerance, it is an unbounded check
    lo = max(0.0, st["centre"] - K_SIGMA * st["sigma"])
    hi = st["centre"] + K_SIGMA * st["sigma"]
    u = rec.get("uncertainty") or 0.0
    v = rec["measured"]
    rec["nominal"] = st["centre"]
    rec["sigma"] = st["sigma"]
    rec["golden_n"] = st["n"]
    rec["tol_lower"] = round(lo, 2)
    rec["tol_upper"] = round(hi, 2)
    rec["deviation"] = round(v - st["centre"], 2)
    if rec["status"] in ("MEASURED", "ADVISORY"):
        if lo + u <= v <= hi - u:
            rec["verdict"] = "PASS"
        elif v < lo - u or v > hi + u:
            rec["verdict"] = "FAIL"
        else:
            rec["verdict"] = "REVIEW"
    return rec


def overlay(bgr, recs):
    """Draw the measurement geometry the way an operator would check it."""
    img = bgr.copy()
    col = {"PASS": (140, 220, 80), "REVIEW": (92, 171, 255),
           "FAIL": (80, 69, 255), None: (200, 200, 200)}
    for r in recs:
        g = r.get("geometry") or {}
        c = col.get(r.get("verdict"), (200, 200, 200))
        t = g.get("type")
        if t == "gap":
            a, b = np.int32(g["a"]), np.int32(g["b"])
            cv2.line(img, tuple(a), tuple(b), c, 2, cv2.LINE_AA)
            for p in (a, b):
                cv2.line(img, (p[0] - 7, p[1]), (p[0] + 7, p[1]), c, 2, cv2.LINE_AA)
            cv2.putText(img, str(r["no"]), (a[0] + 9, (a[1] + b[1]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
        elif t == "segment":
            cv2.line(img, tuple(np.int32(g["a"])), tuple(np.int32(g["b"])),
                     c, 2, cv2.LINE_AA)
            cv2.putText(img, str(r["no"]), tuple(np.int32(g["a"]) + np.int32([5, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
        elif t == "arc" and g.get("pts"):
            pts = np.int32(g["pts"]).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, c, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(np.int32(g["tip"])), 4, c, -1, cv2.LINE_AA)
        elif t == "cross":
            p = np.int32(g["p"])
            cv2.drawMarker(img, tuple(p), c, cv2.MARKER_TILTED_CROSS, 13, 2)
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    samples, all_stats = [], {}
    for sku, keys in TESTS.items():
        t0 = time.time()
        stats, used = golden_stats(sku, exclude=set(keys))
        all_stats[sku] = {"stats": stats, "n_golden": used}
        print(f"[{sku}] golden from {used} images ({time.time()-t0:.0f}s)", flush=True)
        for key in keys:
            path = os.path.join(SKU_DIR, sku, key + ".jpg")
            bgr, mask, recs, summary = measure(path)
            recs = [judge(r, stats) for r in recs]
            sid = f"{sku}_{key}"
            d = os.path.join(OUT, sid)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "image.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(os.path.join(d, "measure.jpg"), overlay(bgr, recs),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            verdicts = [r.get("verdict") for r in recs if r.get("verdict")]
            rec = {
                "id": sid, "sku": sku, "view": "lateral",
                "w": int(mask.shape[1]), "h": int(mask.shape[0]),
                "summary": summary,
                "verdict": ("FAIL" if "FAIL" in verdicts else
                            "REVIEW" if "REVIEW" in verdicts else
                            "PASS" if verdicts else "NO_VERDICT"),
                "items": recs,
            }
            json.dump(rec, open(os.path.join(d, "result.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            samples.append({k: rec[k] for k in ("id", "sku", "view", "w", "h", "verdict")})
            print(f"  {sid}: {rec['verdict']} "
                  f"({summary['n_measured']}M/{summary['n_advisory']}A/"
                  f"{summary['n_not_sensed']}N)", flush=True)

    manifest = {
        "samples": samples,
        "golden": all_stats,
        "k_sigma": K_SIGMA,
        "spec": {
            "source": "SHC글로벌 · Air Max 90 Lasted Upper 검사항목 및 기구장치 검토건 (Smart Vision Tech, 2019-06-21)",
            "items": [{k: it[k] for k in
                       ("id", "no", "name_en", "name_ko", "sensing", "cameras",
                        "view", "feasibility", "vendor_logic", "our_method")}
                      for it in ITEMS],
            "cameras": CAMERAS,
            "rig": RIG,
        },
    }
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("INSPECT DONE:", len(samples), "samples", flush=True)


if __name__ == "__main__":
    main()
