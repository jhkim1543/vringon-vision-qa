# -*- coding: utf-8 -*-
"""Inspect the rendered GSO views — the only track where millimetres are real.

Two things happen here that cannot happen on a product photo:

1. Every remaining inspection item gets exercised. Items 3, 8 and 9 need the
   top, rear and bottom cameras, and these renders supply them.
2. Measurements can be checked against ground truth. The meshes are metric and
   the camera is orthographic, so mm-per-pixel is exact; the pipeline's own
   estimate of the part length can be compared with the mesh bounding box, and
   that error is a real number rather than a claim.

Standing caveat, repeated in the UI: GSO shoes are FINISHED shoes. They are not
lasted uppers, they have no strobel board, and no Nike model appears anywhere in
the dataset. What is validated here is geometry, not appearance.
"""
import os, sys, json, glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from engine import white_bg_mask, part_regions
from inspect_items import inspect
from run_inspection import judge, overlay, K_SIGMA, MIN_SIGMA
from spec_airmax90 import ITEMS, CAMERAS, RIG

ROOT = os.path.join(os.path.dirname(__file__), "..")
VIEWS_DIR = os.path.join(ROOT, "data", "gso_views")
OUT = os.path.join(ROOT, "docs", "assets", "gso")
VIEWS = ("lateral", "top", "rear", "bottom")
N_DEMO = 6          # models shipped as demo samples; all are used for statistics


def load_view(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None, None
    mask = white_bg_mask(bgr)
    if mask.sum() < 500:
        return None, None
    return bgr, mask


def measure_model(mdir, view):
    p = os.path.join(mdir, view + ".jpg")
    bgr, mask = load_view(p)
    if bgr is None:
        return None
    parts = part_regions(mask) if view == "lateral" else {}
    recs, summary = inspect(bgr, mask, parts, view=view)
    return bgr, mask, recs, summary


def main():
    os.makedirs(OUT, exist_ok=True)
    index = json.load(open(os.path.join(VIEWS_DIR, "index.json"), encoding="utf-8"))
    models = index["models"]

    # ---- pass 1: measure everything, and check length against ground truth
    per_view = {v: {} for v in VIEWS}
    gt_rows = []
    for m in models:
        mdir = os.path.join(VIEWS_DIR, m["name"])
        for v in VIEWS:
            r = measure_model(mdir, v)
            if r is None:
                continue
            _, _, recs, summary = r
            per_view[v].setdefault("recs", {})[m["name"]] = recs
            if v == "lateral":
                mmpp = m["views"][v]["mm_per_px"]
                est = summary["ref_len_px"] * mmpp
                gt_rows.append({"name": m["name"], "true_mm": m["length_mm"],
                                "est_mm": round(est, 1),
                                "err_mm": round(est - m["length_mm"], 1),
                                "err_pct": round(100 * (est - m["length_mm"]) / m["length_mm"], 2)})
        print(f"  measured {m['name'][:40]}", flush=True)

    errs = np.array([abs(r["err_pct"]) for r in gt_rows]) if gt_rows else np.array([])
    ground_truth = {
        "n": len(gt_rows),
        "mean_abs_err_pct": round(float(errs.mean()), 2) if errs.size else None,
        "p95_abs_err_pct": round(float(np.percentile(errs, 95)), 2) if errs.size else None,
        "worst": sorted(gt_rows, key=lambda r: -abs(r["err_pct"]))[:5],
        "note": ("파이프라인이 추정한 부품 기준길이를 메시 실측 길이와 대조한 값입니다. "
                 "정사영 렌더라 mm/픽셀이 정확하므로, 이 오차는 실루엣·기준길이 추정 "
                 "알고리즘 자체의 오차입니다."),
    }
    print("GROUND TRUTH:", json.dumps(ground_truth["mean_abs_err_pct"]), flush=True)

    # ---- pass 2: golden statistics per view, across models
    stats = {}
    for v in VIEWS:
        vals = {}
        for name, recs in per_view[v].get("recs", {}).items():
            for r in recs:
                if r["measured"] is not None:
                    vals.setdefault(r["item_id"], []).append(r["measured"])
        s = {}
        for k, arr in vals.items():
            a = np.array(arr, float)
            med = float(np.median(a))
            mad = float(np.median(np.abs(a - med))) * 1.4826
            s[k] = {"n": len(a), "centre": round(med, 2),
                    "sigma": round(max(mad, MIN_SIGMA), 2)}
        stats[v] = s

    # ---- pass 3: emit demo samples
    samples = []
    for m in models[:N_DEMO]:
        mdir = os.path.join(VIEWS_DIR, m["name"])
        for v in VIEWS:
            r = measure_model(mdir, v)
            if r is None:
                continue
            bgr, mask, recs, summary = r
            mmpp = m["views"][v]["mm_per_px"]
            recs = [judge(x, stats[v]) for x in recs]
            for x in recs:                     # renders have a real scale
                if x["measured"] is not None and x["units"].startswith("‰"):
                    x["mm"] = round(x["measured"] / 1000.0 * summary["ref_len_px"] * mmpp, 2)
            sid = f"{m['name'][:38]}__{v}"
            d = os.path.join(OUT, sid)
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "image.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(os.path.join(d, "measure.jpg"), overlay(bgr, recs),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            verdicts = [x.get("verdict") for x in recs if x.get("verdict")]
            rec = {"id": sid, "sku": m["name"], "view": v,
                   "w": int(mask.shape[1]), "h": int(mask.shape[0]),
                   "mm_per_px": mmpp, "length_mm": m["length_mm"],
                   "summary": summary,
                   "verdict": ("FAIL" if "FAIL" in verdicts else
                               "REVIEW" if "REVIEW" in verdicts else
                               "PASS" if verdicts else "NO_VERDICT"),
                   "items": recs}
            json.dump(rec, open(os.path.join(d, "result.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            samples.append({k: rec[k] for k in
                            ("id", "sku", "view", "w", "h", "verdict", "length_mm")})

    manifest = {
        "samples": samples, "golden": stats, "k_sigma": K_SIGMA,
        "ground_truth": ground_truth,
        "dataset": {
            "name": "Google Scanned Objects",
            "license": "CC BY 4.0 — Google Research / Open Robotics",
            "n_models_rendered": len(models),
            "caveat": ("GSO 신발은 밑창이 붙은 완제품이며 라스트 갑피가 아닙니다. "
                       "스트로벨 보드가 존재하지 않고 Nike 모델도 없습니다. "
                       "여기서 검증되는 것은 기하이지 외관이 아닙니다."),
        },
        "spec": {
            "source": "SHC글로벌 · Air Max 90 Lasted Upper 검사항목 및 기구장치 검토건 (Smart Vision Tech, 2019-06-21)",
            "items": [{k: it[k] for k in
                       ("id", "no", "name_en", "name_ko", "sensing", "cameras",
                        "view", "feasibility", "vendor_logic", "our_method")}
                      for it in ITEMS],
            "cameras": CAMERAS, "rig": RIG,
        },
    }
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("GSO INSPECT DONE:", len(samples), "samples", flush=True)


if __name__ == "__main__":
    main()
