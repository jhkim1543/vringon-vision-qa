# -*- coding: utf-8 -*-
"""Controlled-acquisition track: the same unit re-photographed in a fixture.

The field track fails because every reference is a DIFFERENT physical shoe
photographed freely, so unit and pose variation rivals a real defect. A
factory station does not work that way: the shoe sits in a jig under fixed
lighting, so the only normal variation is placement jitter and sensor noise.

This script reproduces that condition honestly — one source photo becomes a
reference bank of re-photographed instances (sub-degree rotation, ~1% scale,
a few pixels of shift, small gain/white-balance drift, sensor noise, JPEG
requantisation) plus a held-out instance used as the test image. Nothing
about the engine changes; only the acquisition condition does.
"""
import os, sys, json, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from engine import white_bg_mask, crop_align, part_regions, PatchCore
from make_defects import GENERATORS
import run_samples as rs

ROOT = os.path.join(os.path.dirname(__file__), "..")
SKU_DIR = os.path.join(ROOT, "data", "sku")
OUT = os.path.join(ROOT, "docs", "assets", "samples")

N_REF = 6
RNG = np.random.default_rng(2026)

# one source per defect type, spread across the three SKUs
PLAN = [
    ("Superstar", "01_0031", "upper_contamination"),
    ("Superstar", "01_0047", "loose_thread"),
    ("Stan-Smith", "01_0113", "bottom_contamination"),
    ("Stan-Smith", "01_0124", "excess_cement"),
    ("Gazelle", "01_0242", "toe_scuff"),
    ("Gazelle", "01_0246", "upper_contamination"),
]


def rephotograph(bgr, mask, rng):
    """One more shot of the SAME shoe in the same jig."""
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2),
                                rng.uniform(-1.2, 1.2),          # jig play
                                1 + rng.uniform(-0.012, 0.012))  # focus/height
    M[0, 2] += rng.uniform(-0.008, 0.008) * w
    M[1, 2] += rng.uniform(-0.008, 0.008) * h
    img = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    m = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    f = img.astype(np.float32)
    f *= (1 + rng.uniform(-0.025, 0.025))                        # exposure drift
    f *= np.float32([1 + rng.uniform(-0.012, 0.012), 1.0,        # white balance
                     1 + rng.uniform(-0.012, 0.012)])
    f += rng.normal(0, 1.2, f.shape)                             # sensor noise
    img = np.clip(f, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), m


def main():
    records = []
    for sku, key, dclass in PLAN:
        src = os.path.join(SKU_DIR, sku, key + ".jpg")
        bgr0 = cv2.imread(src)
        m0 = white_bg_mask(bgr0)
        bgr0, m0, _ = crop_align(bgr0, m0)

        shots = [rephotograph(bgr0, m0, RNG) for _ in range(N_REF + 1)]
        refs, (test_bgr, test_mask) = shots[:N_REF], shots[N_REF]

        t0 = time.time()
        pc = PatchCore().fit([r[0] for r in refs], [r[1] for r in refs])
        t_fit = time.time() - t0
        print(f"[{sku}_{key}] rig bank {N_REF} shots, thr={pc.det_thr:.2f} "
              f"({t_fit:.0f}s)", flush=True)

        meta = {"sku": sku, "n_refs": N_REF}
        rec = rs.analyze(f"rig_{sku}_{key}_ok", test_bgr, test_mask, refs,
                         {**meta, "kind": "normal"}, pc=pc)
        records.append(rec)

        parts = part_regions(test_mask)
        r = None
        for _ in range(8):
            r = GENERATORS[dclass](test_bgr, parts, RNG)
            if r is not None and r[1].sum() > 60:
                break
        if r is None:
            print("!! synth failed", dclass, sku, key, flush=True)
            continue
        dbgr, gt_mask, gt_type, _ = r
        rec = rs.analyze(f"rig_{sku}_{key}_{dclass}", dbgr, test_mask, refs,
                         {**meta, "kind": "defect"}, gt=(gt_mask, gt_type), pc=pc)
        records.append(rec)

    # merge into the manifest, tagging both tracks
    mp = os.path.join(OUT, "manifest.json")
    with open(mp, encoding="utf-8") as f:
        manifest = json.load(f)
    for s in manifest["samples"]:
        s["track"] = "rig" if s["id"].startswith("rig_") else "field"
    for r in records:
        r["track"] = "rig"
    keep = [s for s in manifest["samples"] if not s["id"].startswith("rig_")]
    manifest["samples"] = keep + records
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"RIG DONE: {len(records)} samples", flush=True)


if __name__ == "__main__":
    main()
