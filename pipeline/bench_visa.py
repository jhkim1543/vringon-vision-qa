# -*- coding: utf-8 -*-
"""Benchmark the same PatchCore engine on real VisA (cashew) defects.

Proves the anomaly engine's quality on real industrial ground truth:
image-level AUROC + pixel-level AUROC against human-annotated masks.
"""
import os, sys, glob, json, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from engine import PatchCore, patch_features
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
VISA = os.path.join(ROOT, "data", "visa", "cashew", "Data")
OUT = os.path.join(ROOT, "docs", "assets", "visa")
os.makedirs(OUT, exist_ok=True)

N_TRAIN = 120     # normals for the memory bank
N_TEST_OK = 40    # held-out normals
RES = 320

def load(path):
    bgr = cv2.imread(path)
    s = min(bgr.shape[:2])
    bgr = bgr[:s, :s]
    return cv2.resize(bgr, (RES * 2, RES * 2), interpolation=cv2.INTER_AREA)

def main():
    normals = sorted(glob.glob(os.path.join(VISA, "Images", "Normal", "*.JPG")))
    anoms = sorted(glob.glob(os.path.join(VISA, "Images", "Anomaly", "*.JPG")))
    masks = sorted(glob.glob(os.path.join(VISA, "Masks", "Anomaly", "*.png")))
    print(f"normals={len(normals)} anoms={len(anoms)} masks={len(masks)}", flush=True)
    train = normals[:N_TRAIN]
    test_ok = normals[N_TRAIN:N_TRAIN + N_TEST_OK]
    test_ng = anoms
    mask_of = {os.path.splitext(os.path.basename(m))[0]: m for m in masks}

    t0 = time.time()
    pc = PatchCore(coreset_ratio=0.12, use_tiles=False).fit([load(p) for p in train])
    print(f"bank fit in {time.time()-t0:.0f}s", flush=True)

    img_scores, img_labels = [], []
    pix_scores, pix_labels = [], []
    examples = []
    z_cache = {}    # key -> subsampled z map, for metric recomputes without re-run
    for i, p in enumerate(test_ok + test_ng):
        is_ng = i >= len(test_ok)
        bgr = load(p)
        z = pc.heatmap(bgr, out_size=(RES, RES))
        img_scores.append(float(z.max())); img_labels.append(int(is_ng))
        key = os.path.splitext(os.path.basename(p))[0]
        sub = np.arange(0, RES, 2)
        z_cache[("ng_" if is_ng else "ok_") + key] = z[np.ix_(sub, sub)].astype(np.float16)
        if is_ng and key in mask_of:
            gm = cv2.imread(mask_of[key], 0)   # VisA masks are {0,1,2}, not 0/255
            s = min(gm.shape[:2]); gm = gm[:s, :s]
            gm = cv2.resize(gm, (RES, RES), interpolation=cv2.INTER_NEAREST)
            pix_scores.append(z[np.ix_(sub, sub)].reshape(-1))
            pix_labels.append((gm[np.ix_(sub, sub)] > 0).reshape(-1).astype(int))
        elif not is_ng:
            pix_scores.append(z[np.ix_(sub, sub)].reshape(-1))
            pix_labels.append(np.zeros(len(sub) ** 2, int))
        if i % 20 == 0:
            print(f"scored {i}/{len(test_ok)+len(test_ng)}", flush=True)
        # save a few visual examples for the demo
        if is_ng and len(examples) < 6 and key in mask_of:
            disp = cv2.resize(bgr, (RES, RES))
            heat = np.clip((z - 2) / 5, 0, 1)
            hm = (cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
            vis = (disp * 0.55 + hm * 0.45).astype(np.uint8)
            cv2.imwrite(os.path.join(OUT, f"{key}_orig.jpg"), disp)
            cv2.imwrite(os.path.join(OUT, f"{key}_heat.jpg"), vis)
            gm = cv2.imread(mask_of[key], 0)
            s = min(gm.shape[:2]); gm = cv2.resize(gm[:s, :s], (RES, RES),
                                                   interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(OUT, f"{key}_gt.png"), (gm > 0).astype(np.uint8) * 255)
            examples.append(key)
    np.savez_compressed(os.path.join(OUT, "z_cache.npz"), **z_cache)

    img_auroc = roc_auc_score(img_labels, img_scores)
    pix_auroc = roc_auc_score(np.concatenate(pix_labels), np.concatenate(pix_scores))
    res = {
        "dataset": "VisA cashew (CC BY 4.0, real factory defects)",
        "n_bank": N_TRAIN, "n_test_normal": len(test_ok), "n_test_anomaly": len(test_ng),
        "image_auroc": round(float(img_auroc), 4),
        "pixel_auroc": round(float(pix_auroc), 4),
        "examples": examples,
    }
    with open(os.path.join(OUT, "benchmark.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("RESULT:", json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
