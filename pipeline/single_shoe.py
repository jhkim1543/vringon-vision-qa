# -*- coding: utf-8 -*-
"""Reject photographs that contain more than one shoe.

Every measurement in this project assumes one shoe in the frame: the silhouette
becomes the reference length, the bottom line becomes the datum, the toe run
becomes the tip arc. A product photo of a PAIR breaks all three at once and the
pipeline reports plausible-looking numbers for a shape that is two shoes.

Two silhouette statistics separate them, and the thresholds below were set from
the labelled examples in the Stan-Smith library rather than guessed:

  solidity   area / convex-hull area. A single lateral shoe is a fairly convex
             wedge; two shoes at different angles carve a notch between them.
  humps      count of distinct maxima in the top profile. One shoe has one
             collar; a pair shows two.
"""
import numpy as np
import cv2

SOLIDITY_MIN = 0.72
HUMPS_MAX = 1


def profile_humps(mask, smooth=9, prominence=0.12):
    """Distinct maxima of the silhouette's top edge."""
    h, w = mask.shape
    top = np.full(w, np.nan, np.float32)
    for x in range(w):
        col = np.where(mask[:, x] > 0)[0]
        if len(col):
            top[x] = h - col.min()            # height above image bottom
    ok = ~np.isnan(top)
    if ok.sum() < 20:
        return 0, top
    xs = np.where(ok)[0]
    t = top[xs]
    k = max(3, int(len(t) * 0.06)) | 1
    t = cv2.GaussianBlur(t.reshape(-1, 1), (1, k), 0).ravel()
    rng = t.max() - t.min()
    if rng <= 0:
        return 1, top
    tn = (t - t.min()) / rng
    peaks = []
    for i in range(1, len(tn) - 1):
        if tn[i] >= tn[i - 1] and tn[i] > tn[i + 1]:
            peaks.append(i)
    # keep peaks separated by a real valley, not by ripple
    keep = []
    for p in peaks:
        if not keep:
            keep.append(p)
            continue
        valley = tn[keep[-1]:p].min()
        if min(tn[keep[-1]], tn[p]) - valley >= prominence:
            keep.append(p)
        elif tn[p] > tn[keep[-1]]:
            keep[-1] = p
    return max(1, len(keep)), top


def solidity(mask):
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return 0.0
    c = max(cs, key=cv2.contourArea)
    a = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    ah = cv2.contourArea(hull)
    return float(a / ah) if ah > 0 else 0.0


def check(mask):
    s = solidity(mask)
    n, _ = profile_humps(mask)
    single = (s >= SOLIDITY_MIN) and (n <= HUMPS_MAX)
    return {"single": bool(single), "solidity": round(s, 3), "humps": int(n)}


if __name__ == "__main__":
    import os, sys, glob
    sys.path.insert(0, os.path.dirname(__file__))
    from engine import white_bg_mask, crop_align
    root = os.path.join(os.path.dirname(__file__), "..", "data", "sku")
    bad = []
    for sku in sorted(os.listdir(root)):
        for f in sorted(glob.glob(os.path.join(root, sku, "*.jpg"))):
            bgr = cv2.imread(f)
            m = white_bg_mask(bgr)
            _, m, _ = crop_align(bgr, m)
            r = check(m)
            key = os.path.splitext(os.path.basename(f))[0]
            flag = "" if r["single"] else "   <-- MULTI"
            print(f"{sku:<12}{key:<10} solidity={r['solidity']:.3f} "
                  f"humps={r['humps']}{flag}", flush=True)
            if not r["single"]:
                bad.append(f"{sku}/{key}")
    print("\nrejected:", bad)
