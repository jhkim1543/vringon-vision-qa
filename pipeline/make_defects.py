# -*- coding: utf-8 -*-
"""Part-aware synthetic defect generator with ground-truth masks.

Defects are only placed on parts where they physically occur
(loose thread near stitch/collar, cement on the sole boundary, ...),
following the Defect x Valid-Part matrix from the footwear literature.
"""
import numpy as np
import cv2

def _blob(shape, cx, cy, rx, ry, rng, irregular=0.55):
    """Irregular soft blob mask (float 0..1)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    base = np.clip(1.2 - d, 0, 1)
    noise = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 9)
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-6)
    m = base * (1 - irregular + irregular * noise)
    m = np.clip((m - 0.25) * 2.2, 0, 1)
    return cv2.GaussianBlur(m, (0, 0), 2.0)

def _sample_point(part_mask, rng, erode=7):
    m = cv2.erode(part_mask, np.ones((erode, erode), np.uint8))
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        ys, xs = np.where(part_mask > 0)
    if len(ys) == 0:
        return None
    i = rng.integers(len(ys))
    return int(xs[i]), int(ys[i])

def upper_contamination(bgr, parts, rng):
    p = _sample_point(parts["upper"], rng, erode=11)
    if p is None: return None
    cx, cy = p
    h, w = bgr.shape[:2]
    scale = max(h, w)
    m = _blob((h, w), cx, cy, rng.uniform(.05, .09) * scale, rng.uniform(.035, .06) * scale, rng)
    m *= parts["upper"].astype(np.float32)
    out = bgr.astype(np.float32)
    tint = np.array(rng.choice([[38, 52, 70], [30, 38, 48], [45, 70, 95]]), np.float32)
    stain = out * (1 - 0.68 * m[..., None]) + tint * 0.68 * m[..., None]
    return stain.astype(np.uint8), (m > 0.18).astype(np.uint8), "upper_contamination", "upper"

def bottom_contamination(bgr, parts, rng):
    zone = (parts["outsole"] | parts["midsole"]).astype(np.uint8)
    p = _sample_point(zone, rng, erode=3)
    if p is None: return None
    cx, cy = p
    h, w = bgr.shape[:2]
    scale = max(h, w)
    m = _blob((h, w), cx, cy, rng.uniform(.07, .12) * scale, rng.uniform(.02, .035) * scale, rng)
    m *= zone.astype(np.float32)
    out = bgr.astype(np.float32)
    dark = out * (1 - 0.72 * m[..., None]) + np.float32([25, 30, 34]) * 0.72 * m[..., None]
    return dark.astype(np.uint8), (m > 0.18).astype(np.uint8), "bottom_contamination", "midsole"

def excess_cement(bgr, parts, rng):
    band = parts["cement_boundary"]
    ys, xs = np.where(band > 0)
    if len(xs) == 0: return None
    x0 = rng.integers(xs.min(), max(xs.min() + 1, xs.max() - 60))
    seg = (xs >= x0) & (xs <= x0 + rng.integers(50, 110))
    if seg.sum() < 30: return None
    h, w = bgr.shape[:2]
    m = np.zeros((h, w), np.float32)
    m[ys[seg], xs[seg]] = 1
    # smear upward onto the upper
    kern = np.zeros((13, 5), np.uint8); kern[:9, :] = 1
    m = cv2.dilate(m, kern)
    m = cv2.GaussianBlur(m, (0, 0), 3.0)
    m = np.clip(m * 1.6, 0, 1)
    out = bgr.astype(np.float32)
    glue = np.float32([120, 190, 215])  # yellowish
    res = out * (1 - 0.62 * m[..., None]) + glue * 0.62 * m[..., None]
    # glossy speckle
    spec = (rng.random((h, w)) > 0.985).astype(np.float32) * m
    spec = cv2.GaussianBlur(spec, (0, 0), 1.0)
    res = np.clip(res + spec[..., None] * 90, 0, 255)
    return res.astype(np.uint8), (m > 0.22).astype(np.uint8), "excess_cement", "cement_boundary"

def loose_thread(bgr, parts, rng):
    zone = (parts["collar"] | parts["heel"] | parts["upper"]).astype(np.uint8)
    p = _sample_point(zone, rng, erode=9)
    if p is None: return None
    cx, cy = p
    h, w = bgr.shape[:2]
    canvas = np.zeros((h, w), np.float32)
    pts = [(cx, cy)]
    ang = rng.uniform(0, 2 * np.pi)
    for _ in range(rng.integers(6, 11)):
        ang += rng.uniform(-1.1, 1.1)
        step = rng.uniform(6, 15)
        nx = int(np.clip(pts[-1][0] + np.cos(ang) * step, 1, w - 2))
        ny = int(np.clip(pts[-1][1] + np.sin(ang) * step, 1, h - 2))
        pts.append((nx, ny))
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(canvas, a, b, 1.0, 3, cv2.LINE_AA)
    canvas = np.clip(cv2.GaussianBlur(canvas, (0, 0), 0.5) * 1.4, 0, 1)
    # thread color: contrast with local patch
    local = bgr[max(0, cy - 8): cy + 8, max(0, cx - 8): cx + 8].mean(axis=(0, 1))
    col = np.float32([235, 238, 240]) if local.mean() < 128 else np.float32([40, 46, 52])
    out = bgr.astype(np.float32)
    res = out * (1 - canvas[..., None]) + col * canvas[..., None]
    gt = cv2.dilate((canvas > 0.15).astype(np.uint8), np.ones((5, 5), np.uint8))
    return res.astype(np.uint8), gt, "loose_thread", "upper"

def toe_scuff(bgr, parts, rng):
    p = _sample_point(parts["toe"], rng, erode=7)
    if p is None: return None
    cx, cy = p
    h, w = bgr.shape[:2]
    m = np.zeros((h, w), np.float32)
    ang = rng.uniform(-0.5, 0.5)
    for i in range(rng.integers(5, 9)):
        off = i * rng.uniform(3, 5.5)
        x0 = int(cx - 26 + rng.uniform(-6, 6)); y0 = int(cy + off - 12)
        x1 = int(cx + 26 + rng.uniform(-6, 6)); y1 = int(y0 + np.tan(ang) * 52)
        cv2.line(m, (x0, y0), (x1, y1), rng.uniform(.7, 1.), rng.integers(2, 4), cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 0.8) * parts["toe"]
    out = bgr.astype(np.float32)
    local = out[max(0, cy - 10): cy + 10, max(0, cx - 10): cx + 10].mean(axis=(0, 1))
    col = local * 0.35 if local.mean() > 110 else np.clip(local * 2.4 + 60, 0, 255)
    res = out * (1 - m[..., None]) + col * m[..., None]
    gt = cv2.dilate((m > 0.12).astype(np.uint8), np.ones((3, 3), np.uint8))
    return res.astype(np.uint8), gt, "toe_scuff", "toe"

GENERATORS = {
    "upper_contamination": upper_contamination,
    "bottom_contamination": bottom_contamination,
    "excess_cement": excess_cement,
    "loose_thread": loose_thread,
    "toe_scuff": toe_scuff,
}
