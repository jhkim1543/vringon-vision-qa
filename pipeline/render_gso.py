# -*- coding: utf-8 -*-
"""Render the camera views the spec needs, from metric GSO meshes.

The spec measures items 3 and 9 from the top (CAM3) and bottom (CAM5), and
items 6-8 from the rear (CAM4). No public photograph set gives us those, but
GSO ships metric meshes under CC BY 4.0, so we can render them ourselves at a
KNOWN scale — which does two things at once: it fills the missing views, and
it finally lets a measurement be checked against ground-truth millimetres
instead of only against other measurements.

Deliberately a plain orthographic z-buffer rasteriser: an orthographic camera
removes perspective foreshortening, so mm-per-pixel is exact and constant, and
that is the whole point of rendering rather than photographing.

Honesty boundary: GSO shoes are FINISHED shoes with soles attached, not lasted
uppers. The strobel board does not exist in any of them. Geometry algorithms
can be exercised; strobel appearance and the red mark cannot.
"""
import os, sys, json, glob
import numpy as np
import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
GSO = os.path.join(ROOT, "data", "gso", "models")
OUT = os.path.join(ROOT, "data", "gso_views")

RES = 640
MARGIN = 0.06
# (name, forward axis, up axis) in mesh coordinates; GSO shoes stand on Z=0
VIEWS = {
    "lateral": ("x", "z"),
    "top": ("x", "y"),
    "bottom": ("x", "y"),
    "rear": ("y", "z"),
}


def load_obj(path):
    """Vertices and triangle indices. No materials: we shade by normal."""
    vs, fs = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                vs.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    fs.append((idx[0], idx[i], idx[i + 1]))
    return np.array(vs, np.float32), np.array(fs, np.int32)


def axes_of(v):
    """(length axis, width axis) — GSO shoes are consistently Z-up, but the
    long axis alternates between X and Y from model to model, so it has to be
    detected instead of assumed."""
    ext = v.max(0) - v.min(0)
    la = 0 if ext[0] >= ext[1] else 1
    return la, 1 - la


def orient(v, view, la, wa):
    """Map mesh axes to image axes (u right, w down, d depth) for a view."""
    L, W, H = v[:, la], v[:, wa], v[:, 2]
    if view == "lateral":
        u, w, d = L, -H, W
    elif view == "rear":
        u, w, d = W, -H, L
    elif view == "top":
        u, w, d = L, -W, H
    else:                                     # bottom: look up from below
        u, w, d = L, W, -H
    return np.stack([u, w, d], 1)


def render(vs, fs, view, la, wa):
    """Orthographic z-buffer render on white, plus the exact mm-per-pixel."""
    p = orient(vs, view, la, wa)
    uw = p[:, :2]
    lo, hi = uw.min(0), uw.max(0)
    span = (hi - lo).max() * (1 + 2 * MARGIN)
    if span <= 0:
        return None, None
    scale = RES / span                        # pixels per mesh unit
    centre = (lo + hi) / 2
    px = (uw - centre) * scale + RES / 2

    img = np.full((RES, RES, 3), 255, np.uint8)
    zbuf = np.full((RES, RES), -1e9, np.float32)

    tri = px[fs]                              # (F,3,2)
    depth = p[fs][:, :, 2].mean(1)
    # normals for a simple lambert term, so the render has edges to measure
    a, b, c = vs[fs[:, 0]], vs[fs[:, 1]], vs[fs[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    n = n / ln
    light = np.array([0.35, 0.4, 0.85], np.float32)
    light /= np.linalg.norm(light)
    shade = np.clip(0.30 + 0.70 * np.abs(n @ light), 0, 1)

    order = np.argsort(depth)                 # painter's order, back to front
    for i in order:
        t = tri[i]
        if not np.isfinite(t).all():
            continue
        g = int(round(60 + 150 * shade[i]))
        cv2.fillConvexPoly(img, np.int32(np.round(t)), (g, g, g), cv2.LINE_8)
    # float() matters: scale comes from a float32 mesh, and numpy scalars are
    # not JSON serialisable
    return img, float(1000.0 / scale)         # GSO meshes are in metres


def bbox_mm(vs):
    e = (vs.max(0) - vs.min(0)) * 1000.0
    return [round(float(x), 1) for x in e]


def main():
    os.makedirs(OUT, exist_ok=True)
    index = []
    dirs = sorted(glob.glob(os.path.join(GSO, "*")))
    for d in dirs:
        obj = os.path.join(d, "meshes", "model.obj")
        if not os.path.exists(obj):
            continue
        name = os.path.basename(d)
        try:
            vs, fs = load_obj(obj)
        except Exception as e:
            print("  !!", name, e, flush=True)
            continue
        if len(fs) == 0:
            continue
        ext = bbox_mm(vs)
        od = os.path.join(OUT, name)
        os.makedirs(od, exist_ok=True)
        la, wa = axes_of(vs)
        rec = {"name": name, "bbox_mm": ext, "views": {},
               "length_mm": round(float((vs.max(0) - vs.min(0))[la] * 1000), 1),
               "width_mm": round(float((vs.max(0) - vs.min(0))[wa] * 1000), 1),
               "height_mm": round(float((vs.max(0) - vs.min(0))[2] * 1000), 1)}
        for view in VIEWS:
            img, mmpp = render(vs, fs, view, la, wa)
            if img is None:
                continue
            cv2.imwrite(os.path.join(od, f"{view}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            rec["views"][view] = {"mm_per_px": round(mmpp, 5)}
        index.append(rec)
        print(f"  {name}: {ext} mm, views={list(rec['views'])}", flush=True)
    json.dump({"license": "CC BY 4.0 — Google Research / Open Robotics",
               "source": "Google Scanned Objects via Gazebo Fuel",
               "models": index},
              open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("RENDER DONE:", len(index), flush=True)


if __name__ == "__main__":
    main()
