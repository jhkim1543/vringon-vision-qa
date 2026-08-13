# -*- coding: utf-8 -*-
"""Local inference server for the VRINGON Vision QA demo.

Serves docs/ statically and exposes POST /api/analyze that runs the full
pipeline on an uploaded shoe image (reference SKU chosen by colorway
similarity against the local SKU library).

    .venv\\Scripts\\python.exe pipeline\\server.py   ->  http://localhost:5210
"""
import os, sys, io, glob, json, base64, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

import engine
from engine import white_bg_mask, crop_align, part_regions, PatchCore
import run_samples as rs

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOCS = os.path.join(ROOT, "docs")

app = FastAPI()
_cache = None
_pc_cache = {}

def library():
    global _cache
    if _cache is None:
        _cache = {}
        for sku_dir in glob.glob(os.path.join(rs.SKU_DIR, "*")):
            sku = os.path.basename(sku_dir)
            for f in glob.glob(os.path.join(sku_dir, "*.jpg")):
                key = os.path.splitext(os.path.basename(f))[0]
                bgr, mask, _ = rs.load_aligned(f)
                _cache[(sku, key)] = (bgr, mask, rs.fg_hist(bgr, mask))
    return _cache

def png_b64(arr):
    ok, buf = cv2.imencode(".png", arr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()

def jpg_b64(arr):
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return JSONResponse({"error": "이미지를 읽을 수 없습니다"}, status_code=400)
    scale = 700 / max(arr.shape[:2])
    if scale < 1:
        arr = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    mask = white_bg_mask(arr)
    bgr, mask, _ = crop_align(arr, mask)
    hist = rs.fg_hist(bgr, mask)

    # pick best-matching references across the whole library by colorway
    lib = library()
    sims = sorted(((float(cv2.compareHist(hist, c[2], cv2.HISTCMP_CORREL)), s, k, c)
                   for (s, k), c in lib.items()), key=lambda x: -x[0])
    sku = sims[0][1]
    refs = [(c[0], c[1]) for _, s, _, c in sims[:8] if s == sku][:8]
    ref_keys = tuple(k for _, s, k, _ in sims[:8] if s == sku)

    # Reference adequacy decides whether a verdict means anything. A near-1.0
    # top match is the uploaded shoe's own library photo (golden sample); a
    # merely similar colorway is a different physical unit, where pose and
    # unit-to-unit variation rival a real defect.
    top_sim = float(sims[0][0])
    ref_mode = ("golden" if top_sim >= 0.985 else
                "near" if top_sim >= 0.90 else "mismatch")

    if ref_keys not in _pc_cache:
        _pc_cache[ref_keys] = PatchCore().fit([r[0] for r in refs], [r[1] for r in refs])
    pc = _pc_cache[ref_keys]

    parts = part_regions(mask)
    h, w = mask.shape
    t0 = time.time()
    z = pc.heatmap(bgr, mask, out_size=(w, h))
    t_inf = time.time() - t0

    thr = pc.det_thr or rs.Z_PIX
    dets, _ = rs.extract_detections(z, mask, bgr, parts, thr)
    sev = [d["severity"] for d in dets]
    verdict = "fail" if "major" in sev else ("review" if sev else "pass")
    if ref_mode == "mismatch":
        # No comparable reference unit exists, so every score reflects model
        # and pose differences rather than condition. Abstaining beats issuing
        # a confident FAIL on a perfectly good shoe.
        verdict = "unknown"

    rec = {
        "id": f"upload_{int(time.time())}", "w": w, "h": h,
        "sku": sku, "colorway_refs": len(refs), "kind": "upload",
        "ref_sim_top1": round(top_sim, 3), "ref_mode": ref_mode,
        "verdict": verdict, "detections": dets, "gt": None,
        "parts_pct": {k: round(100.0 * v.sum() / max(1, mask.sum()), 1)
                      for k, v in parts.items()},
        "z_mean": round(float(z[mask > 0].mean()), 2),
        "z_max": round(float(z[mask > 0].max()), 2),
        "det_thr": round(float(thr), 2),
        "t_bank_s": 0.0, "t_infer_s": round(t_inf, 1),
    }
    return {
        "record": rec,
        "images": {
            "image": jpg_b64(bgr),
            "parts": png_b64(rs.parts_png(parts, mask)),
            "heat": png_b64(rs.heat_png(z, mask, thr)),
        },
    }

app.mount("/", StaticFiles(directory=DOCS, html=True), name="docs")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1",
                port=int(os.environ.get("VRINGON_QA_PORT", "5210")))
