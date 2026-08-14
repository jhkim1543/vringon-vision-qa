# -*- coding: utf-8 -*-
"""Contact sheet per predicted view, so the classifier can be checked by eye."""
import os, sys, glob, random
import numpy as np
import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
VIEWS = os.path.join(ROOT, "data", "views")
OUT = os.path.join(ROOT, "data", "view_sheets")
CELL, COLS = 150, 10


def sheet(view, n=40, seed=0):
    files = sorted(glob.glob(os.path.join(VIEWS, view, "*", "*.jpg")))
    if not files:
        return None
    random.Random(seed).shuffle(files)
    files = files[:n]
    rows = (len(files) + COLS - 1) // COLS
    canvas = np.full((rows * CELL, COLS * CELL, 3), 245, np.uint8)
    for i, f in enumerate(files):
        im = cv2.imread(f)
        if im is None:
            continue
        s = CELL / max(im.shape[:2])
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        r, c = divmod(i, COLS)
        y, x = r * CELL, c * CELL
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
        cv2.putText(canvas, os.path.basename(f)[:9], (x + 2, y + CELL - 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 160), 1)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{view}.jpg")
    cv2.imwrite(p, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{view}: {len(files)} shown of {len(glob.glob(os.path.join(VIEWS, view, '*', '*.jpg')))} -> {p}",
          flush=True)
    return p


if __name__ == "__main__":
    for v in (sys.argv[1:] or ["lateral", "top", "rear", "front", "bottom", "angled"]):
        sheet(v)
