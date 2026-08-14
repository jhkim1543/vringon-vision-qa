# -*- coding: utf-8 -*-
"""Run the eleven Air Max 90 inspection items and emit QIF-shaped records.

One record per item, always — an item that cannot be sensed from the view we
have is reported with status NOT_SENSED and the camera it would need, rather
than being dropped. A report that silently omits what it could not do is how a
vision demo misleads a QA engineer.

Values are in per-mille of the part's own reference length. A single
uncontrolled photograph cannot yield millimetres: focal length, object
distance, obliquity and lens distortion are four unknowns that one image of an
object of unknown size cannot separate. Ratios of two co-planar lengths in the
same image (items 4 and 5) are the most scale-robust things here and are the
measurements the demo stands on.
"""
import math
import numpy as np
import cv2

import metrology as M
from spec_airmax90 import ITEMS, BY_ID


# --------------------------------------------------------------- geometry aids

def bottom_line(mask, x_lo=0.15, x_hi=0.85):
    """Robust sole/bottom line from the lowest silhouette points."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return None
    x0, x1 = xs.min(), xs.max()
    lo, hi = x0 + (x1 - x0) * x_lo, x0 + (x1 - x0) * x_hi
    pts = []
    for x in range(int(lo), int(hi) + 1):
        col = np.where(mask[:, x] > 0)[0]
        if len(col):
            pts.append((x, col.max()))
    if len(pts) < 20:
        return None
    p = np.array(pts, np.float64)
    # trim the toe-spring and heel curl, which are not part of the ground line
    keep = p[:, 1] >= np.quantile(p[:, 1], 0.35)
    p = p[keep]
    line = M.fit_line_odr(p)
    line["pts"] = p          # kept for the Monte-Carlo uncertainty of item 6
    return line


def signed_height_map(shape, line):
    """Perpendicular distance above the bottom line, per pixel."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = np.array([-line["dir"][1], line["dir"][0]], np.float32)
    d = (xx - line["point"][0]) * n[0] + (yy - line["point"][1]) * n[1]
    return -d if n[1] > 0 else d          # positive upward from the line


def band(mask, height, lo, hi, x_lo=None, x_hi=None):
    b = (mask > 0) & (height >= lo) & (height <= hi)
    if x_lo is not None:
        xs = np.arange(mask.shape[1])[None, :]
        b &= (xs >= x_lo) & (xs <= x_hi)
    return b.astype(np.uint8)


def dominant_line_in_band(gray, band_mask, ref_dir, tol_deg=14, min_pts=40):
    """Mudguard-style edge: LSD seeds filtered by direction, then a sub-pixel
    ODR refit of the gradient points that actually support the seed."""
    segs = M.lsd_segments(gray, band_mask, min_len=14)
    ref = math.atan2(ref_dir[1], ref_dir[0])
    cand = []
    for x1, y1, x2, y2 in segs:
        ang = math.atan2(y2 - y1, x2 - x1)
        d = abs(((ang - ref + math.pi / 2) % math.pi) - math.pi / 2)
        if math.degrees(d) <= tol_deg:
            cand.append(((x1, y1, x2, y2), math.hypot(x2 - x1, y2 - y1)))
    pts = M.edge_points_in_band(gray, band_mask, keep=0.18)
    if len(pts) < min_pts:
        return None, None
    if cand:
        # keep gradient points near the strongest seed, so a competing edge
        # in the same band does not drag the fit
        (sx1, sy1, sx2, sy2), _ = max(cand, key=lambda c: c[1])
        sp = np.array([sx1, sy1], np.float64)
        sd = np.array([sx2 - sx1, sy2 - sy1], np.float64)
        sd /= (np.linalg.norm(sd) + 1e-9)
        dist = np.abs(M.point_line_distance(pts, sp, sd))
        sel = dist <= max(4.0, np.quantile(dist, 0.35))
        if sel.sum() >= min_pts:
            pts = pts[sel]
    return M.fit_line_odr(pts), pts


def toe_run(in_toe):
    """Longest contiguous circular run of True — the toe arc of a closed contour.

    All runs must be considered, not the first one found: the contour's start
    index can land inside the toe zone, which splits the arc into a short head
    and a long tail, and taking the head silently reports a fraction of the arc.
    """
    n = len(in_toe)
    if not in_toe.any() or in_toe.all():
        return None
    starts = [i for i in range(n) if in_toe[i] and not in_toe[(i - 1) % n]]
    best = None
    for s in starts:
        run = [s]
        while in_toe[(run[-1] + 1) % n] and len(run) < n:
            run.append((run[-1] + 1) % n)
        if run[0] > run[-1]:                 # wrapped; arc_length needs a range
            continue
        if best is None or len(run) > len(best):
            best = run
    return best


def _rec(item, **kw):
    """QIF-shaped characteristic record."""
    r = {
        "item_id": item["id"], "no": item["no"],
        "name_en": item["name_en"], "name_ko": item["name_ko"],
        "cameras": item["cameras"], "sensing": item["sensing"],
        "required_view": item["view"], "feasibility": item["feasibility"],
        "vendor_logic": item["vendor_logic"], "our_method": item["our_method"],
        "units": "‰ of part length", "measured": None, "status": "NOT_SENSED",
        "uncertainty": None, "geometry": None, "note": "",
    }
    r.update(kw)
    return r


# --------------------------------------------------------------- items

def item_tip_metrics(bgr, mask, parts, gray, bl, ref_len):
    """Items 1 and 2 — toe profile arc and tip-to-throat, from the toe contour."""
    out = {}
    toe = parts.get("toe")
    it1, it2 = BY_ID["tip_length"], BY_ID["tip_center"]
    if toe is None or toe.sum() < 200 or bl is None:
        out["tip_length"] = _rec(it1, note="토 영역을 분리하지 못했습니다.")
        out["tip_center"] = _rec(it2, note="토 영역을 분리하지 못했습니다.")
        return out

    cont = M.subpixel_contour(mask, smooth=1.2)
    if cont is None or len(cont) < 80:
        out["tip_length"] = _rec(it1, note="윤곽 추출 실패")
        out["tip_center"] = _rec(it2, note="윤곽 추출 실패")
        return out

    ys, xs = np.where(mask > 0)
    x0, x1 = xs.min(), xs.max()
    toe_zone = cont[:, 0] <= x0 + 0.28 * (x1 - x0)
    if toe_zone.sum() < 40:
        out["tip_length"] = _rec(it1, note="토 윤곽 구간 부족")
        out["tip_center"] = _rec(it2, note="토 윤곽 구간 부족")
        return out

    sp = M.spline_curvature(cont, periodic=True)
    in_toe = sp["x"] <= x0 + 0.28 * (x1 - x0)
    tip_i = M.curvature_extremum(sp, sel=in_toe)
    tip_pt = np.array([sp["x"][tip_i], sp["y"][tip_i]])

    # The toe profile is the contiguous run of contour that lies inside the toe
    # zone: it enters at the sole, wraps the tip and leaves along the vamp.
    # Taking that whole run is stable; walking outward from the tip until some
    # height threshold is not — the tip itself can sit at the sole corner, and
    # the walk then terminates immediately and reports a near-zero arc.
    run = toe_run(in_toe)
    arc_px = M.arc_length(sp, run[0], run[-1]) if run is not None else 0.0
    if run is not None and tip_i not in run:
        tip_i = run[len(run) // 2]
        tip_pt = np.array([sp["x"][tip_i], sp["y"][tip_i]])

    out["tip_length"] = _rec(
        it1, measured=round(M.permille(arc_px, ref_len), 2),
        status="ADVISORY",
        geometry={"type": "arc", "tip": tip_pt.tolist(),
                  "pts": ([] if run is None else
                          np.stack([sp["x"], sp["y"]], 1)[run[0]:run[-1] + 1:6]
                          .round(1).tolist())},
        note=("측면 투영 호장입니다. 정면에서 가려지는 구간을 3D로 중합하는 "
              "원 사양과 달리 단축(foreshortening)을 풀 수 없어 참값이 아니며, "
              "골든 대비 상대 비교로만 사용합니다."))

    # tip centre: tip landmark to the bottom of the lace/eyestay opening
    lace = parts.get("collar")
    if lace is not None and lace.sum() > 100:
        lys, lxs = np.where(lace > 0)
        throat = np.array([lxs.mean(), lys.max()])
        d = float(np.linalg.norm(throat - tip_pt))
        out["tip_center"] = _rec(
            it2, measured=round(M.permille(d, ref_len), 2), status="MEASURED",
            geometry={"type": "segment", "a": tip_pt.tolist(), "b": throat.tolist()},
            note=("측면 뷰 기준 '팁~스로트' 거리입니다. 원 사양의 정면 Tip Center와는 "
                  "다른 양이므로 같은 이름으로 비교하면 안 됩니다."))
    else:
        out["tip_center"] = _rec(it2, note="아이스테이/칼라 영역 미검출")
    return out


def item_mudguard(bgr, mask, gray, bl, ref_len, which):
    """Items 4 and 5 — the measurements that genuinely work from a lateral photo."""
    item = BY_ID["forefoot_mudguard" if which == "forefoot" else "heel_mudguard"]
    if bl is None:
        return _rec(item, note="바닥선 검출 실패")
    ys, xs = np.where(mask > 0)
    x0, x1 = xs.min(), xs.max()
    w = x1 - x0
    if which == "forefoot":
        xa, xb = x0 + 0.08 * w, x0 + 0.38 * w
    else:
        xa, xb = x0 + 0.66 * w, x0 + 0.94 * w

    height = signed_height_map(mask.shape, bl)
    bm = band(mask, height, 0.045 * ref_len, 0.30 * ref_len, xa, xb)
    if bm.sum() < 400:
        return _rec(item, note="머드가드 탐색 밴드가 너무 작습니다.")

    line, pts = dominant_line_in_band(gray, bm, bl["dir"])
    if line is None:
        return _rec(item, note="머드가드 에지를 찾지 못했습니다.")

    xm = 0.5 * (xa + xb)
    ybl = bl["point"][1] + (xm - bl["point"][0]) * bl["dir"][1] / (bl["dir"][0] + 1e-9)
    yml = line["point"][1] + (xm - line["point"][0]) * line["dir"][1] / (line["dir"][0] + 1e-9)
    dist = abs(ybl - yml)

    # uncertainty: edge scatter of the mudguard fit plus the bottom-line fit
    u_px = math.hypot(line["rms"] / max(math.sqrt(line["n"]), 1),
                      bl["rms"] / max(math.sqrt(bl["n"]), 1))
    return _rec(
        item, measured=round(M.permille(dist, ref_len), 2), status="MEASURED",
        uncertainty=round(M.permille(u_px, ref_len), 3),
        geometry={"type": "gap", "x": float(xm),
                  "a": [float(xm), float(ybl)], "b": [float(xm), float(yml)],
                  "line_rms": round(line["rms"], 2), "n_pts": int(line["n"])},
        note="동일 이미지 내 두 평행 근사선의 수직 간격이라 스케일에 가장 강건한 항목입니다.")


def item_heel_height(mask, gray, bl, ref_len, parts):
    """Item 6 — intersection of the heel wall line and the sole-top line.

    Deliberately not a corner detector: an intersection of two robust lines is
    analytically more stable on a noisy outline than goodFeaturesToTrack.
    """
    item = BY_ID["heel_height"]
    if bl is None:
        return _rec(item, note="바닥선 검출 실패")
    ys, xs = np.where(mask > 0)
    x0, x1 = xs.min(), xs.max()
    w = x1 - x0
    height = signed_height_map(mask.shape, bl)
    heel = band(mask, height, 0.06 * ref_len, 0.65 * ref_len,
                x0 + 0.80 * w, x0 + 1.00 * w)
    if heel.sum() < 250:
        return _rec(item, note="힐 밴드가 너무 작습니다.")
    # heel outer wall is the near-vertical edge; use the silhouette rightmost pts
    pts = []
    for y in range(mask.shape[0]):
        row = np.where(heel[y] > 0)[0]
        if len(row):
            pts.append((row.max(), y))
    if len(pts) < 20:
        return _rec(item, note="힐 외벽 점 부족")
    p = np.array(pts, np.float64)
    wall = M.fit_line_odr(p)
    x = M.line_intersection(wall, bl)
    if x is None:
        return _rec(item, note="힐 외벽선과 바닥선이 평행에 가깝습니다.")
    top_y = p[:, 1].min()
    hgt = abs(x[1] - top_y)
    # both point sets must be the lines' OWN supports; passing the same set
    # twice fits two near-identical lines and the intersection blows up
    u = M.intersection_uncertainty(wall, bl, p, bl.get("pts", p), trials=120)
    return _rec(
        item, measured=round(M.permille(hgt, ref_len), 2), status="ADVISORY",
        uncertainty=None if not u else round(M.permille(u["radius"], ref_len), 3),
        geometry={"type": "segment", "a": [float(x[0]), float(top_y)],
                  "b": [float(x[0]), float(x[1])]},
        note=("측면 실루엣 기준 힐 프로파일 높이입니다. 원 사양은 후면(CAM4) "
              "중심선에서 재는 항목이라 동일 값이 아닙니다."))


def item_heel_overlay(mask, gray, bl, ref_len):
    """Item 7 — two oblique lines and their intersection, at the heel."""
    item = BY_ID["heel_overlay"]
    if bl is None:
        return _rec(item, note="바닥선 검출 실패")
    ys, xs = np.where(mask > 0)
    x0, x1 = xs.min(), xs.max()
    w = x1 - x0
    height = signed_height_map(mask.shape, bl)
    zone = band(mask, height, 0.05 * ref_len, 0.45 * ref_len,
                x0 + 0.62 * w, x0 + 0.98 * w)
    if zone.sum() < 300:
        return _rec(item, note="힐 오버레이 밴드 부족")
    segs = M.lsd_segments(gray, zone, min_len=16)
    if len(segs) < 2:
        return _rec(item, note="사선 2개를 찾지 못했습니다.")
    def ang(s):
        return math.degrees(math.atan2(s[3] - s[1], s[2] - s[0])) % 180
    segs = sorted(segs, key=lambda s: -math.hypot(s[2] - s[0], s[3] - s[1]))
    a = segs[0]
    b = next((s for s in segs[1:] if 12 < abs(ang(s) - ang(a)) < 168), None)
    if b is None:
        return _rec(item, note="교차하는 두 사선을 찾지 못했습니다.")
    to_line = lambda s: M.fit_line_odr(np.array(
        [[s[0], s[1]], [s[2], s[3]],
         [(s[0] + s[2]) / 2, (s[1] + s[3]) / 2]], np.float64))
    la, lb = to_line(a), to_line(b)
    x = M.line_intersection(la, lb)
    if x is None:
        return _rec(item, note="두 사선이 평행합니다.")
    hh = signed_height_map(mask.shape, bl)
    yi = int(np.clip(x[1], 0, mask.shape[0] - 1))
    xi = int(np.clip(x[0], 0, mask.shape[1] - 1))
    d = float(hh[yi, xi])
    return _rec(
        item, measured=round(M.permille(abs(d), ref_len), 2), status="ADVISORY",
        geometry={"type": "cross", "p": [float(x[0]), float(x[1])],
                  "a": list(map(float, a)), "b": list(map(float, b))},
        note=("측면 뷰에서 근사한 값입니다. 원 사양은 후면(CAM4) + 사전 원근 변조를 "
              "전제로 하며, 두 사선 교점은 평행에 가까울수록 급격히 불안정해집니다."))


def item_material(bgr, mask):
    """Items 10 and 11 — capability judgements, expressed as numbers."""
    spec = M.specular_fraction(bgr, mask)
    tex = M.texture_energy(bgr, mask)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fg = mask > 0
    dark = (g < 70) & fg
    dark_frac = float(dark.sum()) / max(int(fg.sum()), 1)

    risk = min(1.0, spec * 3.0 + min(tex / 22.0, 1.0) * 0.5)
    r10 = _rec(BY_ID["colorway_capability"],
               measured=round(100 * (1 - risk), 1), units="%",
               status="ADVISORY",
               geometry={"specular_fraction": round(spec, 4),
                         "texture_energy": round(tex, 2)},
               note=("정반사 화소 비율과 텍스처 에너지로 계산한 검사 여력입니다. "
                     "벤더 지적대로 색상보다 재질(광택·엠보싱)이 지배적입니다."))

    # Item 11 is specifically about BLACK material, so the contrast head-room
    # has to be measured inside the dark region, not over the whole shoe — a
    # white shoe otherwise scores full marks on a question about black.
    snr_all = M.edge_snr(bgr, fg.astype(np.uint8))
    snr_dark = M.edge_snr(bgr, dark.astype(np.uint8)) if dark_frac > 0.02 else 0.0
    ratio = (snr_dark / snr_all) if snr_all > 1e-6 and dark_frac > 0.02 else None
    if ratio is None:
        note = ("이 개체는 어두운 영역이 2% 미만이라 검정 소재 검사 여력을 "
                "직접 측정할 수 없습니다. 검정 컬러웨이 샘플이 필요합니다.")
        val = None
        status = "NOT_SENSED"
    else:
        val = round(100 * float(np.clip(ratio, 0, 1.2)), 1)
        note = ("어두운 영역의 에지 SNR을 같은 개체의 전체 SNR과 비교한 상대 "
                "여력입니다. 벤더는 타색상 대비 90% 수준을 예측했습니다.")
        status = "ADVISORY"
    r11 = _rec(BY_ID["black_capability"], measured=val, units="%", status=status,
               geometry={"edge_snr_all": round(snr_all, 2),
                         "edge_snr_dark": round(snr_dark, 2),
                         "dark_fraction": round(dark_frac, 3)},
               note=note)
    return r10, r11


def item_eyestay_width(bgr, mask, ref_len):
    """Item 3 — throat opening width, from the top view.

    The vendor keys on a designated colour (red lining seen between the lace
    rows). On a shaded render there is no lining colour, so the same opening is
    found by its depth instead: the throat is the recessed, darker band running
    down the middle of the shoe. Measured at three stations so left/right
    asymmetry shows up rather than being averaged away.
    """
    item = BY_ID["eyestay_width"]
    ys, xs = np.where(mask > 0)
    if len(xs) < 200:
        return _rec(item, note="실루엣 부족")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inner = cv2.erode(mask, np.ones((9, 9), np.uint8))
    v = g[inner > 0]
    if v.size < 200:
        return _rec(item, note="내부 영역 부족")
    thr = float(np.quantile(v, 0.22))          # recessed = darker
    rec_mask = ((g <= thr) & (inner > 0)).astype(np.uint8)
    rec_mask = cv2.morphologyEx(rec_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(rec_mask)
    if n < 2:
        return _rec(item, note="개구부(오목 영역)를 찾지 못했습니다.")
    # the throat is the large central component, not a shadow at the rim
    cx0 = 0.5 * (x0 + x1)
    best, score = None, -1e9
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 0.005 * mask.sum():
            continue
        s = a - 3.0 * abs(cents[i][0] - cx0) * 20
        if s > score:
            score, best = s, i
    if best is None:
        return _rec(item, note="유효한 개구부 후보가 없습니다.")
    comp = (lab == best).astype(np.uint8)
    cys, cxs = np.where(comp > 0)
    ly0, ly1 = cys.min(), cys.max()
    widths, stations = [], []
    for f in (0.30, 0.50, 0.70):
        y = int(ly0 + (ly1 - ly0) * f)
        row = np.where(comp[y] > 0)[0]
        if len(row) < 2:
            continue
        widths.append(float(row.max() - row.min()))
        stations.append([float(row.min()), float(y), float(row.max())])
    if not widths:
        return _rec(item, note="개구 폭 계측 실패")
    w = float(np.median(widths))
    spread = float(np.max(widths) - np.min(widths))
    return _rec(item, measured=round(M.permille(w, ref_len), 2), status="MEASURED",
                uncertainty=round(M.permille(spread / 2, ref_len), 2),
                geometry={"type": "stations", "rows": stations},
                note=("상부 뷰에서 오목한 스로트 영역을 분리해 3지점 개구폭의 중앙값을 "
                      "계측했습니다. 원 사양은 지정 색상(빨간 안감) 추출을 쓰지만 "
                      "렌더에는 안감 색이 없어 깊이(음영)로 대체했습니다."))


def item_heel_center(bgr, mask, ref_len):
    """Item 8 — heel centring, from the rear view.

    Offset between the silhouette's own mirror-symmetry axis and the geometric
    centre of the heel band. Left-right asymmetry is invisible in profile, so
    this item genuinely exists only in a rear or top view.
    """
    item = BY_ID["heel_center"]
    ys, xs = np.where(mask > 0)
    if len(xs) < 200:
        return _rec(item, note="실루엣 부족")
    y0, y1 = ys.min(), ys.max()
    band_m = mask.copy()
    band_m[: int(y0 + 0.35 * (y1 - y0))] = 0     # heel counter band
    b = band_m > 0
    if b.sum() < 200:
        return _rec(item, note="힐 밴드 부족")
    bxs = np.where(b.any(axis=0))[0]
    geo_c = 0.5 * (bxs.min() + bxs.max())

    # symmetry axis: the vertical line whose mirror overlaps the mask best
    best_c, best_s = geo_c, -1
    for dx in np.linspace(-0.06, 0.06, 25):
        c = geo_c + dx * (bxs.max() - bxs.min())
        sh = int(round(2 * c - mask.shape[1] + 1))
        flip = np.fliplr(b)
        flip = np.roll(flip, sh, axis=1)
        s = float((b & flip).sum())
        if s > best_s:
            best_s, best_c = s, c
    off = abs(best_c - geo_c)
    return _rec(item, measured=round(M.permille(off, ref_len), 2), status="MEASURED",
                geometry={"type": "axis", "sym_x": float(best_c),
                          "geo_x": float(geo_c),
                          "y": [float(y0 + 0.35 * (y1 - y0)), float(y1)]},
                note=("후면 실루엣의 대칭축과 힐 밴드 기하 중심의 수평 편차입니다. "
                      "0에 가까울수록 중심이 맞은 것입니다."))


def item_strobel(bgr, mask, ref_len):
    """Item 9 geometry — centreline and quarter-station widths, bottom view.

    The vendor's sequence exactly: principal axis, centreline, left/right
    extremes, quarter positions from the overall length, width at three
    stations. What we do NOT have is a strobel board: every public shoe is a
    finished shoe, so this runs on an OUTSOLE outline. The geometry transfers;
    the strobel appearance model and the red mark do not.
    """
    item = BY_ID["strobel"]
    ys, xs = np.where(mask > 0)
    if len(xs) < 300:
        return _rec(item, note="실루엣 부족")
    p = np.stack([xs, ys], 1).astype(np.float64)
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    ax, per = vt[0], vt[1]
    t = (p - c) @ ax
    t0, t1 = t.min(), t.max()

    widths, stations = [], []
    for f in (0.25, 0.50, 0.75):
        tv = t0 + (t1 - t0) * f
        sel = np.abs(t - tv) < max(1.5, 0.004 * (t1 - t0))
        if sel.sum() < 8:
            continue
        s = (p[sel] - c) @ per
        widths.append(float(s.max() - s.min()))
        a = c + ax * tv + per * s.min()
        b = c + ax * tv + per * s.max()
        stations.append([a.tolist(), b.tolist()])
    if len(widths) < 2:
        return _rec(item, note="4분할 폭 계측 실패")
    waist = widths[1] / max(0.5 * (widths[0] + widths[2]), 1e-6) if len(widths) == 3 else None
    return _rec(item, measured=round(M.permille(float(np.median(widths)), ref_len), 2),
                status="MEASURED",
                geometry={"type": "stations2", "segs": stations,
                          "widths_permille": [round(M.permille(w, ref_len), 1) for w in widths],
                          "waist_ratio": None if waist is None else round(waist, 3)},
                note=("중심축→중심선→종단 대비 4분할 3지점 폭이라는 원 사양 순서를 그대로 "
                      "따랐습니다. 다만 공개 데이터에 라스트 갑피가 없어 스트로벨 보드가 "
                      "아닌 아웃솔 외곽에서 기하만 검증했습니다. 홀 내 RED MARK 검출은 "
                      "실제 리그 촬영 없이는 검증할 수 없습니다."))


def not_sensed(item_id, view):
    it = BY_ID[item_id]
    cams = ", ".join(it["cameras"]) or "-"
    return _rec(it, status="NOT_SENSED",
                note=(f"이 항목은 {it['view']} 뷰({cams})가 필요합니다. "
                      f"현재 입력은 {view} 뷰라 계측하지 않습니다. "
                      f"알고리즘은 구현되어 있으며 해당 뷰가 들어오면 동작합니다."))


# --------------------------------------------------------------- driver

def inspect(bgr, mask, parts, view="lateral"):
    """Every item, always — measured, advisory, or explicitly not sensed."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ref_len, _, _ = M.reference_length(mask)
    bl = bottom_line(mask)
    recs = {}

    if view == "lateral":
        recs.update(item_tip_metrics(bgr, mask, parts, gray, bl, ref_len))
        recs["forefoot_mudguard"] = item_mudguard(bgr, mask, gray, bl, ref_len, "forefoot")
        recs["heel_mudguard"] = item_mudguard(bgr, mask, gray, bl, ref_len, "heel")
        recs["heel_height"] = item_heel_height(mask, gray, bl, ref_len, parts)
        recs["heel_overlay"] = item_heel_overlay(mask, gray, bl, ref_len)
        for k in ("eyestay_width", "heel_center", "strobel"):
            recs[k] = not_sensed(k, view)
    elif view == "top":
        recs["eyestay_width"] = item_eyestay_width(bgr, mask, ref_len)
        for k in ("tip_length", "tip_center", "forefoot_mudguard", "heel_mudguard",
                  "heel_height", "heel_overlay", "heel_center", "strobel"):
            recs[k] = not_sensed(k, view)
    elif view == "rear":
        recs["heel_center"] = item_heel_center(bgr, mask, ref_len)
        recs["heel_height"] = item_heel_height(mask, gray, bl, ref_len, parts)
        recs["heel_overlay"] = item_heel_overlay(mask, gray, bl, ref_len)
        for k in ("tip_length", "tip_center", "eyestay_width",
                  "forefoot_mudguard", "heel_mudguard", "strobel"):
            recs[k] = not_sensed(k, view)
    elif view == "bottom":
        recs["strobel"] = item_strobel(bgr, mask, ref_len)
        for k in ("tip_length", "tip_center", "eyestay_width", "forefoot_mudguard",
                  "heel_mudguard", "heel_height", "heel_overlay", "heel_center"):
            recs[k] = not_sensed(k, view)
    else:
        for it in ITEMS:
            if it["feasibility"] != "advisory":
                recs[it["id"]] = not_sensed(it["id"], view)

    r10, r11 = item_material(bgr, mask)
    recs["colorway_capability"] = r10
    recs["black_capability"] = r11

    ordered = [recs[it["id"]] for it in ITEMS if it["id"] in recs]
    summary = {
        "ref_len_px": round(ref_len, 1),
        "view": view,
        "n_measured": sum(1 for r in ordered if r["status"] == "MEASURED"),
        "n_advisory": sum(1 for r in ordered if r["status"] == "ADVISORY"),
        "n_not_sensed": sum(1 for r in ordered if r["status"] == "NOT_SENSED"),
        "bottom_line_rms": None if bl is None else round(bl["rms"], 2),
    }
    return ordered, summary
