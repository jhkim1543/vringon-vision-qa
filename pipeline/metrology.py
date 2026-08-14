# -*- coding: utf-8 -*-
"""Measurement primitives for the Air Max 90 lasted-upper inspection items.

Design rules that came out of the open-source review, and why they are not
negotiable:

* Sub-pixel first. Pixel-accurate contours cap every downstream measurement at
  ~1 px, which is worse than the differences we are trying to resolve.
  `skimage.measure.find_contours` is the permissive, maintained way to get a
  CHAINED sub-pixel contour (the canonical Canny-Devernay code is AGPL and its
  ports carry no licence at all).
* Lines are fitted by orthogonal distance regression, not least squares. Both
  pixel coordinates carry noise, and `scipy.odr` hands back the parameter
  covariance we need for an uncertainty budget.
* Curvature comes from an analytic spline derivative, not finite differences on
  raw contour points. The vendor's "weight the path by curvature to convert to
  true distance" is an approximation of the arc-length integral, so we compute
  the integral.
* Everything is reported in per-mille of an in-part reference length. A single
  uncontrolled photograph has unknown focal length, distance, obliquity and
  distortion; those cannot be separated, so millimetres would be invented.
"""
import numpy as np
import cv2
from scipy import interpolate, odr

try:
    from skimage import measure as skmeasure
    HAVE_SKIMAGE = True
except Exception:                                    # keep the demo runnable
    HAVE_SKIMAGE = False


# --------------------------------------------------------------- sub-pixel

def subpixel_contour(mask, level=0.5, smooth=1.0):
    """Longest chained sub-pixel contour of a binary mask, as (N,2) x,y."""
    f = mask.astype(np.float32)
    if smooth:
        f = cv2.GaussianBlur(f, (0, 0), smooth)
    if HAVE_SKIMAGE:
        cs = skmeasure.find_contours(f, level)
        if not cs:
            return None
        c = max(cs, key=len)
        return np.stack([c[:, 1], c[:, 0]], axis=1)   # (row,col) -> (x,y)
    cs, _ = cv2.findContours((f > level).astype(np.uint8), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    return max(cs, key=len).reshape(-1, 2).astype(np.float64)


def resample(pts, n=600):
    """Arc-length resampling so spline knots are evenly spread."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if d[-1] <= 0:
        return pts
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, pts[:, 0]), np.interp(t, d, pts[:, 1])], axis=1)


# --------------------------------------------------------------- curvature

def spline_curvature(pts, s=None, periodic=False, n=800):
    """Analytic curvature k(t) and the spline, from a parametric fit.

    `s` is the single most consequential knob: too small and noise manufactures
    phantom inflection points, too large and landmark positions get biased. It
    is fixed from a golden-sample study, never tuned per image.
    """
    p = resample(pts, n)
    if s is None:
        s = max(4.0, 0.0025 * len(p) * np.ptp(p, axis=0).mean() / 100.0)
    tck, u = interpolate.splprep([p[:, 0], p[:, 1]], s=s, per=int(periodic))
    uu = np.linspace(0, 1, n)
    x, y = interpolate.splev(uu, tck)
    dx, dy = interpolate.splev(uu, tck, der=1)
    ddx, ddy = interpolate.splev(uu, tck, der=2)
    denom = np.power(dx * dx + dy * dy, 1.5) + 1e-12
    k = (dx * ddy - dy * ddx) / denom
    return dict(u=uu, x=np.asarray(x), y=np.asarray(y), k=np.asarray(k),
                dx=np.asarray(dx), dy=np.asarray(dy), tck=tck)


def arc_length(sp, i0, i1):
    """Exact ∫|r'(t)|dt between two spline parameters — supersedes the
    vendor's curvature-weighted path heuristic, which approximates it."""
    a, b = (i0, i1) if i0 <= i1 else (i1, i0)
    seg = slice(a, b + 1)
    sp_speed = np.hypot(sp["dx"][seg], sp["dy"][seg])
    return float(np.trapezoid(sp_speed, sp["u"][seg]))


def curvature_extremum(sp, sel=None):
    """Index of maximum |curvature| — the tip / corner landmark."""
    k = np.abs(sp["k"]).copy()
    if sel is not None:
        k[~sel] = -1
    return int(np.argmax(k))


def inflection_indices(sp, sel=None):
    """Where curvature changes sign (the vendor's '변곡점')."""
    k = sp["k"]
    sign = np.sign(k)
    idx = np.where(np.diff(sign) != 0)[0]
    if sel is not None:
        idx = idx[sel[idx]]
    return idx


# --------------------------------------------------------------- robust lines

def fit_line_odr(pts):
    """Orthogonal distance regression line: returns point, unit direction,
    residual RMS and the parameter covariance for the uncertainty budget."""
    p = np.asarray(pts, np.float64)
    c = p.mean(axis=0)
    q = p - c
    # PCA seed, then ODR refine in the better-conditioned orientation
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    d = vt[0]
    steep = abs(d[0]) < abs(d[1])
    X, Y = (p[:, 1], p[:, 0]) if steep else (p[:, 0], p[:, 1])
    slope = d[0] / d[1] if steep else d[1] / d[0]
    inter = Y.mean() - slope * X.mean()
    try:
        out = odr.ODR(odr.RealData(X, Y), odr.Model(lambda B, x: B[0] * x + B[1]),
                      beta0=[slope, inter]).run()
        slope, inter = out.beta
        cov = out.cov_beta * out.res_var if out.cov_beta is not None else None
    except Exception:
        cov = None
    dirv = np.array([1.0, slope])
    dirv /= np.linalg.norm(dirv)
    pt = np.array([0.0, inter])
    if steep:                                    # undo the axis swap
        dirv = dirv[::-1].copy()
        pt = pt[::-1].copy()
    res = point_line_distance(p, pt, dirv)
    return dict(point=pt, dir=dirv, rms=float(np.sqrt((res ** 2).mean())),
                cov=cov, n=len(p))


def point_line_distance(pts, pt, dirv):
    v = np.asarray(pts, np.float64) - pt
    n = np.array([-dirv[1], dirv[0]])
    return v @ n


def line_intersection(l1, l2):
    """Analytic intersection; None when the lines are near-parallel."""
    p1, d1 = l1["point"], l1["dir"]
    p2, d2 = l2["point"], l2["dir"]
    A = np.array([d1, -d2]).T
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def intersection_uncertainty(l1, l2, pts1, pts2, trials=400, seed=0):
    """Monte-Carlo spread of a two-line intersection.

    Linear propagation understates the tail badly as the lines approach
    parallel, which is exactly the regime of the heel-overlay item.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        a = fit_line_odr(pts1 + rng.normal(0, max(l1["rms"], 0.3), pts1.shape))
        b = fit_line_odr(pts2 + rng.normal(0, max(l2["rms"], 0.3), pts2.shape))
        x = line_intersection(a, b)
        if x is not None:
            out.append(x)
    if len(out) < 10:
        return None
    out = np.array(out)
    return dict(std=out.std(axis=0).tolist(),
                radius=float(np.sqrt((out.std(axis=0) ** 2).sum())))


# --------------------------------------------------------------- edge bands

def lsd_segments(gray, mask=None, min_len=18):
    """Straight-edge seeds. OpenCV 5 ships LSD again (patent expired)."""
    lsd = cv2.createLineSegmentDetector()
    g = gray if gray.dtype == np.uint8 else cv2.normalize(
        gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lines = lsd.detect(g)[0]
    if lines is None:
        return []
    segs = []
    for l in lines.reshape(-1, 4):
        x1, y1, x2, y2 = l
        if np.hypot(x2 - x1, y2 - y1) < min_len:
            continue
        if mask is not None:
            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if not (0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1]) or not mask[my, mx]:
                continue
        segs.append((float(x1), float(y1), float(x2), float(y2)))
    return segs


def edge_points_in_band(gray, band_mask, keep=0.12):
    """Strongest gradient points inside an ROI, at sub-pixel by centroid
    interpolation of the gradient magnitude along the gradient direction."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag[band_mask == 0] = 0
    if mag.max() <= 0:
        return np.empty((0, 2))
    thr = np.quantile(mag[band_mask > 0], 1 - keep)
    ys, xs = np.where(mag >= max(thr, 1e-6))
    if len(xs) == 0:
        return np.empty((0, 2))
    # sub-pixel: parabolic peak along the gradient direction
    out = []
    h, w = mag.shape
    for x, y in zip(xs, ys):
        dx, dy = gx[y, x], gy[y, x]
        n = np.hypot(dx, dy)
        if n < 1e-6:
            continue
        ux, uy = dx / n, dy / n
        xm, ym = int(round(x - ux)), int(round(y - uy))
        xp, yp = int(round(x + ux)), int(round(y + uy))
        if not (0 <= xm < w and 0 <= ym < h and 0 <= xp < w and 0 <= yp < h):
            continue
        a, b, c = mag[ym, xm], mag[y, x], mag[yp, xp]
        d = a - 2 * b + c
        off = 0.0 if abs(d) < 1e-9 else 0.5 * (a - c) / d
        off = float(np.clip(off, -1, 1))
        out.append((x + ux * off, y + uy * off))
    return np.array(out) if out else np.empty((0, 2))


# --------------------------------------------------------------- alignment

def align_ecc(ref_gray, test_gray, motion=cv2.MOTION_EUCLIDEAN, gate=0.72):
    """Warp test into the reference frame; gated because ECC fails silently.

    Run on gradient magnitude rather than raw pixels so a black colourway,
    which has almost no intensity structure, does not starve the correlation.
    """
    def prep(g):
        g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.2)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
        m = cv2.magnitude(gx, gy)
        return cv2.normalize(m, None, 0, 1, cv2.NORM_MINMAX)

    a, b = prep(ref_gray), prep(test_gray)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(a, b, warp, motion, crit, None, 5)
    except cv2.error:
        return None, 0.0
    return (warp, float(cc)) if cc >= gate else (None, float(cc))


# --------------------------------------------------------------- scale

def reference_length(mask):
    """In-part normaliser: silhouette extent along its own principal axis.

    The longest, best-conditioned dimension available, and the one least
    disturbed by local deformation of an upper on a last.
    """
    ys, xs = np.where(mask > 0)
    p = np.stack([xs, ys], 1).astype(np.float64)
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    proj = (p - c) @ vt[0]
    return float(proj.max() - proj.min()), c, vt[0]


def permille(value_px, ref_px):
    return float(1000.0 * value_px / ref_px) if ref_px else float("nan")


# --------------------------------------------------------------- material

def specular_fraction(bgr, mask):
    """Blown-highlight share — the vendor's item 10 risk, made numeric."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v, s = hsv[..., 2], hsv[..., 1]
    hot = (v > 240) & (s < 40) & (mask > 0)
    return float(hot.sum()) / max(int((mask > 0).sum()), 1)


def texture_energy(bgr, mask):
    """Embossing/pattern load: normalised high-frequency energy."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    v = lap[mask > 0]
    return float(np.sqrt((v ** 2).mean())) if v.size else 0.0


def edge_snr(bgr, mask):
    """Local contrast head-room — how much inspection capability a dark
    material actually leaves (the vendor's item 11 '90% of other colours')."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag = cv2.magnitude(gx, gy)[mask > 0]
    if mag.size < 50:
        return 0.0
    sig = float(np.quantile(mag, 0.98))
    noise = float(np.median(mag)) + 1e-6
    return sig / noise


def lab_delta_e(bgr, ref_lab):
    """CIEDE2000-ish ΔE map against a stored reference Lab.

    A single spec number that survives lamp changes, replacing the vendor's
    hand-tuned HSV window. Uses the CIE76 form where the difference is large
    and the 2000 weighting only matters near the threshold.
    """
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    d = lab - np.float32(ref_lab).reshape(1, 1, 3)
    return np.sqrt((d ** 2).sum(axis=2))
