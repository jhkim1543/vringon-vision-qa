# -*- coding: utf-8 -*-
"""Core QA engine: silhouette, part regions, PatchCore anomaly model.

PatchCore-style: frozen WideResNet50 layer2+layer3 patch features,
coreset-subsampled memory bank, kNN distance -> anomaly heatmap.
"""
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.neighbors import NearestNeighbors

DEVICE = "cpu"
INPUT = 384

# ---------------------------------------------------------------- silhouette

def white_bg_mask(bgr):
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    fg = ((gray < 232) | (sat > 28)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n < 2:
        return np.ones((h, w), np.uint8)
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (lab == big).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return mask

def crop_align(bgr, mask, margin=0.06):
    """Crop to silhouette bbox + margin; flip so the toe points left."""
    ys, xs = np.where(mask > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mw, mh = int((x1 - x0) * margin), int((y1 - y0) * margin)
    h, w = mask.shape
    x0, x1 = max(0, x0 - mw), min(w, x1 + mw)
    y0, y1 = max(0, y0 - mh), min(h, y1 + mh)
    bgr, mask = bgr[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    # toe = side with lower silhouette; heel/collar side is taller
    colh = mask.sum(axis=0).astype(np.float32)
    n = len(colh)
    left, right = colh[: n // 3].mean(), colh[-n // 3:].mean()
    flipped = False
    if left > right:  # tall side on left -> heel on left -> flip
        bgr, mask, flipped = bgr[:, ::-1].copy(), mask[:, ::-1].copy(), True
    return bgr, mask, flipped

# ---------------------------------------------------------------- part regions

PART_COLORS = {  # BGR, from VRINGON raw palette
    "upper":    (250, 177, 135),   # blue03 A7B1FF
    "toe":      (255, 147, 135),   # blue04-ish
    "heel":     (193, 43, 35),     # blue07
    "collar":   (232, 74, 68),     # blue06
    "midsole":  (156, 221, 36),    # green03
    "outsole":  (84, 132, 0),      # green07
    "cement_boundary": (92, 171, 255),  # orange03
}

def part_regions(mask):
    """Rule-based lateral-view region split from the silhouette.

    Per column: outsole = bottom 11% of that column's fg span, midsole next 9%.
    Toe/heel/collar carved from the upper by x/y extents.
    """
    h, w = mask.shape
    parts = {k: np.zeros((h, w), np.uint8) for k in PART_COLORS}
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return parts
    gx0, gx1 = xs.min(), xs.max()
    gy0, gy1 = ys.min(), ys.max()
    bw = gx1 - gx0 + 1

    for x in range(gx0, gx1 + 1):
        col = np.where(mask[:, x] > 0)[0]
        if len(col) < 4:
            continue
        top, bot = col.min(), col.max()
        span = bot - top + 1
        o = max(3, int(span * 0.11))
        m = max(3, int(span * 0.09))
        parts["outsole"][bot - o + 1: bot + 1, x] = 1
        parts["midsole"][bot - o - m + 1: bot - o + 1, x] = 1
        parts["upper"][top: bot - o - m + 1, x] = 1

    up = parts["upper"]
    uys, uxs = np.where(up > 0)
    if len(uys):
        # toe: leftmost 20% of upper, lower half
        toe_x = gx0 + int(bw * 0.20)
        toe = up.copy(); toe[:, toe_x:] = 0
        umid = int(uys.min() + (uys.max() - uys.min()) * 0.45)
        toe[:umid, :] = 0
        parts["toe"] = toe
        # heel: rightmost 16%
        heel_x = gx1 - int(bw * 0.16)
        heel = up.copy(); heel[:, :heel_x] = 0
        parts["heel"] = heel
        # collar: top 22% rows of upper, right 55%
        cy = int(uys.min() + (uys.max() - uys.min()) * 0.22)
        col_r = up.copy(); col_r[cy:, :] = 0
        col_r[:, : gx0 + int(bw * 0.45)] = 0
        parts["collar"] = col_r
        parts["upper"] = up & ~(toe | heel | col_r)

    # cement boundary: band around upper/midsole interface
    mid_edge = cv2.dilate(parts["midsole"], np.ones((9, 9), np.uint8))
    up_all = (parts["upper"] | parts["toe"] | parts["heel"]).astype(np.uint8)
    up_edge = cv2.dilate(up_all, np.ones((9, 9), np.uint8))
    parts["cement_boundary"] = (mid_edge & up_edge) & mask
    return parts

def part_of_point(parts, x, y):
    order = ["cement_boundary", "toe", "heel", "collar", "midsole", "outsole", "upper"]
    for p in order:
        if parts[p][y, x]:
            return p
    return "upper"

# ---------------------------------------------------------------- patchcore

_model = None

def _backbone():
    global _model
    if _model is None:
        m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        m.eval()
        _model = m
    return _model

_feats = {}

def _hook(name):
    def fn(mod, i, o):
        _feats[name] = o
    return fn

_hooked = False

def _ensure_hooks():
    global _hooked
    m = _backbone()
    if not _hooked:
        m.layer2.register_forward_hook(_hook("l2"))
        m.layer3.register_forward_hook(_hook("l3"))
        _hooked = True

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

@torch.no_grad()
def patch_features(bgr):
    """Return (C, Hp, Wp) patch feature map at layer2 resolution (INPUT/8)."""
    _ensure_hooks()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
    x = (x - MEAN) / STD
    _backbone()(x)
    f2, f3 = _feats["l2"], _feats["l3"]
    f2 = F.avg_pool2d(f2, 3, 1, 1)
    f3 = F.avg_pool2d(f3, 3, 1, 1)
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([f2, f3], dim=1)[0]           # (1536, 40, 40)
    return f

def to_patches(f, fg_mask=None):
    C, H, W = f.shape
    v = f.reshape(C, -1).T.numpy()               # (H*W, C)
    if fg_mask is not None:
        m = cv2.resize(fg_mask, (W, H), interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
        return v[m], m
    return v, np.ones(H * W, bool)

# SAHI-style views: full image + 2x2 overlapping tiles (fractions of the crop)
TILES = [(0.0, 0.0, 1.0, 1.0),
         (0.0, 0.0, 0.58, 0.58), (0.42, 0.0, 1.0, 0.58),
         (0.0, 0.42, 0.58, 1.0), (0.42, 0.42, 1.0, 1.0)]

def view_crops(bgr, mask=None):
    """Yield (bgr_crop, mask_crop, box_px) for each SAHI view."""
    h, w = bgr.shape[:2]
    out = []
    for fx0, fy0, fx1, fy1 in TILES:
        x0, y0 = int(fx0 * w), int(fy0 * h)
        x1, y1 = int(fx1 * w), int(fy1 * h)
        mc = mask[y0:y1, x0:x1] if mask is not None else None
        out.append((bgr[y0:y1, x0:x1], mc, (x0, y0, x1, y1)))
    return out

@torch.no_grad()
def multi_view_patches(bgr, mask=None):
    """Patch vectors from all SAHI views, foreground-filtered."""
    vs = []
    for crop, mc, _ in view_crops(bgr, mask):
        f = patch_features(crop)
        v, _ = to_patches(f, mc)
        vs.append(v)
    return np.concatenate(vs, axis=0)

# ------------------------------------------------- within-image self-reference

def self_ref_map(bgr, parts, mask, k_frac=0.06, out_size=None):
    """Anomaly map from within-part self-similarity — no reference images.

    A stain/scuff/thread is anomalous because it differs from the REST OF ITS
    OWN PART, not because it differs from other shoes. Scoring each patch by
    its k-th nearest neighbour among same-part patches (k large enough that a
    small defect cluster cannot support itself) cancels the pose, lighting and
    unit-to-unit variation that swamps a cross-image memory bank.
    """
    f = patch_features(bgr)
    C, H, W = f.shape
    v = f.reshape(C, -1).T.numpy()
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

    # merge the fine parts into coarse zones with homogeneous appearance
    zones = {
        "upper": parts["upper"] | parts["collar"] | parts["heel"] | parts["toe"],
        "midsole": parts["midsole"],
        "outsole": parts["outsole"],
    }
    z = np.full(H * W, -1.0, np.float32)
    for zm in zones.values():
        er = max(3, int(min(zm.shape) * 0.012)) | 1
        zme = cv2.erode(zm, np.ones((er, er), np.uint8))
        g = cv2.resize(zme, (W, H), interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
        n = int(g.sum())
        if n < 40:
            continue
        sub = v[g]
        k = max(3, int(n * k_frac))
        nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(sub)
        d, _ = nn.kneighbors(sub)
        d = d[:, -1]                       # k-th NN distance (self excluded)
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med))) * 1.4826
        z[g] = (d - med) / max(mad, 1e-6)
    zm2 = z.reshape(H, W)
    h, w = mask.shape
    zf = cv2.resize(zm2, (w, h), interpolation=cv2.INTER_CUBIC)
    zf = cv2.GaussianBlur(zf, (0, 0), sigmaX=1.2)
    zf = zf * (mask > 0) + (-1.0) * (mask == 0)
    if out_size is not None and out_size != (w, h):
        zf = cv2.resize(zf, out_size, interpolation=cv2.INTER_CUBIC)
    return zf


class PatchCore:
    """Multi-view (SAHI-style) PatchCore with LOO quantile threshold."""

    def __init__(self, coreset_ratio=0.25, n_neighbors=1, seed=0, use_tiles=True):
        self.ratio = coreset_ratio
        self.k = n_neighbors
        self.rng = np.random.default_rng(seed)
        self.use_tiles = use_tiles
        self.nn = None
        self.cal_mu = self.cal_sigma = None
        self.det_thr = None   # detection threshold in robust-z units

    def _views(self, bgr, mask):
        views = view_crops(bgr, mask)
        return views if self.use_tiles else views[:1]

    def fit(self, ref_bgrs, ref_masks=None):
        # per ref: list of (vectors, keep_mask, (Hp, Wp), box_px), plus crop hw + fg mask
        ref_views, banks = [], []
        for i, bgr in enumerate(ref_bgrs):
            m = ref_masks[i] if ref_masks else None
            views = []
            for c, mc, box in self._views(bgr, m):
                f = patch_features(c)
                v, keep = to_patches(f, mc)
                views.append((v, keep, f.shape[-2:], box))
            ref_views.append((views, bgr.shape[:2], m))
            banks.append(np.concatenate([v for v, *_ in views], axis=0))
        bank = np.concatenate(banks, axis=0)
        n = len(bank)
        keep_n = max(3000, int(n * self.ratio))
        if keep_n < n:
            idx = self.rng.choice(n, keep_n, replace=False)
            bank = bank[idx]
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(bank)

        # leave-one-out: score each ref's patches against the OTHER refs
        loo_dists = []   # per ref: list of per-view distance vectors
        for i in range(len(banks)):
            others = [b for j, b in enumerate(banks) if j != i] or [banks[i]]
            ob = np.concatenate(others, axis=0)
            if len(ob) > 9000:
                ob = ob[self.rng.choice(len(ob), 9000, replace=False)]
            loo = NearestNeighbors(n_neighbors=self.k).fit(ob)
            per_view = []
            for v, keep, hw, box in ref_views[i][0]:
                d, _ = loo.kneighbors(v)
                per_view.append(d.mean(axis=1))
            loo_dists.append(per_view)
        allp = np.concatenate([d for pv in loo_dists for d in pv])
        med = float(np.median(allp))
        mad = float(np.median(np.abs(allp - med))) * 1.4826
        self.cal_mu, self.cal_sigma = med, max(mad, 1e-6)

        # calibrate the threshold in FUSED-HEATMAP space: rebuild each ref's
        # LOO heatmap through the exact test-time path (z-norm, max-fuse, blur)
        ref_maxes = []
        for i, (views, hw, m) in enumerate(ref_views):
            per_view = []
            for (v, keep, (Hp, Wp), box), d in zip(views, loo_dists[i]):
                grid = np.zeros(Hp * Wp, np.float32)
                grid[keep] = (d - self.cal_mu) / self.cal_sigma
                per_view.append((grid.reshape(Hp, Wp), box))
            acc = self._fuse(per_view, hw)
            fg = acc[m > 0] if m is not None else acc.reshape(-1)
            if fg.size:
                ref_maxes.append(float(fg.max()))
        self.ref_maxes = ref_maxes
        # LOO peaks are biased high vs. test conditions (a test image keeps its
        # near-twin refs in the bank; each LOO ref loses its own patches), so
        # the min over refs is the closest analog of a well-matched normal
        peak = min(ref_maxes) if ref_maxes else 3.0
        self.det_thr = max(1.2, peak * 1.06)
        return self

    @staticmethod
    def _fuse(per_view, hw):
        """Max-fuse per-view z grids into crop space + blur (test-time path)."""
        h, w = hw
        acc = np.full((h, w), -1e9, np.float32)
        for z_grid, (x0, y0, x1, y1) in per_view:
            zr = cv2.resize(z_grid, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
            region = acc[y0:y1, x0:x1]
            np.maximum(region, zr, out=region)
        return cv2.GaussianBlur(acc, (0, 0), sigmaX=1.0)

    def _score_view(self, crop):
        f = patch_features(crop)
        C, H, W = f.shape
        v, _ = to_patches(f, None)
        d, _ = self.nn.kneighbors(v)
        return d.mean(axis=1).reshape(H, W)

    def heatmap(self, bgr, fg_mask=None, out_size=None):
        """Robust-z anomaly map at full crop resolution (max-fused over views)."""
        h, w = bgr.shape[:2]
        per_view = []
        for crop, _, box in self._views(bgr, fg_mask):
            s = self._score_view(crop)
            per_view.append(((s - self.cal_mu) / self.cal_sigma, box))
        acc = self._fuse(per_view, (h, w))
        if fg_mask is not None:
            acc = acc * (fg_mask > 0) + (-1.0) * (fg_mask == 0)
        if out_size is not None and out_size != (w, h):
            acc = cv2.resize(acc, out_size, interpolation=cv2.INTER_CUBIC)
        return acc
