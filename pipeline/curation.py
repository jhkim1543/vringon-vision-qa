# -*- coding: utf-8 -*-
"""Which library photographs contain exactly one shoe.

Every measurement assumes a single shoe: the silhouette becomes the reference
length, its lowest edge becomes the datum line, and the forward contour run
becomes the toe arc. A product photo of a PAIR breaks all three at once and the
pipeline still returns confident-looking numbers — for a shape that is two
shoes. Pairs also poison the golden statistics, which widens every tolerance.

These lists are eyeballed, not inferred. A silhouette heuristic was tried first
(solidity and top-profile hump count) and rejected: single shoes legitimately
produce two humps for the toe box and collar, and up to four on a high-contrast
black upper, while a tightly overlapping pair produces three. The metric could
not separate the classes on this library, so a curated list is the honest
answer rather than a threshold that looks principled and is not.
"""

# photographs containing more than one shoe — excluded everywhere
PAIRS = {
    "Superstar": {"01_0009", "01_0022", "01_0023", "01_0036", "01_0037"},
    "Stan-Smith": {"01_0113", "01_0124"},
    "Gazelle": {"01_0222"},
}


def is_pair(sku, key):
    return key in PAIRS.get(sku, ())


def single_only(sku, keys):
    return [k for k in keys if not is_pair(sku, k)]


def filter_files(sku, files):
    """Drop pair photographs from a list of file paths."""
    import os
    out = []
    for f in files:
        k = os.path.splitext(os.path.basename(f))[0]
        if not is_pair(sku, k):
            out.append(f)
    return out
