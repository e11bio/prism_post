"""
How we score a segmentation against the sparse labels.

The labels only cover part of the crop, and a segmentation may leave voxels
unassigned (label 0). So we score on the voxels that are labelled in both,
and report separately how many labelled voxels the segmentation covers.
Scoring the unassigned voxels as one big "segment 0" would count them as a
giant merge, which is misleading.

Besides VOI we count false merges and false splits directly, since those are
what you actually look for when you inspect a segmentation:

* a false merge is a segment that contains more than one label, each with
  at least min_overlap voxels. We count the extra labels, so a segment with
  three labels in it is two merges
* a false split is a label spread over more than one segment, again each
  with at least min_overlap voxels. Same counting

min_overlap keeps a few stray boundary voxels from counting as a merge.

Note - the labels were drawn inside this crop, so a neuron that
leaves and re-enters the crop can have a different id on each piece. The
barcode relabel can (correctly) join such pieces, and that would score as a
merge. merge_gt_by_barcode() merges labels whose barcodes are near identical
before scoring to take that out of the numbers. It only looks at large
labels and uses a strict threshold, and the scripts print what it merged.
"""

import numpy as np
from skimage.metrics import variation_of_information

MIN_OVERLAP = 100

FIELDS = [
    "voi_split",
    "voi_merge",
    "voi_sum",
    "gt_covered",
    "false_merges",
    "false_splits",
]


def score(gt, seg, min_overlap=MIN_OVERLAP):
    gt = np.asarray(gt).ravel()
    seg = np.asarray(seg).ravel()
    labelled = gt > 0
    covered = labelled & (seg > 0)
    gt_covered = covered.sum() / max(labelled.sum(), 1)

    g, s = gt[covered], seg[covered]
    if len(g) == 0:
        return dict.fromkeys(FIELDS[:3], float("nan")) | {
            "gt_covered": 0.0,
            "false_merges": 0,
            "false_splits": 0,
        }
    split, merge = variation_of_information(g, s)

    # contingency table of (label, segment) pairs on the covered voxels
    pairs, counts = np.unique(np.stack([g, s]), axis=1, return_counts=True)
    big = counts >= min_overlap
    labels_per_segment = np.unique(pairs[1, big], return_counts=True)[1]
    segments_per_label = np.unique(pairs[0, big], return_counts=True)[1]

    return {
        "voi_split": float(split),
        "voi_merge": float(merge),
        "voi_sum": float(split + merge),
        "gt_covered": float(gt_covered),
        "false_merges": int((labels_per_segment - 1).sum()),
        "false_splits": int((segments_per_label - 1).sum()),
    }


def format_row(name, result, width=26):
    return (
        f"{name:<{width}}{result['voi_split']:>8.3f}{result['voi_merge']:>8.3f}{result['voi_sum']:>8.3f}"
        f"{result['gt_covered']:>9.3f}{result['false_merges']:>8d}{result['false_splits']:>8d}"
    )


def format_header(width=26):
    return f"{'':<{width}}{'split':>8}{'merge':>8}{'sum':>8}{'covered':>9}{'merges':>8}{'splits':>8}"


def merge_gt_by_barcode(gt, barcodes, threshold=0.95, min_size=5000):
    """
    Merge ground truth labels whose mean barcodes (over barcodes, (c, z, y, x))
    are more similar than threshold. Only labels with at least min_size voxels
    take part, small ones have noisy barcodes. Returns (new gt, merged groups).
    """
    ids, inverse = np.unique(gt, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse, minlength=len(ids)).astype(np.float64)
    means = np.stack(
        [
            np.bincount(inverse, weights=c.ravel(), minlength=len(ids)) / counts
            for c in barcodes.astype(np.float32)
        ],
        axis=1,
    )
    keep = (ids != 0) & (counts >= min_size)
    ids_k, means = ids[keep], means[keep]
    means = (means - means.mean(axis=0)) / (means.std(axis=0) + 1e-8)
    means = means - means.mean(axis=1, keepdims=True)
    means = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-8)
    sim = means @ means.T

    # union find over the similar pairs
    parent = {int(i): int(i) for i in ids_k}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(*np.where(np.triu(sim, 1) > threshold)):
        ra, rb = find(int(ids_k[a])), find(int(ids_k[b]))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    groups = {}
    for i in ids_k:
        r = find(int(i))
        if r != int(i):
            groups.setdefault(r, [r]).append(int(i))
    if not groups:
        return gt, []

    src = np.array([i for g in groups.values() for i in g[1:]], dtype=gt.dtype)
    dst = np.array([r for r, g in groups.items() for _ in g[1:]], dtype=gt.dtype)
    order = np.argsort(src)
    src, dst = src[order], dst[order]
    idx = np.searchsorted(src, gt)
    idx[idx == len(src)] = 0
    hit = src[idx] == gt
    out = gt.copy()
    out[hit] = dst[idx[hit]]
    return out, sorted(groups.values())
